package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"transcript/transcription"
	"transcript/verification"
)

// Handler serves HTTP endpoints for running benchmarks.
type Handler struct {
	AudioService   transcription.TranscriptionService
	MinutesService transcription.MeetingMinutesService
	Verifier       *verification.Verifier
	TestDataDir    string
}

// NewHandler creates a new benchmark Handler.
func NewHandler(
	audioSvc transcription.TranscriptionService,
	momSvc transcription.MeetingMinutesService,
	verifier *verification.Verifier,
) *Handler {
	// Locate testdata directory relative to current working directory
	testDataDir := filepath.Join("benchmark", "testdata")
	if _, err := os.Stat(testDataDir); os.IsNotExist(err) {
		// Fallback if running from root
		testDataDir = filepath.Join("backend", "benchmark", "testdata")
	}

	return &Handler{
		AudioService:   audioSvc,
		MinutesService: momSvc,
		Verifier:       verifier,
		TestDataDir:    testDataDir,
	}
}

// HandleRunBenchmark handles POST /api/benchmark/run
// Supports multipart/form-data upload of multiple files with difficulty tier assignment.
func (h *Handler) HandleRunBenchmark(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 500MB max memory for bulk audio uploads
	if err := r.ParseMultipartForm(500 << 20); err != nil {
		http.Error(w, fmt.Sprintf("Failed to parse multipart form: %v", err), http.StatusBadRequest)
		return
	}

	auditMode := r.FormValue("audit_mode")
	if auditMode == "" {
		auditMode = "fable"
	}

	workersStr := r.FormValue("workers")
	workers := 8
	if workersStr != "" {
		if wVal, err := strconv.Atoi(workersStr); err == nil && wVal > 0 {
			workers = wVal
		}
	}

	// Global or per-file difficulty
	globalDifficulty := r.FormValue("difficulty") // "auto", "Easy", "Medium", "Hard"

	// Parse optional difficulties map (JSON format: {"filename.txt": "Easy", ...})
	difficultiesMap := make(map[string]string)
	if diffJson := r.FormValue("difficulties"); diffJson != "" {
		_ = json.Unmarshal([]byte(diffJson), &difficultiesMap)
	}

	formFiles := r.MultipartForm.File["files"]
	if len(formFiles) == 0 {
		http.Error(w, "No files uploaded. Provide files in 'files' multipart field.", http.StatusBadRequest)
		return
	}

	var inputs []BenchmarkItemInput
	for _, fh := range formFiles {
		f, err := fh.Open()
		if err != nil {
			log.Printf("[BenchmarkHandler] Failed to open uploaded file %s: %v", fh.Filename, err)
			continue
		}
		data, err := io.ReadAll(f)
		_ = f.Close()
		if err != nil {
			log.Printf("[BenchmarkHandler] Failed to read uploaded file %s: %v", fh.Filename, err)
			continue
		}

		baseName := filepath.Base(fh.Filename)
		itemDiff := globalDifficulty
		if val, ok := difficultiesMap[fh.Filename]; ok && val != "" {
			itemDiff = val
		} else if val, ok := difficultiesMap[baseName]; ok && val != "" {
			itemDiff = val
		} else {
			for k, v := range difficultiesMap {
				if strings.EqualFold(k, fh.Filename) || strings.EqualFold(k, baseName) {
					itemDiff = v
					break
				}
			}
		}

		inputs = append(inputs, BenchmarkItemInput{
			Filename:   fh.Filename,
			Content:    data,
			Difficulty: itemDiff,
		})
	}

	if len(inputs) == 0 {
		http.Error(w, "No valid files could be read for benchmarking", http.StatusBadRequest)
		return
	}

	runner := NewBulkBenchmarkRunner(h.AudioService, h.MinutesService, h.Verifier, workers, auditMode)
	benchCtx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	summary, err := runner.RunInputs(benchCtx, inputs)
	if err != nil {
		http.Error(w, fmt.Sprintf("Benchmark execution failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Automatically archive reports to reports directory
	h.archiveReports(summary)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(summary)
}

// HandleRunSamples handles POST /api/benchmark/samples
// Executes benchmark directly on curated sample datasets (Easy, Medium, Hard).
func (h *Handler) HandleRunSamples(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method == http.MethodGet {
		// List available samples
		h.listSamples(w, r)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	auditMode := r.URL.Query().Get("audit_mode")
	if auditMode == "" {
		auditMode = "fable"
	}

	workers := 8
	if wStr := r.URL.Query().Get("workers"); wStr != "" {
		if wVal, err := strconv.Atoi(wStr); err == nil && wVal > 0 {
			workers = wVal
		}
	}

	runner := NewBulkBenchmarkRunner(h.AudioService, h.MinutesService, h.Verifier, workers, auditMode)
	benchCtx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	summary, err := runner.RunFolder(benchCtx, h.TestDataDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to run sample benchmark: %v", err), http.StatusInternalServerError)
		return
	}

	h.archiveReports(summary)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(summary)
}

func (h *Handler) archiveReports(summary *BulkBenchmarkSummary) {
	if summary == nil {
		return
	}
	reportsDir := filepath.Join("benchmark", "reports")
	if _, err := os.Stat(reportsDir); os.IsNotExist(err) {
		reportsDir = filepath.Join("backend", "benchmark", "reports")
	}
	_ = os.MkdirAll(reportsDir, 0755)

	timestamp := time.Now().Format("20060102_150405")
	csvPath := filepath.Join(reportsDir, fmt.Sprintf("benchmark_%s.csv", timestamp))
	jsonPath := filepath.Join(reportsDir, fmt.Sprintf("benchmark_%s.json", timestamp))

	_ = summary.ExportCSV(csvPath)
	_ = summary.ExportJSON(jsonPath)
	log.Printf("[Benchmark] Auto-archived reports to %s and %s", csvPath, jsonPath)
}

type SampleItemInfo struct {
	Filename   string `json:"filename"`
	Difficulty string `json:"difficulty"`
	SizeBytes  int64  `json:"size_bytes"`
	Summary    string `json:"summary"`
}

func (h *Handler) listSamples(w http.ResponseWriter, r *http.Request) {
	entries, err := os.ReadDir(h.TestDataDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list testdata: %v", err), http.StatusInternalServerError)
		return
	}

	samples := make([]SampleItemInfo, 0)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}

		filePath := filepath.Join(h.TestDataDir, e.Name())
		content, _ := os.ReadFile(filePath)
		diff := DetectDifficulty("", e.Name(), string(content), false)

		var desc string
		switch diff {
		case TierEasy:
			desc = "Concise meeting (2-3 speakers), clear action items & explicit tags."
		case TierMedium:
			desc = "Standard cross-functional review (4-5 speakers), decisions & tradeoffs."
		case TierHard:
			desc = "Complex executive meeting (6+ speakers), absent members, rapid dialogue."
		}

		samples = append(samples, SampleItemInfo{
			Filename:   e.Name(),
			Difficulty: diff,
			SizeBytes:  info.Size(),
			Summary:    desc,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(samples)
}

// SavedReportSummary represents a summarized historical benchmark report for the frontend.
type SavedReportSummary struct {
	Filename       string                 `json:"filename"`
	Timestamp      string                 `json:"timestamp"`
	TotalFiles     int                    `json:"total_files"`
	PassRate       float64                `json:"pass_rate"`
	AverageScore   float64                `json:"average_score"`
	AverageLatency float64                `json:"average_latency"`
	Tiers          map[string]*TierMetric `json:"tiers"`
	SizeBytes      int64                  `json:"size_bytes"`
}

// HandleReports handles GET and DELETE on /api/benchmark/reports
func (h *Handler) HandleReports(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusOK)
		return
	}

	reportsDir := filepath.Join("benchmark", "reports")
	if _, err := os.Stat(reportsDir); os.IsNotExist(err) {
		reportsDir = filepath.Join("backend", "benchmark", "reports")
	}

	if r.Method == http.MethodGet {
		fileQuery := r.URL.Query().Get("file")
		if fileQuery != "" {
			cleanName := filepath.Base(fileQuery)
			filePath := filepath.Join(reportsDir, cleanName)
			data, err := os.ReadFile(filePath)
			if err != nil {
				http.Error(w, "Report file not found", http.StatusNotFound)
				return
			}
			if strings.HasSuffix(cleanName, ".csv") {
				w.Header().Set("Content-Type", "text/csv")
				w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", cleanName))
			} else {
				w.Header().Set("Content-Type", "application/json")
			}
			w.Write(data)
			return
		}

		entries, err := os.ReadDir(reportsDir)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte("[]"))
			return
		}

		reports := make([]SavedReportSummary, 0)
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			filePath := filepath.Join(reportsDir, e.Name())
			data, err := os.ReadFile(filePath)
			if err != nil {
				continue
			}
			var s BulkBenchmarkSummary
			if err := json.Unmarshal(data, &s); err == nil {
				reports = append(reports, SavedReportSummary{
					Filename:       e.Name(),
					Timestamp:      info.ModTime().Format("2006-01-02 15:04:05"),
					TotalFiles:     s.TotalFiles,
					PassRate:       s.PassRate,
					AverageScore:   s.AverageScore,
					AverageLatency: s.AverageLatencySec,
					Tiers:          s.Tiers,
					SizeBytes:      info.Size(),
				})
			}
		}

		sort.Slice(reports, func(i, j int) bool {
			return reports[i].Filename > reports[j].Filename
		})

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(reports)
		return
	}

	if r.Method == http.MethodDelete {
		fileQuery := r.URL.Query().Get("file")
		if fileQuery == "" {
			http.Error(w, "file query param required", http.StatusBadRequest)
			return
		}
		cleanName := filepath.Base(fileQuery)
		filePath := filepath.Join(reportsDir, cleanName)
		_ = os.Remove(filePath)
		if strings.HasSuffix(cleanName, ".json") {
			csvName := strings.TrimSuffix(cleanName, ".json") + ".csv"
			_ = os.Remove(filepath.Join(reportsDir, csvName))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"deleted"}`))
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}
