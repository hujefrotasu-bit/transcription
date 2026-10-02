package benchmark

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"transcript/transcription"
	"transcript/verification"
)

// Difficulty levels for benchmark testing
const (
	TierEasy   = "Easy"
	TierMedium = "Medium"
	TierHard   = "Hard"
)

// BenchmarkItemInput represents an item queued for evaluation.
type BenchmarkItemInput struct {
	Filename   string `json:"filename"`
	Content    []byte `json:"content,omitempty"`
	FilePath   string `json:"file_path,omitempty"`
	Difficulty string `json:"difficulty,omitempty"` // "Easy", "Medium", "Hard", or "" ("auto")
}

// BenchmarkItemResult holds the evaluation metrics for a single processed file.
type BenchmarkItemResult struct {
	Filename         string                    `json:"filename"`
	FileType         string                    `json:"file_type"` // "transcript" or "audio"
	Difficulty       string                    `json:"difficulty"` // "Easy", "Medium", "Hard"
	Duration         time.Duration             `json:"duration"`
	DurationSeconds  float64                   `json:"duration_seconds"`
	AuditScore       float64                   `json:"audit_score"`
	AttendeesCount   int                       `json:"attendees_count"`
	Attendees        []string                  `json:"attendees"`
	ActionItemsCount int                       `json:"action_items_count"`
	DecisionsCount   int                       `json:"decisions_count"`
	DiscussionsCount int                       `json:"discussions_count"`
	IssuesCount      int                       `json:"issues_count"`
	ErrorsCount      int                       `json:"errors_count"`
	Status           string                    `json:"status"` // "PASS" (>=80), "WARN" (70-79), "FAIL" (<70 or err)
	ErrorMessage     string                    `json:"error_message,omitempty"`
	AuditReport      *verification.AuditResult `json:"audit_report,omitempty"`
}

// TierMetric aggregates benchmark performance for a specific difficulty level.
type TierMetric struct {
	Tier              string  `json:"tier"` // "Easy", "Medium", "Hard"
	TotalFiles        int     `json:"total_files"`
	PassedFiles       int     `json:"passed_files"`
	WarnFiles         int     `json:"warn_files"`
	FailedFiles       int     `json:"failed_files"`
	PassRate          float64 `json:"pass_rate"`
	AverageScore      float64 `json:"average_score"`
	AverageLatencySec float64 `json:"average_latency_sec"`
	TotalAttendees    int     `json:"total_attendees"`
	TotalActionItems  int     `json:"total_action_items"`
	TotalDecisions    int     `json:"total_decisions"`
	TotalDiscussions  int     `json:"total_discussions"`
	TotalErrors       int     `json:"total_errors"`
}

// BulkBenchmarkSummary aggregates metrics across all processed files and difficulty tiers.
type BulkBenchmarkSummary struct {
	TotalFiles        int                    `json:"total_files"`
	PassedFiles       int                    `json:"passed_files"`
	WarnFiles         int                    `json:"warn_files"`
	FailedFiles       int                    `json:"failed_files"`
	PassRate          float64                `json:"pass_rate"`
	AverageScore      float64                `json:"average_score"`
	AverageLatencySec float64                `json:"average_latency_sec"`
	TotalDurationSec  float64                `json:"total_duration_sec"`
	Tiers             map[string]*TierMetric `json:"tiers"`
	AnalysisNotes     []string               `json:"analysis_notes"`
	Results           []BenchmarkItemResult  `json:"results"`
}

// BulkBenchmarkRunner orchestrates parallel benchmarking for transcripts and audio files.
type BulkBenchmarkRunner struct {
	AudioService   transcription.TranscriptionService
	MinutesService transcription.MeetingMinutesService
	Verifier       *verification.Verifier
	Workers        int
	AuditMode      string // "fast", "fable", or "none"
}

// NewBulkBenchmarkRunner initializes a runner with the given services, worker count, and audit mode.
func NewBulkBenchmarkRunner(
	audioSvc transcription.TranscriptionService,
	momSvc transcription.MeetingMinutesService,
	verifier *verification.Verifier,
	workers int,
	auditMode string,
) *BulkBenchmarkRunner {
	if workers <= 0 {
		workers = 4
	}
	auditMode = strings.ToLower(strings.TrimSpace(auditMode))
	if auditMode == "" {
		auditMode = "fable"
	}
	return &BulkBenchmarkRunner{
		AudioService:   audioSvc,
		MinutesService: momSvc,
		Verifier:       verifier,
		Workers:        workers,
		AuditMode:      auditMode,
	}
}

// RunFolder discovers all audio and transcript files in the given directory and benchmarks them concurrently.
func (r *BulkBenchmarkRunner) RunFolder(ctx context.Context, folderPath string) (*BulkBenchmarkSummary, error) {
	files, err := os.ReadDir(folderPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read benchmark directory: %w", err)
	}

	var inputs []BenchmarkItemInput
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(f.Name()))
		if isSupportedBenchmarkFile(ext) {
			filePath := filepath.Join(folderPath, f.Name())
			inputs = append(inputs, BenchmarkItemInput{
				Filename: f.Name(),
				FilePath: filePath,
			})
		}
	}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("no supported audio or transcript files found in %s", folderPath)
	}

	return r.RunInputs(ctx, inputs)
}

// RunFiles runs the benchmark against a specific list of file paths.
func (r *BulkBenchmarkRunner) RunFiles(ctx context.Context, filePaths []string) (*BulkBenchmarkSummary, error) {
	var inputs []BenchmarkItemInput
	for _, p := range filePaths {
		inputs = append(inputs, BenchmarkItemInput{
			Filename: filepath.Base(p),
			FilePath: p,
		})
	}
	return r.RunInputs(ctx, inputs)
}

// RunInputs runs the benchmark against in-memory or on-disk inputs in parallel.
func (r *BulkBenchmarkRunner) RunInputs(ctx context.Context, inputs []BenchmarkItemInput) (*BulkBenchmarkSummary, error) {
	startTime := time.Now()
	total := len(inputs)
	log.Printf("[Benchmark] Starting bulk benchmark on %d files with %d parallel workers (mode=%s)...", total, r.Workers, r.AuditMode)

	type job struct {
		index int
		input BenchmarkItemInput
	}

	jobs := make(chan job, total)
	resultsChan := make(chan BenchmarkItemResult, total)

	var wg sync.WaitGroup

	// Start worker pool
	for w := 0; w < r.Workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := range jobs {
				res := r.processSingleInput(ctx, j.input)
				resultsChan <- res
			}
		}(w)
	}

	// Queue all jobs
	for i, inp := range inputs {
		jobs <- job{index: i + 1, input: inp}
	}
	close(jobs)

	// Wait for workers in background then close results channel
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	var results []BenchmarkItemResult
	completed := 0

	for res := range resultsChan {
		completed++
		statusColor := "PASS"
		if res.Status == "FAIL" {
			statusColor = "FAIL"
		} else if res.Status == "WARN" {
			statusColor = "WARN"
		}

		log.Printf("[Benchmark] [%d/%d] %-28s | %-6s | Tier: %-6s | Score: %5.1f | Time: %5.1fs | Attendees: %d | Status: %s",
			completed, total, filepath.Base(res.Filename), res.FileType, res.Difficulty, res.AuditScore, res.DurationSeconds, res.AttendeesCount, statusColor)
		results = append(results, res)
	}

	// Sort results by filename for predictable output
	sort.Slice(results, func(i, j int) bool {
		return results[i].Filename < results[j].Filename
	})

	totalDuration := time.Since(startTime).Seconds()
	summary := r.computeSummary(results, totalDuration)
	return summary, nil
}

func (r *BulkBenchmarkRunner) processSingleInput(ctx context.Context, input BenchmarkItemInput) BenchmarkItemResult {
	start := time.Now()
	filename := input.Filename
	ext := strings.ToLower(filepath.Ext(filename))
	isAudio := isAudioFile(ext)

	fileType := "transcript"
	if isAudio {
		fileType = "audio"
	}

	// Determine file content
	var fileBytes []byte
	if len(input.Content) > 0 {
		fileBytes = input.Content
	} else if input.FilePath != "" {
		content, err := os.ReadFile(input.FilePath)
		if err != nil {
			return BenchmarkItemResult{
				Filename:        filename,
				FileType:        fileType,
				Difficulty:      TierMedium,
				Duration:        time.Since(start),
				DurationSeconds: time.Since(start).Seconds(),
				Status:          "FAIL",
				ErrorMessage:    fmt.Sprintf("Failed to read file: %v", err),
			}
		}
		fileBytes = content
	}

	res := BenchmarkItemResult{
		Filename:  filename,
		FileType:  fileType,
		Attendees: make([]string, 0),
	}

	var transcript string

	if isAudio {
		if r.AudioService == nil {
			res.Status = "FAIL"
			res.ErrorMessage = "Audio service not configured"
			res.Duration = time.Since(start)
			res.DurationSeconds = res.Duration.Seconds()
			return res
		}

		var reader io.Reader
		if len(fileBytes) > 0 {
			reader = bytes.NewReader(fileBytes)
		} else if input.FilePath != "" {
			f, err := os.Open(input.FilePath)
			if err != nil {
				res.Status = "FAIL"
				res.ErrorMessage = fmt.Sprintf("Failed to open audio: %v", err)
				res.Duration = time.Since(start)
				res.DurationSeconds = res.Duration.Seconds()
				return res
			}
			defer f.Close()
			reader = f
		} else {
			res.Status = "FAIL"
			res.ErrorMessage = "No audio data provided"
			res.Duration = time.Since(start)
			res.DurationSeconds = res.Duration.Seconds()
			return res
		}

		t, err := r.AudioService.Transcribe(ctx, reader, filename)
		if err != nil {
			res.Status = "FAIL"
			res.ErrorMessage = fmt.Sprintf("Transcription failed: %v", err)
			res.Duration = time.Since(start)
			res.DurationSeconds = res.Duration.Seconds()
			return res
		}
		transcript = t
	} else {
		// Text transcript
		transcript = string(fileBytes)

		// Resolve speakers on raw text only if not already tagged with speaker names
		if !hasSpeakerTags(transcript) && r.AudioService != nil {
			resolved := r.AudioService.ResolveSpeakers(ctx, transcript)
			if strings.TrimSpace(resolved) != "" && len(resolved) >= len(transcript)/2 {
				transcript = strings.TrimSpace(resolved)
			}
		}
	}

	if strings.TrimSpace(transcript) == "" {
		res.Status = "FAIL"
		res.ErrorMessage = "Empty transcript"
		res.Duration = time.Since(start)
		res.DurationSeconds = res.Duration.Seconds()
		return res
	}

	// Accurately detect or assign difficulty on real transcript text
	res.Difficulty = DetectDifficulty(input.Difficulty, filename, transcript, isAudio)

	// Extract MoM
	if r.MinutesService == nil {
		res.Status = "FAIL"
		res.ErrorMessage = "Minutes service not configured"
		res.Duration = time.Since(start)
		res.DurationSeconds = res.Duration.Seconds()
		return res
	}

	mom, _, _, mErr := r.MinutesService.ExtractMeetingMinutes(ctx, transcript)
	if mErr != nil {
		res.Status = "FAIL"
		res.ErrorMessage = fmt.Sprintf("MoM extraction error: %v", mErr)
		res.Duration = time.Since(start)
		res.DurationSeconds = res.Duration.Seconds()
		return res
	}

	res.AttendeesCount = len(mom.Attendees)
	for _, a := range mom.Attendees {
		res.Attendees = append(res.Attendees, a.Name)
	}
	res.ActionItemsCount = len(mom.ActionItems)
	res.DecisionsCount = len(mom.Decisions)
	res.DiscussionsCount = len(mom.DiscussionPoints)
	res.IssuesCount = len(mom.RisksIssuesDependencies)

	// Audit Step based on AuditMode
	switch r.AuditMode {
	case "none":
		score := 100.0
		if res.AttendeesCount == 0 {
			score -= 20
		}
		if res.ActionItemsCount == 0 && res.DecisionsCount == 0 {
			score -= 15
		}
		if res.DiscussionsCount == 0 {
			score -= 15
		}
		res.AuditScore = score
		if score >= 80 {
			res.Status = "PASS"
		} else {
			res.Status = "WARN"
		}

	case "fast":
		// Fast programmatic rubric validation (<0.01s):
		score := 100.0
		lowerT := strings.ToLower(transcript)
		for _, a := range res.Attendees {
			lowerA := strings.ToLower(strings.TrimSpace(a))
			if lowerA == "" || !strings.Contains(lowerT, lowerA) {
				score -= 5.0
				res.ErrorsCount++
			}
		}
		if res.AttendeesCount == 0 {
			score -= 20.0
			res.ErrorsCount++
		}
		if res.DiscussionsCount == 0 {
			score -= 15.0
			res.ErrorsCount++
		}
		if res.ActionItemsCount == 0 && res.DecisionsCount == 0 {
			score -= 10.0
			res.ErrorsCount++
		}
		if score < 0 {
			score = 0
		}
		res.AuditScore = score
		if res.AuditScore >= 80.0 {
			res.Status = "PASS"
		} else if res.AuditScore >= 70.0 {
			res.Status = "WARN"
		} else {
			res.Status = "FAIL"
		}

	default: // "fable"
		if r.Verifier != nil {
			momBytes, _ := json.Marshal(mom)
			audit, aErr := r.Verifier.AuditMeetingMinutes(ctx, transcript, string(momBytes))
			if aErr != nil {
				log.Printf("[Benchmark] Audit error for %s: %v", filename, aErr)
				res.AuditScore = 0
				res.Status = "WARN"
				res.ErrorMessage = fmt.Sprintf("Audit warning: %v", aErr)
			} else if audit != nil {
				res.AuditReport = audit
				res.AuditScore = audit.Score
				res.ErrorsCount = len(audit.WhatIsWrong)
				if res.AuditScore >= 80.0 {
					res.Status = "PASS"
				} else if res.AuditScore >= 70.0 {
					res.Status = "WARN"
				} else {
					res.Status = "FAIL"
				}
			}
		} else {
			res.Status = "PASS"
		}
	}

	res.Duration = time.Since(start)
	res.DurationSeconds = res.Duration.Seconds()
	return res
}

// DetectDifficulty automatically infers or normalizes difficulty level.
func DetectDifficulty(explicitTier, filename, content string, isAudio bool) string {
	cleanExplicit := strings.TrimSpace(strings.ToLower(explicitTier))
	if cleanExplicit == "easy" {
		return TierEasy
	}
	if cleanExplicit == "medium" || cleanExplicit == "med" {
		return TierMedium
	}
	if cleanExplicit == "hard" {
		return TierHard
	}

	// Check explicit filename hints (e.g. easy_standup.txt, meeting_hard.txt, tier1, tier3)
	lowerName := strings.ToLower(filename)
	if strings.Contains(lowerName, "easy") || strings.Contains(lowerName, "tier1") || strings.Contains(lowerName, "tier-1") {
		return TierEasy
	}
	if strings.Contains(lowerName, "hard") || strings.Contains(lowerName, "tier3") || strings.Contains(lowerName, "tier-3") {
		return TierHard
	}
	if strings.Contains(lowerName, "medium") || strings.Contains(lowerName, "med") || strings.Contains(lowerName, "tier2") || strings.Contains(lowerName, "tier-2") {
		return TierMedium
	}

	// Text content heuristics
	words := len(strings.Fields(content))
	lowerContent := strings.ToLower(content)

	// Filter out common non-speaker header prefixes (e.g. Date:, Time:, Topic:)
	nonSpeakerHeaders := map[string]bool{
		"date": true, "time": true, "location": true, "attendees": true,
		"agenda": true, "topic": true, "note": true, "notes": true,
		"status": true, "action": true, "actions": true, "decision": true,
		"decisions": true, "subject": true, "title": true, "meeting": true,
		"summary": true, "chair": true, "chairperson": true, "present": true,
		"absent": true, "apologies": true, "http": true, "https": true,
	}

	speakerSet := make(map[string]struct{})
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if idx := strings.Index(trimmed, ":"); idx > 0 && idx < 25 {
			candidate := strings.TrimSpace(trimmed[:idx])
			if !nonSpeakerHeaders[strings.ToLower(candidate)] && !strings.Contains(candidate, " ") && len(candidate) > 1 {
				speakerSet[candidate] = struct{}{}
			}
		}
	}
	estimatedSpeakers := len(speakerSet)

	// Check complex cues: absent members, conflicts, rapid multi-party debate
	hasAbsentCues := strings.Contains(lowerContent, "apologies") ||
		strings.Contains(lowerContent, "off sick") ||
		strings.Contains(lowerContent, "not present") ||
		strings.Contains(lowerContent, "couldn't make it") ||
		strings.Contains(lowerContent, "absent")

	hasConflictCues := strings.Contains(lowerContent, "disagree") ||
		strings.Contains(lowerContent, "dispute") ||
		strings.Contains(lowerContent, "objection") ||
		strings.Contains(lowerContent, "conflict") ||
		strings.Contains(lowerContent, "penalty")

	// 1. Hard rules:
	// - 6+ speakers with substantial conversation (>750 words)
	// - Very long meetings (>1500 words)
	// - Multi-party conflicts/absent members with >500 words
	if (estimatedSpeakers >= 6 && words > 750) || words > 1500 || ((hasAbsentCues || hasConflictCues) && estimatedSpeakers >= 4 && words > 500) {
		return TierHard
	}

	// 2. Easy rules:
	// - Short meeting (<= 500 words) with no absent/conflict cues, regardless of speaker count
	// - 1 to 3 speakers with <= 800 words
	if !hasAbsentCues && !hasConflictCues {
		if words <= 500 || (estimatedSpeakers > 0 && estimatedSpeakers <= 3 && words <= 800) {
			return TierEasy
		}
	}

	// 3. Otherwise standard Medium (4-5 speakers, or 500-1200 words)
	return TierMedium
}

func (r *BulkBenchmarkRunner) computeSummary(results []BenchmarkItemResult, totalDurationSec float64) *BulkBenchmarkSummary {
	summary := &BulkBenchmarkSummary{
		TotalFiles:       len(results),
		TotalDurationSec: totalDurationSec,
		Results:          results,
		Tiers: map[string]*TierMetric{
			TierEasy:   {Tier: TierEasy},
			TierMedium: {Tier: TierMedium},
			TierHard:   {Tier: TierHard},
		},
		AnalysisNotes: make([]string, 0),
	}

	var scoreSum float64
	var latencySum float64

	tierScores := make(map[string][]float64)
	tierLatencies := make(map[string][]float64)

	for _, res := range results {
		latencySum += res.DurationSeconds
		scoreSum += res.AuditScore

		switch res.Status {
		case "PASS":
			summary.PassedFiles++
		case "WARN":
			summary.WarnFiles++
		case "FAIL":
			summary.FailedFiles++
		}

		tierKey := res.Difficulty
		if tierKey == "" {
			tierKey = TierMedium
		}
		tm, exists := summary.Tiers[tierKey]
		if !exists {
			tm = &TierMetric{Tier: tierKey}
			summary.Tiers[tierKey] = tm
		}

		tm.TotalFiles++
		switch res.Status {
		case "PASS":
			tm.PassedFiles++
		case "WARN":
			tm.WarnFiles++
		case "FAIL":
			tm.FailedFiles++
		}

		tm.TotalAttendees += res.AttendeesCount
		tm.TotalActionItems += res.ActionItemsCount
		tm.TotalDecisions += res.DecisionsCount
		tm.TotalDiscussions += res.DiscussionsCount
		tm.TotalErrors += res.ErrorsCount

		tierScores[tierKey] = append(tierScores[tierKey], res.AuditScore)
		tierLatencies[tierKey] = append(tierLatencies[tierKey], res.DurationSeconds)
	}

	if summary.TotalFiles > 0 {
		summary.AverageScore = scoreSum / float64(summary.TotalFiles)
		summary.AverageLatencySec = latencySum / float64(summary.TotalFiles)
		summary.PassRate = (float64(summary.PassedFiles) / float64(summary.TotalFiles)) * 100.0
	}

	// Compute averages per tier
	for tierName, tm := range summary.Tiers {
		scores := tierScores[tierName]
		latencies := tierLatencies[tierName]
		if tm.TotalFiles > 0 {
			var sSum, lSum float64
			for _, s := range scores {
				sSum += s
			}
			for _, l := range latencies {
				lSum += l
			}
			tm.AverageScore = sSum / float64(tm.TotalFiles)
			tm.AverageLatencySec = lSum / float64(tm.TotalFiles)
			tm.PassRate = (float64(tm.PassedFiles) / float64(tm.TotalFiles)) * 100.0
		}
	}

	// Generate comparative insights & diagnostics across tiers
	easyTm := summary.Tiers[TierEasy]
	medTm := summary.Tiers[TierMedium]
	hardTm := summary.Tiers[TierHard]

	var notes []string

	// Check if we have evaluations across multiple tiers
	hasEasy := easyTm != nil && easyTm.TotalFiles > 0
	hasMed := medTm != nil && medTm.TotalFiles > 0
	hasHard := hardTm != nil && hardTm.TotalFiles > 0

	if hasEasy && hasHard {
		scoreDiff := easyTm.AverageScore - hardTm.AverageScore
		if scoreDiff > 15.0 {
			notes = append(notes, fmt.Sprintf("⚠️ Complexity Sensitivity: The LLM excels on Easy transcripts (%.1f%%) but drops significantly on Hard multi-speaker transcripts (%.1f%%, -%.1f%%). Consider improving speaker diarization or cross-talk segmentation.", easyTm.AverageScore, hardTm.AverageScore, scoreDiff))
		} else if scoreDiff < -10.0 {
			notes = append(notes, fmt.Sprintf("⚠️ Simple Dialog Overfitting: The LLM scored lower on Easy transcripts (%.1f%%) than Hard transcripts (%.1f%%). The model prompt may expect elaborate structure and penalize minimal, concise standup dialogues.", easyTm.AverageScore, hardTm.AverageScore))
		} else {
			notes = append(notes, fmt.Sprintf("✅ Cross-Tier Consistency: The model maintains balanced quality between Easy (%.1f%%) and Hard (%.1f%%) transcripts (variance: %.1f%%).", easyTm.AverageScore, hardTm.AverageScore, scoreDiff))
		}

		if easyTm.AverageLatencySec > 0 && hardTm.AverageLatencySec > 0 {
			multiplier := hardTm.AverageLatencySec / easyTm.AverageLatencySec
			notes = append(notes, fmt.Sprintf("⏱️ Latency Scaling: Processing Hard meetings takes %.1fs on average vs %.1fs on Easy meetings (%.1fx multiplier).", hardTm.AverageLatencySec, easyTm.AverageLatencySec, multiplier))
		}
	}

	if hasMed && hasEasy && easyTm.AverageScore > 0 && medTm.AverageScore > 0 {
		if medTm.AverageScore < easyTm.AverageScore-10.0 {
			notes = append(notes, fmt.Sprintf("ℹ️ Medium Tier Gap: Medium transcripts scored %.1f%% vs Easy %.1f%%. Verify if action items and decision boundaries in standard reviews are being captured.", medTm.AverageScore, easyTm.AverageScore))
		}
	}

	if len(notes) == 0 && summary.TotalFiles > 0 {
		notes = append(notes, fmt.Sprintf("Evaluated %d files across active difficulty tiers with an overall average score of %.1f%%.", summary.TotalFiles, summary.AverageScore))
	}

	summary.AnalysisNotes = notes
	return summary
}

// ExportCSV writes the benchmark results into a CSV file.
func (s *BulkBenchmarkSummary) ExportCSV(outPath string) error {
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"Filename", "Type", "Difficulty", "Status", "Audit Score", "Latency (s)",
		"Attendees Count", "Attendees", "Action Items", "Decisions",
		"Discussions", "Errors Count", "Error Message",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, r := range s.Results {
		row := []string{
			r.Filename,
			r.FileType,
			r.Difficulty,
			r.Status,
			fmt.Sprintf("%.1f", r.AuditScore),
			fmt.Sprintf("%.2f", r.DurationSeconds),
			fmt.Sprintf("%d", r.AttendeesCount),
			strings.Join(r.Attendees, "; "),
			fmt.Sprintf("%d", r.ActionItemsCount),
			fmt.Sprintf("%d", r.DecisionsCount),
			fmt.Sprintf("%d", r.DiscussionsCount),
			fmt.Sprintf("%d", r.ErrorsCount),
			r.ErrorMessage,
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// ExportJSON writes the full structured summary into a JSON file.
func (s *BulkBenchmarkSummary) ExportJSON(outPath string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outPath, data, 0644)
}

func isSupportedBenchmarkFile(ext string) bool {
	return isTextFile(ext) || isAudioFile(ext)
}

func isTextFile(ext string) bool {
	return ext == ".txt" || ext == ".json"
}

func isAudioFile(ext string) bool {
	return ext == ".mp3" || ext == ".wav" || ext == ".m4a" || ext == ".ogg" || ext == ".flac" || ext == ".aac"
}

func hasSpeakerTags(text string) bool {
	lines := strings.Split(text, "\n")
	tagged := 0
	total := 0
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed == "" {
			continue
		}
		total++
		idx := strings.Index(trimmed, ":")
		if idx > 0 && idx < 25 {
			tagged++
		}
	}
	return total > 0 && (float64(tagged)/float64(total) >= 0.5)
}
