package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/joho/godotenv"

	"transcript/benchmark"
	"transcript/transcription"
	"transcript/verification"
)

func main() {
	dirFlag := flag.String("dir", "benchmark/testdata", "Directory containing audio and transcript files to benchmark")
	workersFlag := flag.Int("workers", 8, "Number of concurrent worker threads (recommended: 4-8)")
	auditFlag := flag.String("audit", "fable", "Audit mode: 'fable' (Live Claude Fable LLM QC audit)")
	outDirFlag := flag.String("out", "benchmark/reports", "Output directory for benchmark CSV and JSON reports")
	envPathFlag := flag.String("env", ".env", "Path to .env configuration file")
	flag.Parse()

	// 1. Load environment
	if err := godotenv.Load(*envPathFlag); err != nil {
		_ = godotenv.Load("../.env")
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("ERROR: GEMINI_API_KEY is not set in environment or .env file.")
	}

	// 2. Initialize transcription service
	audioSvc, err := transcription.NewGeminiTranscriptionService()
	if err != nil {
		log.Fatalf("ERROR: Failed to initialize transcription service: %v", err)
	}

	// 3. Initialize MoM service (pure in-memory mode, nil db is supported)
	momSvc, err := transcription.NewGeminiMeetingMinutesService(nil)
	if err != nil {
		log.Fatalf("ERROR: Failed to initialize meeting minutes service: %v", err)
	}

	// 4. Initialize Fable Verifier
	fableKey := os.Getenv("FABLE_API_KEY")
	fableBaseURL := os.Getenv("FABLE_BASE_URL")
	fableModel := os.Getenv("FABLE_MODEL")
	fableClient := verification.NewFableClient(fableKey, fableBaseURL, fableModel)
	verifier := verification.NewVerifier(fableClient)

	// 5. Ensure output directory exists
	if err := os.MkdirAll(*outDirFlag, 0755); err != nil {
		log.Fatalf("ERROR: Failed to create output directory %s: %v", *outDirFlag, err)
	}

	fmt.Println("================================================================================")
	fmt.Println("                     TRANSCRIPT AI - BULK BENCHMARK RUNNER                      ")
	fmt.Println("================================================================================")
	fmt.Printf(" Target Directory  : %s\n", *dirFlag)
	fmt.Printf(" Parallel Workers  : %d\n", *workersFlag)
	fmt.Printf(" Transcription Svc : %s\n", audioSvc.Model())
	fmt.Printf(" MoM Service       : %s\n", momSvc.Model())
	fmt.Printf(" Audit Mode        : %s\n", *auditFlag)
	fmt.Printf(" Report Output Dir : %s\n", *outDirFlag)
	fmt.Println("================================================================================")
	fmt.Println()

	runner := benchmark.NewBulkBenchmarkRunner(audioSvc, momSvc, verifier, *workersFlag, *auditFlag)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	summary, err := runner.RunFolder(ctx, *dirFlag)
	if err != nil {
		log.Fatalf("ERROR: Benchmark execution failed: %v", err)
	}

	// Format timestamped report filenames
	timestamp := time.Now().Format("20060102_150405")
	csvFilename := filepath.Join(*outDirFlag, fmt.Sprintf("benchmark_%s.csv", timestamp))
	jsonFilename := filepath.Join(*outDirFlag, fmt.Sprintf("benchmark_%s.json", timestamp))

	if err := summary.ExportCSV(csvFilename); err != nil {
		log.Printf("Warning: Failed to export CSV report: %v", err)
	} else {
		log.Printf("Exported CSV Report  : %s", csvFilename)
	}

	if err := summary.ExportJSON(jsonFilename); err != nil {
		log.Printf("Warning: Failed to export JSON report: %v", err)
	} else {
		log.Printf("Exported JSON Report : %s", jsonFilename)
	}

	// Print beautiful formatted table
	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("                            BENCHMARK RESULTS SUMMARY                           ")
	fmt.Println("================================================================================")
	fmt.Printf(" Total Files Processed : %d\n", summary.TotalFiles)
	fmt.Printf(" Passed (Score >= 80)  : %d\n", summary.PassedFiles)
	fmt.Printf(" Warning (Score 70-79) : %d\n", summary.WarnFiles)
	fmt.Printf(" Failed (Score < 70)   : %d\n", summary.FailedFiles)
	fmt.Printf(" Pass Rate             : %.1f%%\n", summary.PassRate)
	fmt.Printf(" Average Audit Score   : %.1f / 100\n", summary.AverageScore)
	fmt.Printf(" Average Latency       : %.2f seconds per meeting\n", summary.AverageLatencySec)
	fmt.Printf(" Total Execution Time  : %.2f seconds\n", summary.TotalDurationSec)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("%-32s | %-10s | %-7s | %-8s | %-8s | %-6s\n", "Filename", "Type", "Score", "Time (s)", "Attnd.", "Status")
	fmt.Println("--------------------------------------------------------------------------------")
	for _, r := range summary.Results {
		fmt.Printf("%-32s | %-10s | %7.1f | %8.2f | %8d | %-6s\n",
			truncate(r.Filename, 32), r.FileType, r.AuditScore, r.DurationSeconds, r.AttendeesCount, r.Status)
	}
	fmt.Println("================================================================================")
	fmt.Println("Benchmark complete!")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
