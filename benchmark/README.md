# Bulk Benchmark Runner

A fast, concurrent benchmarking tool for evaluating transcription, speaker resolution, Minutes of Meeting (MoM) extraction, and Claude Fable audit scores across bulk audio files and transcripts.

---

## Quick Start

### 1. Run Benchmark on Default Test Data
```powershell
cd backend
go run ./benchmark/cmd
```

### 2. Run Benchmark on Any Custom Folder
Place your audio files (`.mp3`, `.wav`, `.m4a`) or transcripts (`.txt`, `.json`) into a directory, then run:
```powershell
go run ./benchmark/cmd -dir="C:/path/to/your/files"
```

### 3. Customize Concurrency (Workers)
Control the parallel worker pool (recommended: 3–4 workers for optimal speed without rate limits):
```powershell
go run ./benchmark/cmd -dir=benchmark/testdata -workers=4
```

---

## Output & Reports
Every benchmark run produces:
1. **Live Terminal Progress Table** with latency, attendees, and audit scores.
2. **CSV Report** (`benchmark/reports/benchmark_<timestamp>.csv`) — openable directly in Microsoft Excel or Google Sheets.
3. **JSON Report** (`benchmark/reports/benchmark_<timestamp>.json`) — complete structured dump with full attendee lists, action item counts, and error breakdowns.

---

## Supported File Formats
- **Transcripts:** `.txt`, `.json`
- **Audio Files:** `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`
