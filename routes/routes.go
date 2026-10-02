package routes

import (
	"net/http"

	"transcript/benchmark"
	"transcript/transcription"
	"transcript/verification"
)

// NewRouter sets up a dedicated ServeMux and registers all HTTP API routes with CORS support.
func NewRouter(
	convHandler *transcription.ConversationHandler,
	verifHandler *verification.Handler,
	benchHandler *benchmark.Handler,
) http.Handler {
	mux := http.NewServeMux()

	// Transcription and conversation endpoints
	mux.HandleFunc("/api/conversations", convHandler.HandleConversations)
	mux.HandleFunc("/api/conversations/", convHandler.GetConversationByID)
	mux.HandleFunc("/api/conversations/audio", convHandler.UploadAudio)

	// Automated Fable Verification and Version History endpoints
	if verifHandler != nil {
		mux.HandleFunc("/api/verification/audit", verifHandler.HandleAudit)
		mux.HandleFunc("/api/verification/versions", verifHandler.HandleVersions)
	}

	// Bulk Benchmarking & Multi-Tier Evaluation endpoints
	if benchHandler != nil {
		mux.HandleFunc("/api/benchmark/run", benchHandler.HandleRunBenchmark)
		mux.HandleFunc("/api/benchmark/samples", benchHandler.HandleRunSamples)
		mux.HandleFunc("/api/benchmark/reports", benchHandler.HandleReports)
	}

	return corsMiddleware(mux)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
