package routes

import (
	"net/http"

	"transcript/transcription"
	"transcript/verification"
)

// NewRouter sets up a dedicated ServeMux and registers all HTTP API routes.
func NewRouter(convHandler *transcription.ConversationHandler, verifHandler *verification.Handler) http.Handler {
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

	return mux
}
