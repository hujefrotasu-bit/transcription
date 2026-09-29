package routes

import (
	"net/http"

	"transcript/transcription"
)

// NewRouter sets up a dedicated ServeMux and registers all HTTP API routes.
func NewRouter(handler *transcription.ConversationHandler) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/conversations", handler.HandleConversations)
	mux.HandleFunc("/api/conversations/", handler.GetConversationByID)
	mux.HandleFunc("/api/conversations/audio", handler.UploadAudio)

	return mux
}
