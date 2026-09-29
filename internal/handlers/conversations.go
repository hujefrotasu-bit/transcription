package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"transcript/internal/topics"
	"transcript/internal/transcription"
)

// ConversationHandler handles all conversation-related HTTP routes.
type ConversationHandler struct {
	DB                   *pgxpool.Pool
	TranscriptionService transcription.TranscriptionService
	TopicService         topics.TopicService
}

// ─────────────────────────────────────────────
// POST /api/conversations  (existing JSON endpoint — unchanged)
// ─────────────────────────────────────────────

type createConversationRequest struct {
	Transcript string `json:"transcript"`
}

type conversationResponse struct {
	ID            string         `json:"id"`
	Status        string         `json:"status"`
	Transcript    string         `json:"transcript,omitempty"`
	AudioFilename *string        `json:"audio_filename,omitempty"`
	Topics        []topics.Topic `json:"topics,omitempty"`
}

// HandleConversations routes GET to GetConversations and POST to CreateConversation.
func (h *ConversationHandler) HandleConversations(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		h.GetConversations(w, r)
	case http.MethodPost:
		h.CreateConversation(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (h *ConversationHandler) CreateConversation(w http.ResponseWriter, r *http.Request) {
	var req createConversationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Transcript == "" {
		http.Error(w, "Transcript is required", http.StatusBadRequest)
		return
	}

	id := uuid.New()
	_, err := h.DB.Exec(
		context.Background(),
		`INSERT INTO conversations (id, transcript, status) VALUES ($1, $2, $3)`,
		id, req.Transcript, "processing",
	)
	if err != nil {
		log.Printf("CreateConversation: db insert error: %v", err)
		http.Error(w, "Failed to save conversation", http.StatusInternalServerError)
		return
	}

	log.Printf("CreateConversation: saved conversation %s", id)

	var extractedTopics []topics.Topic
	if h.TopicService != nil {
		log.Printf("CreateConversation: extracting topics for conversation %s", id)
		var tErr error
		extractedTopics, tErr = h.TopicService.ExtractAndSave(context.Background(), id, req.Transcript)
		if tErr != nil {
			log.Printf("CreateConversation: topic extraction error for conversation %s: %v", id, tErr)
		} else {
			log.Printf("CreateConversation: saved %d topics for conversation %s", len(extractedTopics), id)
		}
	}

	writeJSON(w, http.StatusCreated, conversationResponse{
		ID:     id.String(),
		Status: "processing",
		Topics: extractedTopics,
	})
}

// ─────────────────────────────────────────────
// POST /api/conversations/audio
// ─────────────────────────────────────────────

// maxAudioSize limits memory used when parsing the multipart form (32 MB).
const maxAudioSize = 32 << 20

func (h *ConversationHandler) UploadAudio(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form; keep at most maxAudioSize in memory, rest goes to disk.
	if err := r.ParseMultipartForm(maxAudioSize); err != nil {
		http.Error(w, "Invalid multipart request", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "Audio file is required", http.StatusBadRequest)
		return
	}
	defer file.Close()

	if header.Size == 0 {
		http.Error(w, "Audio file is empty", http.StatusBadRequest)
		return
	}

	log.Printf("UploadAudio: received %q (%d bytes), starting transcription", header.Filename, header.Size)

	// Ensure local uploads directory exists.
	uploadsDir := "uploads"
	if err := os.MkdirAll(uploadsDir, 0755); err != nil {
		log.Printf("UploadAudio: failed to create uploads directory: %v", err)
		http.Error(w, "Failed to create uploads directory", http.StatusInternalServerError)
		return
	}

	cleanFilename := filepath.Base(header.Filename)
	localFilePath := filepath.Join(uploadsDir, cleanFilename)

	dst, err := os.Create(localFilePath)
	if err != nil {
		log.Printf("UploadAudio: failed to create local audio file: %v", err)
		http.Error(w, "Failed to save audio file locally", http.StatusInternalServerError)
		return
	}

	if _, err := io.Copy(dst, file); err != nil {
		dst.Close()
		log.Printf("UploadAudio: failed to write local audio file: %v", err)
		http.Error(w, "Failed to save audio file locally", http.StatusInternalServerError)
		return
	}
	dst.Close()

	log.Printf("UploadAudio: saved audio file locally to %s", localFilePath)

	// Open the saved local file for transcription.
	savedAudioFile, err := os.Open(localFilePath)
	if err != nil {
		log.Printf("UploadAudio: failed to open saved audio file: %v", err)
		http.Error(w, "Failed to read saved audio file", http.StatusInternalServerError)
		return
	}
	defer savedAudioFile.Close()

	// Transcribe via the injected service (Gemini under the hood).
	transcript, err := h.TranscriptionService.Transcribe(r.Context(), savedAudioFile, cleanFilename)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			log.Printf("UploadAudio: transcription cancelled/timed out: %v", err)
			http.Error(w, "Request timed out", http.StatusGatewayTimeout)
			return
		}
		log.Printf("UploadAudio: transcription error: %v", err)
		http.Error(w, "Transcription failed", http.StatusBadGateway)
		return
	}

	if transcript == "" {
		log.Printf("UploadAudio: empty transcript returned for %q", cleanFilename)
		http.Error(w, "Transcription returned an empty result", http.StatusUnprocessableEntity)
		return
	}

	log.Printf("UploadAudio: transcription succeeded (%d chars), saving to database", len(transcript))

	// Check if a conversation with this audio filename already exists to replace it instead of duplicating.
	var id uuid.UUID
	var existing bool
	err = h.DB.QueryRow(r.Context(),
		`SELECT id FROM conversations WHERE audio_filename = $1 ORDER BY created_at DESC LIMIT 1`,
		cleanFilename).Scan(&id)
	if err == nil {
		existing = true
	} else if errors.Is(err, pgx.ErrNoRows) {
		id = uuid.New()
		existing = false
	} else {
		log.Printf("UploadAudio: db query error checking existing file: %v", err)
		http.Error(w, "Failed to check existing conversation", http.StatusInternalServerError)
		return
	}

	if existing {
		// Update existing conversation record.
		_, err = h.DB.Exec(
			r.Context(),
			`UPDATE conversations SET transcript = $1, status = $2, created_at = NOW() WHERE id = $3`,
			transcript, "completed", id,
		)
		if err != nil {
			log.Printf("UploadAudio: db update error: %v", err)
			http.Error(w, "Failed to update transcript", http.StatusInternalServerError)
			return
		}

		// Delete old topics for this conversation so new topics replace them cleanly.
		_, _ = h.DB.Exec(r.Context(), `DELETE FROM topics WHERE conversation_id = $1`, id)

		// Remove any older duplicate conversation records with the same filename.
		_, _ = h.DB.Exec(r.Context(), `DELETE FROM conversations WHERE audio_filename = $1 AND id != $2`, cleanFilename, id)

		log.Printf("UploadAudio: replaced existing conversation %s (audio: %s)", id, cleanFilename)
	} else {
		// Persist new conversation to PostgreSQL.
		_, err = h.DB.Exec(
			r.Context(),
			`INSERT INTO conversations (id, transcript, status, audio_filename) VALUES ($1, $2, $3, $4)`,
			id, transcript, "completed", cleanFilename,
		)
		if err != nil {
			log.Printf("UploadAudio: db insert error: %v", err)
			http.Error(w, "Failed to save transcript", http.StatusInternalServerError)
			return
		}

		log.Printf("UploadAudio: saved new conversation %s (audio: %s)", id, cleanFilename)
	}

	var extractedTopics []topics.Topic
	if h.TopicService != nil {
		log.Printf("UploadAudio: extracting topics for conversation %s", id)
		var tErr error
		extractedTopics, tErr = h.TopicService.ExtractAndSave(r.Context(), id, transcript)
		if tErr != nil {
			log.Printf("UploadAudio: topic extraction error for conversation %s: %v", id, tErr)
		} else {
			log.Printf("UploadAudio: saved %d topics for conversation %s", len(extractedTopics), id)
		}
	}

	writeJSON(w, http.StatusCreated, conversationResponse{
		ID:            id.String(),
		Status:        "completed",
		Transcript:    transcript,
		AudioFilename: &cleanFilename,
		Topics:        extractedTopics,
	})
}

// ConversationDetail represents a complete conversation with its associated topics.
type ConversationDetail struct {
	ID            uuid.UUID      `json:"id"`
	Status        string         `json:"status"`
	Transcript    string         `json:"transcript"`
	AudioFilename *string        `json:"audio_filename"`
	CreatedAt     time.Time      `json:"created_at"`
	Topics        []topics.Topic `json:"topics"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// ─────────────────────────────────────────────
// GET /api/conversations
// ─────────────────────────────────────────────

func (h *ConversationHandler) GetConversations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	convRows, err := h.DB.Query(r.Context(),
		`SELECT id, status, transcript, audio_filename, created_at 
		 FROM conversations 
		 ORDER BY created_at DESC`)
	if err != nil {
		log.Printf("GetConversations: query error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to retrieve conversations")
		return
	}
	defer convRows.Close()

	var conversations []ConversationDetail
	var convIDs []uuid.UUID

	for convRows.Next() {
		var c ConversationDetail
		if err := convRows.Scan(&c.ID, &c.Status, &c.Transcript, &c.AudioFilename, &c.CreatedAt); err != nil {
			log.Printf("GetConversations: scan error: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "Failed to parse conversations")
			return
		}
		c.Topics = []topics.Topic{}
		conversations = append(conversations, c)
		convIDs = append(convIDs, c.ID)
	}
	if err := convRows.Err(); err != nil {
		log.Printf("GetConversations: rows error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to read conversations")
		return
	}

	if len(conversations) == 0 {
		writeJSON(w, http.StatusOK, []ConversationDetail{})
		return
	}

	topicRows, err := h.DB.Query(r.Context(),
		`SELECT id, conversation_id, title, summary, created_at 
		 FROM topics 
		 WHERE conversation_id = ANY($1) 
		 ORDER BY created_at ASC`,
		convIDs)
	if err != nil {
		log.Printf("GetConversations: topic query error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to retrieve topics")
		return
	}
	defer topicRows.Close()

	topicsByConvID := make(map[uuid.UUID][]topics.Topic)
	for topicRows.Next() {
		var t topics.Topic
		if err := topicRows.Scan(&t.ID, &t.ConversationID, &t.Title, &t.Summary, &t.CreatedAt); err != nil {
			log.Printf("GetConversations: topic scan error: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "Failed to parse topics")
			return
		}
		topicsByConvID[t.ConversationID] = append(topicsByConvID[t.ConversationID], t)
	}

	for i := range conversations {
		if tList, ok := topicsByConvID[conversations[i].ID]; ok {
			conversations[i].Topics = tList
		}
	}

	writeJSON(w, http.StatusOK, conversations)
}

// ─────────────────────────────────────────────
// GET /api/conversations/:id
// ─────────────────────────────────────────────

func (h *ConversationHandler) GetConversationByID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	idStr := r.PathValue("id")
	if idStr == "" {
		idStr = strings.TrimPrefix(r.URL.Path, "/api/conversations/")
		idStr = strings.Trim(idStr, "/")
	}

	if idStr == "" {
		h.GetConversations(w, r)
		return
	}

	convID, err := uuid.Parse(idStr)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, "Conversation not found")
		return
	}

	var c ConversationDetail
	err = h.DB.QueryRow(r.Context(),
		`SELECT id, status, transcript, audio_filename, created_at 
		 FROM conversations 
		 WHERE id = $1`,
		convID).Scan(&c.ID, &c.Status, &c.Transcript, &c.AudioFilename, &c.CreatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeJSONError(w, http.StatusNotFound, "Conversation not found")
			return
		}
		log.Printf("GetConversationByID: query error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to retrieve conversation")
		return
	}

	c.Topics = []topics.Topic{}

	topicRows, err := h.DB.Query(r.Context(),
		`SELECT id, conversation_id, title, summary, created_at 
		 FROM topics 
		 WHERE conversation_id = $1 
		 ORDER BY created_at ASC`,
		convID)
	if err != nil {
		log.Printf("GetConversationByID: topic query error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to retrieve topics")
		return
	}
	defer topicRows.Close()

	for topicRows.Next() {
		var t topics.Topic
		if err := topicRows.Scan(&t.ID, &t.ConversationID, &t.Title, &t.Summary, &t.CreatedAt); err != nil {
			log.Printf("GetConversationByID: topic scan error: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "Failed to parse topics")
			return
		}
		c.Topics = append(c.Topics, t)
	}

	writeJSON(w, http.StatusOK, c)
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON: encode error: %v", err)
	}
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, errorResponse{Error: msg})
}
