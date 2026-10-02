package transcription

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"transcript/verification"
)

// ConversationHandler handles all conversation-related HTTP routes.
type ConversationHandler struct {
	DB                   *pgxpool.Pool
	TranscriptionService TranscriptionService
	MinutesService       MeetingMinutesService
	Verifier             *verification.Verifier
}

// ─────────────────────────────────────────────
// POST /api/conversations  (existing JSON endpoint)
// ─────────────────────────────────────────────

type createConversationRequest struct {
	Transcript string `json:"transcript"`
}

type conversationResponse struct {
	ID             string          `json:"id"`
	Status         string          `json:"status"`
	Transcript     string          `json:"transcript,omitempty"`
	AudioFilename  *string         `json:"audio_filename,omitempty"`
	MeetingMinutes *MeetingMinutes `json:"meeting_minutes"`
	TokenUsage     *TokenUsage     `json:"token_usage,omitempty"`
	CreatedAt      string          `json:"created_at,omitempty"`
}

var speakerTagRegex = regexp.MustCompile(`(?i)(?:^|\n)\s*(?:Speaker\s*\d+|[A-Z][a-zA-Z\s]{1,20})\s*:\s*`)
var multipleSpaceRegex = regexp.MustCompile(`\s+`)

func normalizeTranscript(s string) string {
	s = speakerTagRegex.ReplaceAllString(s, " ")
	s = multipleSpaceRegex.ReplaceAllString(s, " ")
	return strings.ToLower(strings.TrimSpace(s))
}

func isTranscriptMatch(a, b string) bool {
	aTrim := strings.TrimSpace(a)
	bTrim := strings.TrimSpace(b)
	if aTrim == "" || bTrim == "" {
		return false
	}
	if aTrim == bTrim {
		return true
	}
	normA := normalizeTranscript(aTrim)
	normB := normalizeTranscript(bTrim)
	if normA == normB {
		return true
	}
	// If both have at least 50 normalized characters, test substring / prefix containment
	if len(normA) >= 50 && len(normB) >= 50 {
		minPrefixLen := 60
		if len(normA) < minPrefixLen {
			minPrefixLen = len(normA)
		}
		if len(normB) < minPrefixLen {
			minPrefixLen = len(normB)
		}
		prefixA := normA[:minPrefixLen]
		prefixB := normB[:minPrefixLen]
		if strings.Contains(normB, prefixA) || strings.Contains(normA, prefixB) {
			return true
		}
	}
	return false
}

func toVerifierUsage(u *TokenUsage) *verification.VersionTokenUsage {
	if u == nil {
		return nil
	}
	return &verification.VersionTokenUsage{
		Model:            u.Model,
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		TotalTokens:      u.TotalTokens,
		EstimatedCostINR: u.EstimatedCostINR,
		EstimatedCostUSD: u.EstimatedCostUSD,
	}
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

	cleanedTranscript := strings.TrimSpace(req.Transcript)
	if cleanedTranscript == "" {
		http.Error(w, "Transcript is required", http.StatusBadRequest)
		return
	}

	// 1. Check if an existing conversation has matching transcript (exact match, normalized, or speaker-stripped)
	var id uuid.UUID
	var isExisting bool
	var existingAudioFilename *string

	rows, qErr := h.DB.Query(r.Context(), `
		SELECT id, audio_filename, transcript FROM conversations 
		ORDER BY created_at DESC LIMIT 50
	`)
	if qErr == nil {
		defer rows.Close()
		for rows.Next() {
			var candID uuid.UUID
			var candAudio *string
			var candTranscript string
			if scanErr := rows.Scan(&candID, &candAudio, &candTranscript); scanErr == nil {
				if isTranscriptMatch(candTranscript, cleanedTranscript) {
					id = candID
					existingAudioFilename = candAudio
					isExisting = true
					break
				}
			}
		}
	}

	if isExisting {
		// Update timestamp to current upload time and reset status to processing
		_, _ = h.DB.Exec(r.Context(), `UPDATE conversations SET created_at = NOW(), status = 'processing' WHERE id = $1`, id)
		log.Printf("CreateConversation: matched existing conversation %s with matching transcript (audio: %v, existing: %t)", id, existingAudioFilename, isExisting)
	} else {
		id = uuid.New()
		isExisting = false
		_, err := h.DB.Exec(
			r.Context(),
			`INSERT INTO conversations (id, transcript, status, created_at) VALUES ($1, $2, $3, NOW())`,
			id, cleanedTranscript, "processing",
		)
		if err != nil {
			log.Printf("CreateConversation: db insert error: %v", err)
			http.Error(w, "Failed to save conversation", http.StatusInternalServerError)
			return
		}
		log.Printf("CreateConversation: created new conversation %s (existing: %t)", id, isExisting)
	}

	var extractedMinutes *MeetingMinutes
	var tokenUsage *TokenUsage

	if h.MinutesService != nil {
		log.Printf("CreateConversation: extracting meeting minutes for conversation %s", id)
		var mErr error
		extractCtx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		extractedMinutes, _, tokenUsage, mErr = h.MinutesService.ExtractAndSaveMeetingMinutes(extractCtx, id, cleanedTranscript)
		if mErr != nil {
			log.Printf("CreateConversation: meeting minutes extraction error for conversation %s: %v", id, mErr)
			_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET status = 'completed' WHERE id = $1`, id)
		} else {
			log.Printf("CreateConversation: saved meeting minutes for conversation %s", id)
			_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET transcript = $1, status = 'completed' WHERE id = $2`, cleanedTranscript, id)

			// 2. Automated Fable verification and version management
			if h.Verifier != nil && extractedMinutes != nil {
				momBytes, _ := json.Marshal(extractedMinutes)
				momStr := string(momBytes)

				prevRecord, _ := h.Verifier.GetLatestVersion(extractCtx, h.DB, id)
				var initialVer int
				if prevRecord == nil {
					initialVer = 1
				} else {
					initialVer = prevRecord.VersionNumber + 1
				}

				// Immediately save version record to PostgreSQL so UI sees Version 1/2 instantly (<8s total request)
				vRec, sErr := h.Verifier.SaveVersion(extractCtx, h.DB, id, initialVer, momStr, nil, nil, toVerifierUsage(tokenUsage))
				if sErr != nil {
					log.Printf("CreateConversation: error saving initial version %d: %v", initialVer, sErr)
				} else if vRec != nil {
					log.Printf("CreateConversation: saved version %d for %s (instant save)", initialVer, id)
				}

				// Run Fable verification asynchronously in the background so HTTP request never times out or 500s!
				go func(convID uuid.UUID, verNum int, transcriptText, momPayload string, prev *verification.MeetingMinutesVersionRecord, usage *TokenUsage) {
					bgCtx, bgCancel := context.WithTimeout(context.Background(), 5*time.Minute)
					defer bgCancel()

					if prev == nil {
						// Initial version audit
						auditResult, aErr := h.Verifier.AuditMeetingMinutes(bgCtx, transcriptText, momPayload)
						if aErr != nil {
							log.Printf("CreateConversation background audit error for %s: %v", convID, aErr)
							return
						}
						_, _ = h.Verifier.SaveVersion(bgCtx, h.DB, convID, verNum, momPayload, auditResult, nil, toVerifierUsage(usage))
						log.Printf("CreateConversation: completed background audit for %s v%d (Score: %.1f)", convID, verNum, auditResult.Score)
					} else {
						// Comparison against previous version
						prevMinutesBytes, _ := json.Marshal(prev.MeetingMinutesData)
						prevMinutesStr := string(prevMinutesBytes)

						compResult, cErr := h.Verifier.CompareVersions(bgCtx, transcriptText, prevMinutesStr, momPayload, prev.Errors, prev.Score)
						if cErr != nil {
							log.Printf("CreateConversation background comparison error for %s: %v", convID, cErr)
							return
						}
						compResult.PreviousVersion = prev.VersionNumber
						compResult.CurrentVersion = verNum
						currAudit, _ := h.Verifier.AuditMeetingMinutes(bgCtx, transcriptText, momPayload)
						if currAudit != nil {
							compResult.PopulateFromAudit(currAudit)
						}
						_, _ = h.Verifier.SaveVersion(bgCtx, h.DB, convID, verNum, momPayload, currAudit, compResult, toVerifierUsage(usage))
						log.Printf("CreateConversation: completed background comparison for %s v%d (Score: %.1f, Delta: %+.1f)",
							convID, verNum, compResult.CurrentScore, compResult.ScoreDelta)
					}
				}(id, initialVer, cleanedTranscript, momStr, prevRecord, tokenUsage)
			}
		}
	}

	writeJSON(w, http.StatusCreated, conversationResponse{
		ID:             id.String(),
		Status:         "completed",
		Transcript:     cleanedTranscript,
		AudioFilename:  existingAudioFilename,
		MeetingMinutes: extractedMinutes,
		TokenUsage:     tokenUsage,
		CreatedAt:      time.Now().UTC().Format(time.RFC3339),
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

	// Check if a conversation with this audio filename already exists to update it instead of duplicating.
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
		// Update existing conversation record to processing status.
		_, err = h.DB.Exec(
			r.Context(),
			`UPDATE conversations SET transcript = $1, status = $2, created_at = NOW() WHERE id = $3`,
			"", "processing", id,
		)
		if err != nil {
			log.Printf("UploadAudio: db update error: %v", err)
			http.Error(w, "Failed to update conversation status", http.StatusInternalServerError)
			return
		}

		// Delete old meeting_minutes for this conversation so new MoM replaces them cleanly.
		_, _ = h.DB.Exec(r.Context(), `DELETE FROM meeting_minutes WHERE conversation_id = $1`, id)

		// Remove any older duplicate conversation records with the same filename.
		_, _ = h.DB.Exec(r.Context(), `DELETE FROM conversations WHERE audio_filename = $1 AND id != $2`, cleanFilename, id)

		log.Printf("UploadAudio: reset conversation %s (audio: %s) to 'processing'", id, cleanFilename)
	} else {
		// Persist new conversation to PostgreSQL with status processing.
		_, err = h.DB.Exec(
			r.Context(),
			`INSERT INTO conversations (id, transcript, status, audio_filename, created_at) VALUES ($1, $2, $3, $4, NOW())`,
			id, "", "processing", cleanFilename,
		)
		if err != nil {
			log.Printf("UploadAudio: db insert error: %v", err)
			http.Error(w, "Failed to save initial conversation", http.StatusInternalServerError)
			return
		}

		log.Printf("UploadAudio: created new conversation %s (audio: %s) with status 'processing'", id, cleanFilename)
	}

	// Launch transcription and MoM extraction asynchronously in the background.
	go h.processAudioBackground(id, localFilePath, cleanFilename)

	// Return immediately with 200 OK so the HTTP request completes in <1s and never times out.
	writeJSON(w, http.StatusOK, conversationResponse{
		ID:            id.String(),
		Status:        "processing",
		Transcript:    "",
		AudioFilename: &cleanFilename,
	})
}

// processAudioBackground performs audio transcription and MoM generation in a detached background goroutine.
func (h *ConversationHandler) processAudioBackground(id uuid.UUID, localFilePath, cleanFilename string) {
	log.Printf("processAudioBackground: starting transcription for conversation %s (%s)", id, cleanFilename)

	savedAudioFile, err := os.Open(localFilePath)
	if err != nil {
		log.Printf("processAudioBackground: failed to open audio file %s: %v", localFilePath, err)
		_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET status = 'failed' WHERE id = $1`, id)
		return
	}
	defer savedAudioFile.Close()

	// Use an independent context with a generous 10-minute timeout for the entire background flow.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Transcribe via Gemini.
	transcript, err := h.TranscriptionService.Transcribe(ctx, savedAudioFile, cleanFilename)
	if err != nil {
		log.Printf("processAudioBackground: transcription failed for conversation %s: %v", id, err)
		_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET status = 'failed' WHERE id = $1`, id)
		return
	}

	if strings.TrimSpace(transcript) == "" {
		log.Printf("processAudioBackground: empty transcript returned for conversation %s", id)
		_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET status = 'failed' WHERE id = $1`, id)
		return
	}

	log.Printf("processAudioBackground: transcription succeeded for %s (%d chars)", id, len(transcript))
	_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET transcript = $1 WHERE id = $2`, transcript, id)

	// Extract and save Meeting Minutes (with dynamic speaker name resolution).
	if h.MinutesService != nil {
		log.Printf("processAudioBackground: extracting meeting minutes for conversation %s", id)
		mom, _, tokenUsage, mErr := h.MinutesService.ExtractAndSaveMeetingMinutes(ctx, id, transcript)
		if mErr != nil {
			log.Printf("processAudioBackground: meeting minutes extraction error for conversation %s: %v", id, mErr)
			// Mark completed with the transcript we already have even if MoM failed
			_, _ = h.DB.Exec(context.Background(), `UPDATE conversations SET status = 'completed' WHERE id = $1`, id)
			return
		}

		// Automatically run Fable audit and version comparison
		if h.Verifier != nil && mom != nil {
			log.Printf("processAudioBackground: running automatic Fable verification for conversation %s", id)
			momBytes, _ := json.Marshal(mom)
			momStr := string(momBytes)

			prevRecord, _ := h.Verifier.GetLatestVersion(ctx, h.DB, id)
			if prevRecord == nil {
				// Initial version (Version 1)
				auditResult, aErr := h.Verifier.AuditMeetingMinutes(ctx, transcript, momStr)
				if aErr != nil {
					log.Printf("processAudioBackground: Fable audit error: %v (saving initial version 1 with audit pending)", aErr)
				}
				// ALWAYS persist Version 1 so it appears in the UI
				vRec, sErr := h.Verifier.SaveVersion(ctx, h.DB, id, 1, momStr, auditResult, nil, toVerifierUsage(tokenUsage))
				if sErr != nil {
					log.Printf("processAudioBackground: failed to save version 1: %v", sErr)
				} else if vRec != nil {
					log.Printf("processAudioBackground: saved initial version 1 for %s (Score: %.1f)", id, vRec.Score)
				}
			} else {
				// Subsequent version (Version 2, 3... automatic comparison!)
				prevMinutesBytes, _ := json.Marshal(prevRecord.MeetingMinutesData)
				prevMinutesStr := string(prevMinutesBytes)
				newVer := prevRecord.VersionNumber + 1

				compResult, cErr := h.Verifier.CompareVersions(ctx, transcript, prevMinutesStr, momStr, prevRecord.Errors, prevRecord.Score)
				var currAudit *verification.AuditResult
				if cErr == nil && compResult != nil {
					compResult.PreviousVersion = prevRecord.VersionNumber
					compResult.CurrentVersion = newVer
					currAudit, _ = h.Verifier.AuditMeetingMinutes(ctx, transcript, momStr)
					if currAudit != nil {
						compResult.PopulateFromAudit(currAudit)
					}
				} else {
					log.Printf("processAudioBackground: Fable compare error: %v (saving version %d with comparison pending)", cErr, newVer)
				}

				// ALWAYS persist Version 2/3 so it appears in the UI
				vRec, sErr := h.Verifier.SaveVersion(ctx, h.DB, id, newVer, momStr, currAudit, compResult, toVerifierUsage(tokenUsage))
				if sErr != nil {
					log.Printf("processAudioBackground: failed to save version %d: %v", newVer, sErr)
				} else if vRec != nil {
					log.Printf("processAudioBackground: saved compared version %d for %s (Score: %.1f, Delta: %+.1f, Improved: %t)",
						newVer, id, vRec.Score, vRec.ScoreDelta, vRec.IsImproved)
				}
			}
		}
	}

	_, _ = h.DB.Exec(context.Background(),
		`UPDATE conversations SET transcript = $1, status = 'completed' WHERE id = $2`,
		transcript, id)
	log.Printf("processAudioBackground: successfully completed conversation %s", id)
}

// ConversationDetail represents a complete conversation with its associated meeting minutes and token usage.
type ConversationDetail struct {
	ID             uuid.UUID       `json:"id"`
	Status         string          `json:"status"`
	Transcript     string          `json:"transcript"`
	AudioFilename  *string         `json:"audio_filename"`
	CreatedAt      time.Time       `json:"created_at"`
	MeetingMinutes *MeetingMinutes `json:"meeting_minutes"`
	TokenUsage     *TokenUsage     `json:"token_usage,omitempty"`
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
		c.MeetingMinutes = nil
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

	// Fetch meeting minutes and token usage for these conversations
	minutesRows, err := h.DB.Query(r.Context(),
		`SELECT conversation_id, data,
		        COALESCE(model, 'gemini-3.5-flash-lite'), COALESCE(input_tokens, 0), COALESCE(output_tokens, 0),
		        COALESCE(total_tokens, 0), COALESCE(estimated_cost_inr, 0), COALESCE(estimated_cost_usd, 0)
		 FROM meeting_minutes 
		 WHERE conversation_id = ANY($1)`,
		convIDs)
	if err != nil {
		log.Printf("GetConversations: meeting minutes query error: %v", err)
		writeJSONError(w, http.StatusInternalServerError, "Failed to retrieve meeting minutes")
		return
	}
	defer minutesRows.Close()

	minutesByConvID := make(map[uuid.UUID]*MeetingMinutes)
	usageByConvID := make(map[uuid.UUID]*TokenUsage)
	for minutesRows.Next() {
		var convID uuid.UUID
		var rawData []byte
		var u TokenUsage
		if err := minutesRows.Scan(&convID, &rawData, &u.Model, &u.InputTokens, &u.OutputTokens, &u.TotalTokens, &u.EstimatedCostINR, &u.EstimatedCostUSD); err != nil {
			log.Printf("GetConversations: minutes scan error: %v", err)
			continue
		}
		var mom MeetingMinutes
		if err := json.Unmarshal(rawData, &mom); err == nil {
			minutesByConvID[convID] = &mom
		}
		usageByConvID[convID] = &u
	}

	for i := range conversations {
		if mom, ok := minutesByConvID[conversations[i].ID]; ok {
			conversations[i].MeetingMinutes = mom
		}
		if u, ok := usageByConvID[conversations[i].ID]; ok {
			conversations[i].TokenUsage = u
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

	c.MeetingMinutes = nil

	var rawMinutes []byte
	var u TokenUsage
	err = h.DB.QueryRow(r.Context(),
		`SELECT data,
		        COALESCE(model, 'gemini-3.5-flash-lite'), COALESCE(input_tokens, 0), COALESCE(output_tokens, 0),
		        COALESCE(total_tokens, 0), COALESCE(estimated_cost_inr, 0), COALESCE(estimated_cost_usd, 0)
		 FROM meeting_minutes WHERE conversation_id = $1`,
		convID).Scan(&rawMinutes, &u.Model, &u.InputTokens, &u.OutputTokens, &u.TotalTokens, &u.EstimatedCostINR, &u.EstimatedCostUSD)
	if err == nil {
		var mom MeetingMinutes
		if err := json.Unmarshal(rawMinutes, &mom); err == nil {
			c.MeetingMinutes = &mom
		}
		c.TokenUsage = &u
	} else if !errors.Is(err, pgx.ErrNoRows) {
		log.Printf("GetConversationByID: minutes query error: %v", err)
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
