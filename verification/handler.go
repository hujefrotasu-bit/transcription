package verification

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Handler provides HTTP endpoints for automated Meeting Minutes auditing, version comparison, and history retrieval.
type Handler struct {
	DB       *pgxpool.Pool
	Verifier *Verifier
}

// NewHandler initializes a new verification HTTP Handler.
func NewHandler(db *pgxpool.Pool, verifier *Verifier) *Handler {
	return &Handler{
		DB:       db,
		Verifier: verifier,
	}
}

type auditRequest struct {
	ConversationID string `json:"conversation_id"`
	Transcript     string `json:"transcript"`
	MeetingMinutes any    `json:"meeting_minutes"`
}

// HandleAudit automatically audits meeting minutes against the transcript.
// If a previous version already exists in the database for this conversation,
// it AUTOMATICALLY runs a comparative version analysis (calculating fixed errors, regressions, and score delta).
// POST /api/verification/audit
func (h *Handler) HandleAudit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req auditRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON request: "+err.Error(), http.StatusBadRequest)
		return
	}

	convID, err := uuid.Parse(req.ConversationID)
	if err != nil {
		http.Error(w, "Valid conversation_id is required: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 1. Fetch transcript from DB if not provided in request body
	transcript := strings.TrimSpace(req.Transcript)
	if transcript == "" && h.DB != nil {
		var dbTranscript string
		err := h.DB.QueryRow(r.Context(), "SELECT transcript FROM conversations WHERE id = $1", convID).Scan(&dbTranscript)
		if err == nil {
			transcript = strings.TrimSpace(dbTranscript)
		}
	}

	if transcript == "" {
		http.Error(w, "transcript is required (not found in database for this conversation_id)", http.StatusBadRequest)
		return
	}

	minutesStr, err := toJSONString(req.MeetingMinutes)
	if err != nil {
		http.Error(w, "Invalid meeting_minutes: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 2. Check if a previous version already exists in meeting_minutes_versions
	var prevRecord *MeetingMinutesVersionRecord
	if h.DB != nil {
		prevRecord, _ = h.Verifier.GetLatestVersion(r.Context(), h.DB, convID)
	}

	// ─────────────────────────────────────────────
	// Case A: Initial Version (Version 1)
	// ─────────────────────────────────────────────
	if prevRecord == nil {
		auditResult, err := h.Verifier.AuditMeetingMinutes(r.Context(), transcript, minutesStr)
		if err != nil {
			http.Error(w, "Fable audit failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		var record *MeetingMinutesVersionRecord
		if h.DB != nil {
			record, _ = h.Verifier.SaveVersion(r.Context(), h.DB, convID, 1, minutesStr, auditResult, nil)
		}

		resp := map[string]any{
			"conversation_id":  convID.String(),
			"version_number":   1,
			"score":            auditResult.Score,
			"score_breakdown":  auditResult.ScoreBreakdown,
			"verified":         auditResult.Verified,
			"what_is_right":    auditResult.WhatIsRight,
			"what_is_wrong":    auditResult.WhatIsWrong,
			"missing_information": auditResult.MissingInformation,
			"improvement_summary": auditResult.ImprovementSummary,
			"is_comparison":    false,
			"record":           record,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	// ─────────────────────────────────────────────
	// Case B: Automatic Version Comparison (Version N)
	// ─────────────────────────────────────────────
	prevMinutesBytes, _ := json.Marshal(prevRecord.MeetingMinutesData)
	prevMinutesStr := string(prevMinutesBytes)

	// Fable automatically runs comparative analysis against the previous version
	compResult, err := h.Verifier.CompareVersions(r.Context(), transcript, prevMinutesStr, minutesStr, prevRecord.Errors, prevRecord.Score)
	if err != nil {
		http.Error(w, "Automatic Fable version comparison failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	newVersionNumber := prevRecord.VersionNumber + 1
	compResult.PreviousVersion = prevRecord.VersionNumber
	compResult.CurrentVersion = newVersionNumber

	// Audit new version for full detailed breakdown
	currAudit, err := h.Verifier.AuditMeetingMinutes(r.Context(), transcript, minutesStr)
	if err != nil {
		// Fallback to comparison score if full audit call fails
		currAudit = &AuditResult{
			Score:       compResult.CurrentScore,
			Verified:    compResult.CurrentScore >= 90.0 && len(compResult.Regressions) == 0,
			WhatIsWrong: compResult.ToVerificationErrors(),
		}
	} else {
		compResult.PopulateFromAudit(currAudit)
	}

	var record *MeetingMinutesVersionRecord
	if h.DB != nil {
		record, _ = h.Verifier.SaveVersion(r.Context(), h.DB, convID, newVersionNumber, minutesStr, currAudit, compResult)
	}

	resp := map[string]any{
		"conversation_id":     convID.String(),
		"version_number":      newVersionNumber,
		"previous_version":    compResult.PreviousVersion,
		"previous_score":      compResult.PreviousScore,
		"current_score":       compResult.CurrentScore,
		"score_delta":         compResult.ScoreDelta,
		"is_improved":         compResult.IsImproved,
		"fixed_errors":        compResult.FixedErrors,
		"still_unfixed":       compResult.StillUnfixed,
		"regressions":         compResult.Regressions,
		"remaining_errors":    compResult.RemainingErrors,
		"missing_information": compResult.MissingInformation,
		"comparison_summary":  compResult.ComparisonSummary,
		"is_comparison":       true,
		"audit":               currAudit,
		"record":              record,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// HandleGetVersions returns all versions for a given conversation.
// GET /api/verification/versions?conversation_id=...
func (h *Handler) HandleGetVersions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	convIDStr := r.URL.Query().Get("conversation_id")
	if convIDStr == "" {
		http.Error(w, "conversation_id query parameter is required", http.StatusBadRequest)
		return
	}

	convID, err := uuid.Parse(convIDStr)
	if err != nil {
		http.Error(w, "Invalid conversation_id UUID", http.StatusBadRequest)
		return
	}

	if h.DB == nil {
		http.Error(w, "Database not available", http.StatusServiceUnavailable)
		return
	}

	versions, err := h.Verifier.GetVersions(r.Context(), h.DB, convID)
	if err != nil {
		http.Error(w, "Failed to retrieve versions: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(versions)
}

func toJSONString(val any) (string, error) {
	if val == nil {
		return "{}", nil
	}
	switch v := val.(type) {
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return "{}", nil
		}
		return trimmed, nil
	default:
		bytes, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return string(bytes), nil
	}
}
