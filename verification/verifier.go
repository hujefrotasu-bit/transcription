package verification

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Verifier executes quality-control audits and comparative version analyses using Claude Fable.
type Verifier struct {
	Client *FableClient
}

// NewVerifier initializes a new Verifier instance.
func NewVerifier(client *FableClient) *Verifier {
	return &Verifier{Client: client}
}

// ─────────────────────────────────────────────
// System Prompts for Fable
// ─────────────────────────────────────────────

const auditSystemPrompt = `You are an expert Quality Control (QC) and Verification system for Minutes of Meeting.
Your mission is to compare the GENERATED_MEETING_MINUTES against the ORIGINAL_TRANSCRIPT and produce a rigorous, evidence-grounded audit report.

The ORIGINAL_TRANSCRIPT is the ultimate source of truth.
- Do NOT use outside knowledge.
- Do NOT assume facts not stated.
- Do NOT accept information merely because it sounds plausible or professional.

==================================================
1. CALIBRATED SCORING FORMULA (100 Points Total)
==================================================
Start from 100 points and apply deductions:
A. Grounding & Accuracy (Max 50 pts):
   - Deduct 15 pts per hallucinated fact, invented decision, or invented action item.
   - Deduct 10 pts per direct contradiction with the transcript.
   - Deduct 5 pts per distorted number or unverified causal claim.
B. Completeness & Recall (Max 30 pts):
   - Deduct 10 pts per omitted explicit decision.
   - Deduct 8 pts per omitted explicit action item.
   - Deduct 4 pts per omitted major discussion topic.
C. Attribution & Consistency (Max 20 pts):
   - Deduct 5 pts per wrong or unsupported attendee.
   - Deduct 5 pts per wrongly attributed action item owner.
   - Deduct 5 pts per unsupported or wrong deadline.

Final Score = Grounding + Completeness + Attribution (clamped between 0.0 and 100.0).

==================================================
2. AUDIT OUTPUT REQUIREMENTS
==================================================
Provide:
1. "score": Final computed float score.
2. "score_breakdown": Sub-scores for "grounding_and_accuracy", "completeness", "attribution", and "total_score".
3. "verified": true ONLY if score >= 90.0 AND zero high/medium severity errors exist.
4. "what_is_right": Array of successfully verified items with exact verbatim "transcript_evidence".
5. "what_is_wrong": Array of specific errors with:
   - "severity": "high", "medium", or "low"
   - "field": Target field path (e.g. "action_items[0].owner", "decisions[1]")
   - "generated_value": What the meeting minutes currently state
   - "problem": Exactly why it is inaccurate or hallucinated
   - "transcript_evidence": Verbatim quote from transcript (or "None found in transcript")
   - "correction_instruction": Clear, actionable guidance on how to fix it
6. "missing_information": Array of critical items discussed in transcript but missing from minutes.
7. "improvement_summary": Concise 2-3 sentence summary explaining what must be corrected.

Output ONLY valid JSON matching this schema. No markdown wrapping. No conversation.`

const compareSystemPrompt = `You are an expert Verification and Version Comparison system for Minutes of Meeting.
You are given:
- ORIGINAL_TRANSCRIPT (ultimate source of truth)
- PREVIOUS_MEETING_MINUTES (Version N-1)
- PREVIOUS_ERRORS (issues previously flagged in Version N-1)
- CURRENT_MEETING_MINUTES (Version N, the updated candidate)

Your job is to compare CURRENT_MEETING_MINUTES against PREVIOUS_MEETING_MINUTES and the transcript to determine if Version N actually IMPROVED:
1. FIXED ERRORS: Check which items from PREVIOUS_ERRORS have now been properly corrected in CURRENT_MEETING_MINUTES.
2. STILL UNFIXED: Check which previous errors still persist.
3. REGRESSIONS: Identify any NEW hallucinations, mistakes, or dropped accurate facts introduced in CURRENT_MEETING_MINUTES that were NOT present in PREVIOUS_MEETING_MINUTES.
4. RECALCULATED SCORE: Calculate the calibrated 0-100 score for CURRENT_MEETING_MINUTES using the standard formula:
   - Grounding & Accuracy (max 50)
   - Completeness & Recall (max 30)
   - Attribution & Consistency (max 20)
5. SCORE DELTA: current_score minus previous_score.
6. IS_IMPROVED: true ONLY if score_delta > 0 AND regressions array is empty.

Output ONLY valid JSON matching this schema:
{
  "previous_score": 0.0,
  "current_score": 0.0,
  "score_delta": 0.0,
  "is_improved": true,
  "fixed_errors": ["..."],
  "still_unfixed": [],
  "regressions": [],
  "remaining_errors": [],
  "missing_information": [],
  "comparison_summary": "..."
}

Output ONLY valid JSON. No markdown code blocks.`

// ─────────────────────────────────────────────
// Verification Methods
// ─────────────────────────────────────────────

// AuditMeetingMinutes audits meeting minutes JSON against the original transcript using Fable.
func (v *Verifier) AuditMeetingMinutes(ctx context.Context, originalTranscript, minutesJSON string) (*AuditResult, error) {
	userPrompt := fmt.Sprintf(`==================================================
INPUTS
======

ORIGINAL_TRANSCRIPT:
%s

GENERATED_MEETING_MINUTES:
%s`, originalTranscript, minutesJSON)

	respText, err := v.Client.CompleteChat(ctx, auditSystemPrompt, userPrompt)
	if err != nil {
		return nil, fmt.Errorf("Fable audit failed: %w", err)
	}

	cleanedJSON := cleanFableJSON(respText)

	var result AuditResult
	if err := json.Unmarshal([]byte(cleanedJSON), &result); err != nil {
		return nil, fmt.Errorf("failed to decode Fable audit result: %w (raw response: %s)", err, respText)
	}

	if result.Score < 0 {
		result.Score = 0
	} else if result.Score > 100 {
		result.Score = 100
	}

	hasCriticalErrors := false
	for _, e := range result.WhatIsWrong {
		if strings.EqualFold(e.Severity, "high") || strings.EqualFold(e.Severity, "medium") {
			hasCriticalErrors = true
			break
		}
	}
	if hasCriticalErrors || len(result.MissingInformation) > 0 {
		result.Verified = false
	}

	return &result, nil
}

// CompareVersions evaluates Version N against Version N-1 and the original transcript.
func (v *Verifier) CompareVersions(ctx context.Context, originalTranscript, prevMinutesJSON, newMinutesJSON string, prevErrors []VerificationError, prevScore float64) (*VersionComparisonResult, error) {
	prevErrorsBytes, _ := json.MarshalIndent(prevErrors, "", "  ")

	userPrompt := fmt.Sprintf(`==================================================
INPUTS
======

ORIGINAL_TRANSCRIPT:
%s

PREVIOUS_MEETING_MINUTES (Version N-1, Previous Score: %.1f):
%s

PREVIOUS_ERRORS:
%s

CURRENT_MEETING_MINUTES (Version N candidate):
%s`, originalTranscript, prevScore, prevMinutesJSON, string(prevErrorsBytes), newMinutesJSON)

	respText, err := v.Client.CompleteChat(ctx, compareSystemPrompt, userPrompt)
	if err != nil {
		return nil, fmt.Errorf("Fable version comparison failed: %w", err)
	}

	cleanedJSON := cleanFableJSON(respText)

	var result VersionComparisonResult
	if err := json.Unmarshal([]byte(cleanedJSON), &result); err != nil {
		return nil, fmt.Errorf("failed to decode Fable comparison result: %w (raw response: %s)", err, respText)
	}

	result.PreviousScore = prevScore
	result.ScoreDelta = result.CurrentScore - prevScore

	if len(result.Regressions) > 0 || result.ScoreDelta <= 0 {
		result.IsImproved = false
	} else {
		result.IsImproved = true
	}

	return &result, nil
}

// ─────────────────────────────────────────────
// Database Persistence (meeting_minutes_versions)
// ─────────────────────────────────────────────

// SaveVersion persists an audited meeting minutes version into meeting_minutes_versions.
func (v *Verifier) SaveVersion(ctx context.Context, pool *pgxpool.Pool, conversationID uuid.UUID, versionNumber int, minutesJSON string, audit *AuditResult, comp *VersionComparisonResult, tokenUsage ...*VersionTokenUsage) (*MeetingMinutesVersionRecord, error) {
	if pool == nil {
		return nil, fmt.Errorf("database connection pool cannot be nil")
	}

	recordID := uuid.New()
	now := time.Now()

	var minutesData map[string]any
	if err := json.Unmarshal([]byte(minutesJSON), &minutesData); err != nil {
		minutesData = map[string]any{"raw": minutesJSON}
	}

	var scoreBreakdownJSON, whatIsRightJSON, errorsJSON, missingInfoJSON []byte
	score := 0.0
	improvementSummary := ""

	if audit != nil {
		score = audit.Score
		improvementSummary = audit.ImprovementSummary
		scoreBreakdownJSON, _ = json.Marshal(audit.ScoreBreakdown)
		whatIsRightJSON, _ = json.Marshal(audit.WhatIsRight)
		errorsJSON, _ = json.Marshal(audit.WhatIsWrong)
		missingInfoJSON, _ = json.Marshal(audit.MissingInformation)
	} else {
		scoreBreakdownJSON = []byte("{}")
		whatIsRightJSON = []byte("[]")
		errorsJSON = []byte("[]")
		missingInfoJSON = []byte("[]")
		if comp != nil {
			score = comp.CurrentScore
			improvementSummary = comp.ComparisonSummary
		}
	}

	fixedErrorsJSON := []byte("[]")
	stillUnfixedJSON := []byte("[]")
	regressionsJSON := []byte("[]")
	scoreDelta := 0.0
	isImproved := false

	if comp != nil {
		fixedErrorsJSON, _ = json.Marshal(comp.FixedErrors)
		stillUnfixedJSON, _ = json.Marshal(comp.StillUnfixed)
		regressionsJSON, _ = json.Marshal(comp.Regressions)
		scoreDelta = comp.ScoreDelta
		isImproved = comp.IsImproved
	}

	modelName := "gemini-3.5-flash-lite"
	inputTokens := 0
	outputTokens := 0
	totalTokens := 0
	estimatedCostINR := 0.0
	estimatedCostUSD := 0.0

	if len(tokenUsage) > 0 && tokenUsage[0] != nil {
		u := tokenUsage[0]
		if u.Model != "" {
			modelName = u.Model
		}
		inputTokens = u.InputTokens
		outputTokens = u.OutputTokens
		totalTokens = u.TotalTokens
		estimatedCostINR = u.EstimatedCostINR
		estimatedCostUSD = u.EstimatedCostUSD
	}

	query := `INSERT INTO meeting_minutes_versions (
		id, conversation_id, version_number, meeting_minutes_data, score, score_breakdown,
		what_is_right, errors, missing_information, improvement_summary,
		fixed_errors, still_unfixed, regressions, score_delta, is_improved,
		model, input_tokens, output_tokens, total_tokens, estimated_cost_inr, estimated_cost_usd,
		created_at
	) VALUES (
		$1, $2, $3, $4, $5, $6,
		$7, $8, $9, $10,
		$11, $12, $13, $14, $15,
		$16, $17, $18, $19, $20, $21,
		$22
	)
	ON CONFLICT (conversation_id, version_number)
	DO UPDATE SET
		meeting_minutes_data = EXCLUDED.meeting_minutes_data,
		score = EXCLUDED.score,
		score_breakdown = EXCLUDED.score_breakdown,
		what_is_right = EXCLUDED.what_is_right,
		errors = EXCLUDED.errors,
		missing_information = EXCLUDED.missing_information,
		improvement_summary = EXCLUDED.improvement_summary,
		fixed_errors = EXCLUDED.fixed_errors,
		still_unfixed = EXCLUDED.still_unfixed,
		regressions = EXCLUDED.regressions,
		score_delta = EXCLUDED.score_delta,
		is_improved = EXCLUDED.is_improved,
		model = EXCLUDED.model,
		input_tokens = EXCLUDED.input_tokens,
		output_tokens = EXCLUDED.output_tokens,
		total_tokens = EXCLUDED.total_tokens,
		estimated_cost_inr = EXCLUDED.estimated_cost_inr,
		estimated_cost_usd = EXCLUDED.estimated_cost_usd,
		created_at = EXCLUDED.created_at
	RETURNING id, created_at`

	err := pool.QueryRow(ctx, query,
		recordID, conversationID, versionNumber, minutesData, score, scoreBreakdownJSON,
		whatIsRightJSON, errorsJSON, missingInfoJSON, improvementSummary,
		fixedErrorsJSON, stillUnfixedJSON, regressionsJSON, scoreDelta, isImproved,
		modelName, inputTokens, outputTokens, totalTokens, estimatedCostINR, estimatedCostUSD,
		now,
	).Scan(&recordID, &now)

	if err != nil {
		return nil, fmt.Errorf("failed to save meeting minutes version: %w", err)
	}

	record := &MeetingMinutesVersionRecord{
		ID:                 recordID,
		ConversationID:     conversationID,
		VersionNumber:      versionNumber,
		MeetingMinutesData: minutesData,
		Score:              score,
		ImprovementSummary: improvementSummary,
		Model:              modelName,
		InputTokens:        inputTokens,
		OutputTokens:       outputTokens,
		TotalTokens:        totalTokens,
		EstimatedCostINR:   estimatedCostINR,
		EstimatedCostUSD:   estimatedCostUSD,
		CreatedAt:          now,
	}

	if audit != nil {
		record.ScoreBreakdown = audit.ScoreBreakdown
		record.WhatIsRight = audit.WhatIsRight
		record.Errors = audit.WhatIsWrong
		record.MissingInformation = audit.MissingInformation
	}

	if comp != nil {
		record.FixedErrors = comp.FixedErrors
		record.StillUnfixed = comp.StillUnfixed
		record.Regressions = comp.Regressions
		record.ScoreDelta = comp.ScoreDelta
		record.IsImproved = comp.IsImproved
	}

	return record, nil
}

// GetLatestVersion retrieves the most recent version record for a conversation.
func (v *Verifier) GetLatestVersion(ctx context.Context, pool *pgxpool.Pool, conversationID uuid.UUID) (*MeetingMinutesVersionRecord, error) {
	if pool == nil {
		return nil, fmt.Errorf("database connection pool cannot be nil")
	}

	query := `SELECT id, conversation_id, version_number, meeting_minutes_data, score, score_breakdown,
	                 what_is_right, errors, missing_information, improvement_summary,
	                 fixed_errors, still_unfixed, regressions, score_delta, is_improved,
	                 COALESCE(model, 'gemini-3.5-flash-lite'), COALESCE(input_tokens, 0), COALESCE(output_tokens, 0),
	                 COALESCE(total_tokens, 0), COALESCE(estimated_cost_inr, 0), COALESCE(estimated_cost_usd, 0),
	                 created_at
	          FROM meeting_minutes_versions
	          WHERE conversation_id = $1
	          ORDER BY version_number DESC
	          LIMIT 1`

	var rec MeetingMinutesVersionRecord
	var scoreBreakdownBytes, whatIsRightBytes, errorsBytes, missingInfoBytes []byte
	var fixedErrorsBytes, stillUnfixedBytes, regressionsBytes []byte

	err := pool.QueryRow(ctx, query, conversationID).Scan(
		&rec.ID, &rec.ConversationID, &rec.VersionNumber, &rec.MeetingMinutesData, &rec.Score,
		&scoreBreakdownBytes, &whatIsRightBytes, &errorsBytes, &missingInfoBytes, &rec.ImprovementSummary,
		&fixedErrorsBytes, &stillUnfixedBytes, &regressionsBytes, &rec.ScoreDelta, &rec.IsImproved,
		&rec.Model, &rec.InputTokens, &rec.OutputTokens, &rec.TotalTokens, &rec.EstimatedCostINR, &rec.EstimatedCostUSD,
		&rec.CreatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil // No previous version exists yet
		}
		return nil, fmt.Errorf("failed to query latest version: %w", err)
	}

	_ = json.Unmarshal(scoreBreakdownBytes, &rec.ScoreBreakdown)
	_ = json.Unmarshal(whatIsRightBytes, &rec.WhatIsRight)
	_ = json.Unmarshal(errorsBytes, &rec.Errors)
	_ = json.Unmarshal(missingInfoBytes, &rec.MissingInformation)
	_ = json.Unmarshal(fixedErrorsBytes, &rec.FixedErrors)
	_ = json.Unmarshal(stillUnfixedBytes, &rec.StillUnfixed)
	_ = json.Unmarshal(regressionsBytes, &rec.Regressions)

	return &rec, nil
}

// GetVersions retrieves all recorded versions for a conversation ordered by version number.
func (v *Verifier) GetVersions(ctx context.Context, pool *pgxpool.Pool, conversationID uuid.UUID) ([]MeetingMinutesVersionRecord, error) {
	if pool == nil {
		return nil, fmt.Errorf("database connection pool cannot be nil")
	}

	query := `SELECT id, conversation_id, version_number, meeting_minutes_data, score, score_breakdown,
	                 what_is_right, errors, missing_information, improvement_summary,
	                 fixed_errors, still_unfixed, regressions, score_delta, is_improved,
	                 COALESCE(model, 'gemini-3.5-flash-lite'), COALESCE(input_tokens, 0), COALESCE(output_tokens, 0),
	                 COALESCE(total_tokens, 0), COALESCE(estimated_cost_inr, 0), COALESCE(estimated_cost_usd, 0),
	                 created_at
	          FROM meeting_minutes_versions
	          WHERE conversation_id = $1
	          ORDER BY version_number ASC`

	rows, err := pool.Query(ctx, query, conversationID)
	if err != nil {
		return nil, fmt.Errorf("failed to query meeting_minutes_versions: %w", err)
	}
	defer rows.Close()

	var versions []MeetingMinutesVersionRecord
	for rows.Next() {
		var rec MeetingMinutesVersionRecord
		var scoreBreakdownBytes, whatIsRightBytes, errorsBytes, missingInfoBytes []byte
		var fixedErrorsBytes, stillUnfixedBytes, regressionsBytes []byte

		err := rows.Scan(
			&rec.ID, &rec.ConversationID, &rec.VersionNumber, &rec.MeetingMinutesData, &rec.Score,
			&scoreBreakdownBytes, &whatIsRightBytes, &errorsBytes, &missingInfoBytes, &rec.ImprovementSummary,
			&fixedErrorsBytes, &stillUnfixedBytes, &regressionsBytes, &rec.ScoreDelta, &rec.IsImproved,
			&rec.Model, &rec.InputTokens, &rec.OutputTokens, &rec.TotalTokens, &rec.EstimatedCostINR, &rec.EstimatedCostUSD,
			&rec.CreatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan meeting_minutes_versions row: %w", err)
		}

		_ = json.Unmarshal(scoreBreakdownBytes, &rec.ScoreBreakdown)
		_ = json.Unmarshal(whatIsRightBytes, &rec.WhatIsRight)
		_ = json.Unmarshal(errorsBytes, &rec.Errors)
		_ = json.Unmarshal(missingInfoBytes, &rec.MissingInformation)
		_ = json.Unmarshal(fixedErrorsBytes, &rec.FixedErrors)
		_ = json.Unmarshal(stillUnfixedBytes, &rec.StillUnfixed)
		_ = json.Unmarshal(regressionsBytes, &rec.Regressions)

		versions = append(versions, rec)
	}

	return versions, nil
}

func cleanFableJSON(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "```") {
		lines := strings.Split(raw, "\n")
		if len(lines) >= 2 {
			if strings.HasPrefix(lines[0], "```") {
				lines = lines[1:]
			}
			if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
				lines = lines[:len(lines)-1]
			}
			raw = strings.Join(lines, "\n")
		}
	}
	return strings.TrimSpace(raw)
}
