package verification

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

// ScoreBreakdown represents the calibrated 100-point scoring formula.
type ScoreBreakdown struct {
	GroundingAndAccuracy float64 `json:"grounding_and_accuracy"` // Max 50 pts: factual grounding, no hallucinations
	Completeness         float64 `json:"completeness"`           // Max 30 pts: all decisions, actions, and key points captured
	Attribution          float64 `json:"attribution"`            // Max 20 pts: correct owners, dates, attendees
	TotalScore           float64 `json:"total_score"`            // 0 - 100
}

// VerifiedItem represents a claim or fact in the Meeting Minutes that was verified as accurate against the transcript.
type VerifiedItem struct {
	Field              string `json:"field"`               // e.g. "decisions", "action_items", "attendees"
	Claim              string `json:"claim"`               // The verified statement or extraction
	TranscriptEvidence string `json:"transcript_evidence"` // Verbatim quote from original transcript
}

func (v *VerifiedItem) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		v.Field = "verified"
		v.Claim = s
		return nil
	}
	type Alias VerifiedItem
	var a Alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*v = VerifiedItem(a)
	return nil
}

// VerificationError describes a specific hallucination, contradiction, or mistake.
type VerificationError struct {
	Severity              string `json:"severity"`               // "high", "medium", "low"
	Field                 string `json:"field"`                  // e.g. "action_items[0].owner", "decisions[1]"
	GeneratedValue        string `json:"generated_value"`        // What the meeting minutes currently state
	Problem               string `json:"problem"`                // Exactly why it is inaccurate or hallucinated
	TranscriptEvidence    string `json:"transcript_evidence"`    // What the transcript actually states (or "None found")
	CorrectionInstruction string `json:"correction_instruction"` // Exact instruction to fix it
}

func (e *VerificationError) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		e.Severity = "medium"
		e.Problem = s
		return nil
	}
	type Alias VerificationError
	var a Alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*e = VerificationError(a)
	return nil
}

// MissingInformation documents crucial items present in the transcript but omitted from the Meeting Minutes.
type MissingInformation struct {
	Field              string `json:"field"`               // e.g. "decisions", "action_items"
	Information        string `json:"information"`         // What was omitted
	TranscriptEvidence string `json:"transcript_evidence"` // Transcript quote proving it was discussed
}

func (m *MissingInformation) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		m.Field = "omission"
		m.Information = s
		return nil
	}
	type Alias MissingInformation
	var a Alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*m = MissingInformation(a)
	return nil
}

// AuditResult is the complete Fable QC evaluation of a Meeting Minutes version against the original transcript.
type AuditResult struct {
	Score              float64              `json:"score"`               // Calibrated 0 - 100
	ScoreBreakdown     ScoreBreakdown       `json:"score_breakdown"`     // Sub-scores
	Verified           bool                 `json:"verified"`            // True only if high/med errors == 0
	WhatIsRight        []VerifiedItem       `json:"what_is_right"`       // List of accurate claims
	WhatIsWrong        []VerificationError  `json:"what_is_wrong"`       // List of errors
	MissingInformation []MissingInformation `json:"missing_information"` // List of omissions
	ImprovementSummary string               `json:"improvement_summary"` // Summary of what needs fixing
}

// VersionComparisonResult compares a candidate Meeting Minutes (vN) against a previous version (vN-1) and the transcript.
type VersionComparisonResult struct {
	PreviousVersion    int      `json:"previous_version"`
	CurrentVersion     int      `json:"current_version"`
	PreviousScore      float64  `json:"previous_score"`
	CurrentScore       float64  `json:"current_score"`
	ScoreDelta         float64  `json:"score_delta"`      // positive = improvement, negative = regression
	IsImproved         bool     `json:"is_improved"`
	FixedErrors        []string `json:"fixed_errors"`     // Errors from previous version successfully resolved
	StillUnfixed       []any    `json:"still_unfixed"`    // Previous errors still unresolved (string or object)
	Regressions        []any    `json:"regressions"`      // New errors introduced in current version (string or object)
	RemainingErrors    []any    `json:"remaining_errors"` // All active errors in current version (string or object)
	MissingInformation []any    `json:"missing_information"`
	ComparisonSummary  string   `json:"comparison_summary"` // Human-readable delta explanation
}

// PopulateFromAudit populates RemainingErrors and MissingInformation from a full AuditResult.
func (c *VersionComparisonResult) PopulateFromAudit(audit *AuditResult) {
	if c == nil || audit == nil {
		return
	}
	rem := make([]any, len(audit.WhatIsWrong))
	for i, v := range audit.WhatIsWrong {
		rem[i] = v
	}
	c.RemainingErrors = rem

	miss := make([]any, len(audit.MissingInformation))
	for i, v := range audit.MissingInformation {
		miss[i] = v
	}
	c.MissingInformation = miss
}

// ToVerificationErrors converts RemainingErrors into []VerificationError slice.
func (c *VersionComparisonResult) ToVerificationErrors() []VerificationError {
	if c == nil || len(c.RemainingErrors) == 0 {
		return nil
	}
	var errs []VerificationError
	for _, item := range c.RemainingErrors {
		switch v := item.(type) {
		case string:
			errs = append(errs, VerificationError{
				Severity: "medium",
				Problem:  v,
			})
		case VerificationError:
			errs = append(errs, v)
		case map[string]any:
			var ve VerificationError
			b, _ := json.Marshal(v)
			if err := json.Unmarshal(b, &ve); err == nil && ve.Problem != "" {
				errs = append(errs, ve)
			} else if prob, ok := v["problem"].(string); ok {
				errs = append(errs, VerificationError{Severity: "medium", Problem: prob})
			}
		}
	}
	return errs
}

// VersionTokenUsage holds token consumption and estimated cost details for a version.
type VersionTokenUsage struct {
	Model            string  `json:"model"`
	InputTokens      int     `json:"input_tokens"`
	OutputTokens     int     `json:"output_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	EstimatedCostINR float64 `json:"estimated_cost_inr"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
}

// MeetingMinutesVersionRecord maps to the meeting_minutes_versions PostgreSQL table.
type MeetingMinutesVersionRecord struct {
	ID                 uuid.UUID           `json:"id"`
	ConversationID     uuid.UUID           `json:"conversation_id"`
	VersionNumber      int                 `json:"version_number"`
	MeetingMinutesData map[string]any      `json:"meeting_minutes_data"`
	Score              float64             `json:"score"`
	ScoreBreakdown     ScoreBreakdown      `json:"score_breakdown"`
	WhatIsRight        []VerifiedItem      `json:"what_is_right"`
	Errors             []VerificationError `json:"errors"`
	MissingInformation []MissingInformation `json:"missing_information"`
	ImprovementSummary string              `json:"improvement_summary"`
	FixedErrors        []string            `json:"fixed_errors"`
	StillUnfixed       []any               `json:"still_unfixed"`
	Regressions        []any               `json:"regressions"`
	ScoreDelta         float64             `json:"score_delta"`
	IsImproved         bool                `json:"is_improved"`
	Model              string              `json:"model"`
	InputTokens        int                 `json:"input_tokens"`
	OutputTokens       int                 `json:"output_tokens"`
	TotalTokens        int                 `json:"total_tokens"`
	EstimatedCostINR   float64             `json:"estimated_cost_inr"`
	EstimatedCostUSD   float64             `json:"estimated_cost_usd"`
	CreatedAt          time.Time           `json:"created_at"`
}
