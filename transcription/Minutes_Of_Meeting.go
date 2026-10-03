package transcription

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ─────────────────────────────────────────────
// Domain Models for Minutes of Meeting
// ─────────────────────────────────────────────

type MeetingDetails struct {
	Title       *string `json:"title"`
	Date        *string `json:"date"`
	Time        *string `json:"time"`
	Location    *string `json:"location"`
	MeetingType *string `json:"meeting_type"`
}

type Attendee struct {
	Name        string  `json:"name"`
	Designation *string `json:"designation"`
}

type DiscussionPoint struct {
	Topic      string `json:"topic"`
	Discussion string `json:"discussion"`
}

type Decision struct {
	Decision string `json:"decision"`
	Remarks  string `json:"remarks"`
}

type ActionItem struct {
	ActionItem string  `json:"action_item"`
	Owner      *string `json:"owner"`
	Priority   *string `json:"priority"`
	DueDate    *string `json:"due_date"`
	Status     *string `json:"status"`
}

type RiskIssueDependency struct {
	Issue          string  `json:"issue"`
	Owner          *string `json:"owner"`
	RequiredAction string  `json:"required_action"`
}

type NextMeeting struct {
	Date   *string `json:"date"`
	Time   *string `json:"time"`
	Agenda *string `json:"agenda"`
}

// UnmarshalJSON allows NextMeeting to accept agenda as either a string, an array of strings, or null.
func (nm *NextMeeting) UnmarshalJSON(data []byte) error {
	var raw struct {
		Date   *string `json:"date"`
		Time   *string `json:"time"`
		Agenda any     `json:"agenda"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	nm.Date = raw.Date
	nm.Time = raw.Time
	if raw.Agenda != nil {
		switch v := raw.Agenda.(type) {
		case string:
			trimmed := strings.TrimSpace(v)
			if trimmed != "" && trimmed != "null" && trimmed != "Unknown" {
				nm.Agenda = &trimmed
			}
		case []any:
			var items []string
			for _, item := range v {
				if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
					items = append(items, strings.TrimSpace(s))
				}
			}
			if len(items) > 0 {
				combined := strings.Join(items, "; ")
				nm.Agenda = &combined
			}
		}
	}
	return nil
}

type MeetingMinutes struct {
	Meeting                 MeetingDetails        `json:"meeting"`
	Attendees               []Attendee            `json:"attendees"`
	Chairperson             *string               `json:"chairperson"`
	MinutesPreparedBy       *string               `json:"minutes_prepared_by"`
	Agenda                  []string              `json:"agenda"`
	DiscussionPoints        []DiscussionPoint     `json:"discussion_points"`
	Decisions               []Decision            `json:"decisions"`
	ActionItems             []ActionItem          `json:"action_items"`
	RisksIssuesDependencies []RiskIssueDependency `json:"risks_issues_dependencies"`
	NextMeeting             NextMeeting           `json:"next_meeting"`
}

// TokenUsage tracks token consumption and estimated costs in INR and USD.
type TokenUsage struct {
	Model            string  `json:"model"`
	InputTokens      int     `json:"input_tokens"`
	OutputTokens     int     `json:"output_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	EstimatedCostINR float64 `json:"estimated_cost_inr"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
}

type MeetingMinutesRecord struct {
	ID               uuid.UUID      `json:"id"`
	ConversationID   uuid.UUID      `json:"conversation_id"`
	Data             MeetingMinutes `json:"data"`
	Model            string         `json:"model"`
	InputTokens      int            `json:"input_tokens"`
	OutputTokens     int            `json:"output_tokens"`
	TotalTokens      int            `json:"total_tokens"`
	EstimatedCostINR float64        `json:"estimated_cost_inr"`
	EstimatedCostUSD float64        `json:"estimated_cost_usd"`
	CreatedAt        time.Time      `json:"created_at"`
}

// ExtractionResult holds both the resolved named transcript and structured MoM.
type ExtractionResult struct {
	NamedTranscript string         `json:"named_transcript"`
	MeetingMinutes  MeetingMinutes `json:"meeting_minutes"`
}

// MeetingMinutesService defines the contract for MoM extraction and persistence.
type MeetingMinutesService interface {
	ExtractMeetingMinutes(ctx context.Context, transcript string) (*MeetingMinutes, string, *TokenUsage, error)
	SaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, mom *MeetingMinutes, usage *TokenUsage) (*MeetingMinutesRecord, error)
	ExtractAndSaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, transcript string) (*MeetingMinutes, string, *TokenUsage, error)
}

// GeminiService implements MeetingMinutesService using Google's Gemini Interactions API.
type GeminiMeetingMinutesService struct {
	apiKey     string
	model      string
	db         *pgxpool.Pool
	httpClient *http.Client
}

// NewGeminiService creates a new GeminiService with the given PostgreSQL connection pool.
// Reads GEMINI_MOM_MODEL or GEMINI_TOPIC_MODEL (defaults to gemini-3.6-flash).
func NewGeminiMeetingMinutesService(db *pgxpool.Pool) (*GeminiMeetingMinutesService, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := os.Getenv("GEMINI_MOM_MODEL")
	if model == "" {
		model = os.Getenv("GEMINI_TOPIC_MODEL")
	}
	if model == "" {
		model = "gemini-3.5-flash-lite"
	}

	return &GeminiMeetingMinutesService{
		apiKey:     key,
		model:      model,
		db:         db,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}, nil
}

// Model returns the Gemini model name in use for meeting minutes.
func (s *GeminiMeetingMinutesService) Model() string {
	return s.model
}

// ─────────────────────────────────────────────
// Gemini Interactions API Schema (Local types)
// ─────────────────────────────────────────────

type momInteractionsRequest struct {
	Model            string               `json:"model"`
	Input            string               `json:"input"`
	ResponseFormat   *momResponseFormat   `json:"response_format,omitempty"`
	GenerationConfig *momGenerationConfig `json:"generation_config,omitempty"`
}

type momGenerationConfig struct {
	Temperature *float64 `json:"temperature,omitempty"`
}

type momResponseFormat struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type"`
}

type momInteractionsResponse struct {
	ID            string               `json:"id"`
	Status        string               `json:"status"`
	OutputText    string               `json:"output_text,omitempty"`
	Steps         []momInteractionStep `json:"steps"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"prompt_token_count"`
		CandidatesTokenCount int `json:"candidates_token_count"`
		TotalTokenCount      int `json:"total_token_count"`
		InputTokens          int `json:"input_tokens"`
		OutputTokens         int `json:"output_tokens"`
	} `json:"usage_metadata,omitempty"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
	Error *struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type momInteractionStep struct {
	Type    string                  `json:"type"`
	Content []momInteractionContent `json:"content"`
}

type momInteractionContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ─────────────────────────────────────────────
// Extraction & Persistence Logic
// ─────────────────────────────────────────────

// ExtractMeetingMinutes sends the transcript to Gemini, resolves speaker names in both transcript & MoM, and parses the structured result.
func (s *GeminiMeetingMinutesService) ExtractMeetingMinutes(ctx context.Context, transcript string) (*MeetingMinutes, string, *TokenUsage, error) {
	trimmedTranscript := strings.TrimSpace(transcript)
	if trimmedTranscript == "" {
		return nil, "", nil, fmt.Errorf("transcript cannot be empty")
	}

	prompt := fmt.Sprintf(`You are an expert AI system for extracting accurate, professional, and audit-grade Minutes of Meeting (MoM) from meeting transcripts.

The transcript provided at the end of this prompt is the ONLY source of truth. Use NO outside knowledge, NO unstated business assumptions, and NO speculation.

==================================================
1. SOURCE OF TRUTH & EVIDENCE FIRST
==================================================
Every extracted attendee, agenda topic, discussion point, decision, action item, owner, deadline, risk, and next-meeting detail MUST be directly grounded in the dialogue. If the transcript is silent or ambiguous, use null or [].
Accuracy is more important than completeness. When uncertain, prefer omission or null over guessing.

==================================================
2. ATTENDEES (STRICT PRESENCE & COMPLETE ROSTER RULE)
==================================================
* CRITICAL ATTENDEE RULE: Include in "attendees" ONLY individuals who were ACTUALLY PRESENT in the conversation and actively participated / spoke in the dialogue.
* ALL SPEAKING PARTICIPANTS MUST BE IN ATTENDEES:
  - If some participants speak under generic tags (e.g. "Speaker 1", "Speaker 2", "Speaker 3") because their personal names were not introduced, you MUST include them in "attendees" (e.g. name: "Speaker 1", "Speaker 2") so that EVERY speaking participant and every action item owner is accounted for in the attendee list. Never assign an action item to a speaker who is not listed in "attendees".
* NON-ATTENDEES MUST BE STRICTLY EXCLUDED:
  - If someone is NOT in the conversation, THEY MUST NOT BE IN "attendees".
  - Merely mentioning a person's name does NOT make them an attendee (e.g. "ask Clive", "check with Sarah in finance", "our contact at Acme", "the client said", "we need an external researcher"). If they did not actively participate in the conversation, DO NOT add them to "attendees".
  - Anyone absent, off sick, on leave, working remotely without attending, or who sent apologies MUST BE STRICTLY EXCLUDED from "attendees". (Record absences in "discussion_points", NEVER in "attendees").
  - Do NOT list "Narrator", generic groups ("All Staff"), or third-party companies as attendees.
* CHAIRPERSON ATTRIBUTION: Set "chairperson" to null unless someone is explicitly designated as chair or explicitly acts as the sole meeting facilitator/lead. Never guess or attribute chairperson to a participant simply because they gave a report, update, or spoke on technical issues.
* LATE ARRIVALS: Only if someone actually arrives during the meeting and actively speaks in the dialogue are they an attendee. If they never arrived or never spoke, they are NOT an attendee.

==================================================
3. AGENDA TOPICS
==================================================
* Extract all formal agenda items, review items, and topics raised for discussion (e.g. Apologies for Absence, Matters Arising, specific agenda topics, Training, and Any Other Business).
* If "matters arising" was raised and participants confirmed none, include it in agenda and record "None" in discussion points.

==================================================
4. DISCUSSION POINTS & DISPUTED NUMBERS
==================================================
* Maintain objective, neutral summaries of substantive discussions.
* NO UNVERIFIED CAUSALITY: Never state that Event A caused Event B unless explicit causal words ("caused by", "because of", "as a result of") are spoken. Report adjacent events as separate factual items (e.g. if the transcript states that the office network went down, the router restarted twice, and IT suspects an ISP problem, report these as separate facts without stating the outage was "due to a router issue").
* NUMBERS & FINANCIAL DATA: Copy numbers verbatim. If units or labels are not stated, do not guess them (e.g. do not label a figure as "revenue difference" if the transcript only clarifies the number as 11).

==================================================
5. DECISIONS VS SUGGESTIONS (STRICT NO-HALLUCINATION RULE)
==================================================
* Decisions require explicit consensus, formal approval, or chair ruling (e.g. "Agreed", "That settles that", "agreed by the group", "let's go with X").
* EXPLORATORY PROPOSALS & BRAINSTORMING ARE NOT DECISIONS:
  - In product design, architectural reviews, or brainstorming sessions (e.g. discussing remote control features, D-pads vs push buttons, touch screens, ergonomic biomorphic shapes, materials), exploratory suggestions are NOT decisions!
  - Casual agreement or positive reactions (e.g. "that sounds like a good strong idea", "I don't see why not", "could be an option", "it's worth considering") do NOT make it a decision.
  - Summarize these ideas in "discussion_points", NEVER in "decisions".
  - If no formal, binding decision was finalized during the dialogue, record "decisions": []. NEVER record an open proposal as an approved decision.
* DECISIONS ON PROCEDURE & VOTING: When participants agree on a course of action for collecting feedback or making selections (e.g. agreeing to collect staff morale options and put them to a company-wide vote), this is an agreed Decision on procedure and MUST be recorded in decisions.
* ARITHMETIC CONSTRAINTS: Respect explicitly stated totals. Never record decisions that invent allocations exceeding established limits (e.g. allocating 9 spaces when the transcript establishes only 5 total spaces). Distinguish initial brainstormed numbers (like 4 sales staff) from finalized allocations (3 for visitors, 2 for Sue and Jason).

==================================================
6. ACTION ITEMS, OWNERS & DEADLINES
==================================================
* EXHAUSTIVE COMMITMENT SWEEP: Scan the transcript thoroughly for all explicit verbal commitments, directives, or agreed tasks starting with active verbs ("I will", "I'll", "let's", "can you", "we need to", "make sure they're scheduled").
* ONE ACTION ITEM PER DISTINCT TASK (DO NOT COMBINE TASKS): Never merge two separate tasks into a single action item, especially if spoken by or assigned to different people. If Person A is asked to do Task 1 and Person B is asked to do Task 2, create TWO separate action items. Never attribute Person B's task to Person A.
* NO DUPLICATE OR REDUNDANT ACTION ITEMS: If an attendee commits to a task initially and then reiterates or confirms that commitment later in the meeting (e.g. "I'll check formatting"), record it as a SINGLE action item. Do not create duplicate action items for the same underlying work.
* THOROUGH COMPLETENESS: Capture all explicit directives and commitments, including technical investigations, bundle analysis, and final reviews agreed to by participants.
* INDIVIDUAL SPOKEN COMMITMENTS: Whenever an attendee states a personal commitment to handle an action (e.g. "I'll coordinate letting the next member know...", "I'll speak with Clive and let you know date..."), capture it as an action item with that person as owner.
* FUTURE SCHEDULING COMMITMENTS: Directives to schedule or carry forward an issue for a subsequent meeting (e.g. "make sure they're scheduled for the next meeting") MUST be recorded as an action item (owner: null if unassigned) AND noted under next_meeting.agenda.
* ACTION DESCRIPTIONS: Begin each action item with a clear active verb (e.g. "Circulate...", "Submit...", "Speak with...", "Schedule...", "Coordinate...").
* OWNERS: Assign an owner ONLY if someone explicitly volunteered (e.g. "I'll send it", "I'll speak with...") or was directly assigned by the chair without objection.
  - Every assigned owner MUST be a valid attendee listed in "attendees". If a task is volunteered by "Speaker 2", owner must be "Speaker 2" and "Speaker 2" must be in "attendees".
  - If a task is assigned to everyone present or company staff (e.g. coming up with 4 ideas), record owner as "All Attendees" or "All Staff".
  - If no specific person volunteered or was assigned, record owner: null.
  - NEVER infer an owner based on who complained or brought up the problem.
* DEADLINES & DUE DATES:
  - When an explicit time or deadline is stated (e.g. "by seven", "by eight", "tonight after nine", "first thing tomorrow", "within two days", "next meeting"), record it in due_date. Do NOT leave due_date as null when an explicit timeframe was spoken.
  - VERBATIM MERIDIAN (DO NOT INVENT AM/PM): If the transcript states "by seven" or "after nine" without specifying AM or PM, record it exactly as stated (e.g. "By 7", "Tonight after 9"). Do NOT append "PM" or "AM" unless explicitly spoken in the dialogue.
  - TENTATIVE SUGGESTION RULE: If a timeframe or deadline was merely suggested by another person (e.g. "maybe in the next two weeks?") but was NOT confirmed or accepted by the task owner or chair, record due_date: null. Do NOT turn tentative suggestions into confirmed deadlines.
* STATUS: Set status to "Pending" for all newly agreed action items.

==================================================
7. RISKS, ISSUES & DEPENDENCIES
==================================================
* Record explicit operational risks, blockers, or problems raised by participants (e.g. parking shortages/disputes, low morale, software training gaps, job security/restructuring concerns).
* Do NOT invent causal links between separate issues.
* owner and required_action MUST be null unless explicitly assigned in the transcript.

==================================================
8. NEXT MEETING
==================================================
* If topics are explicitly requested to be scheduled for the next meeting (e.g. cleanliness issues), record them under agenda as scheduled topics.
* Do not present a single carried-forward item as the entire exclusive agenda.
* date and time must be null unless explicitly confirmed.
* VERBATIM TIME (DO NOT INVENT AM/PM): If the meeting time is stated as "Three", record time as "3:00" or "Three" without assuming "PM" unless explicitly stated.

==================================================
9. OUTPUT FORMAT
==================================================
Return ONLY valid JSON matching this exact structure:

{
  "named_transcript": "The full original dialogue transcript with generic speaker labels replaced by actual identified names (must be the complete full transcript, never just a title)",
  "meeting_minutes": {
    "meeting": {
      "title": null,
      "date": null,
      "time": null,
      "location": null,
      "meeting_type": null
    },
    "attendees": [
      {
        "name": "Full Name",
        "designation": null
      }
    ],
    "chairperson": null,
    "minutes_prepared_by": null,
    "agenda": [],
    "discussion_points": [
      {
        "topic": "Topic Name",
        "discussion": "Factual summary..."
      }
    ],
    "decisions": [
      {
        "decision": "Agreed outcome...",
        "remarks": "Context or details..."
      }
    ],
    "action_items": [
      {
        "action_item": "Active verb description...",
        "owner": null,
        "priority": null,
        "due_date": null,
        "status": "Pending"
      }
    ],
    "risks_issues_dependencies": [
      {
        "issue": "Explicit risk or issue...",
        "owner": null,
        "required_action": null
      }
    ],
    "next_meeting": {
      "date": null,
      "time": null,
      "agenda": null
    }
  }
}

==================================================
TRANSCRIPT
==========

%s`, trimmedTranscript)

	zeroTemp := 0.0
	reqBody := momInteractionsRequest{
		Model: s.model,
		Input: prompt,
		ResponseFormat: &momResponseFormat{
			Type:     "text",
			MIMEType: "application/json",
		},
		GenerationConfig: &momGenerationConfig{
			Temperature: &zeroTemp,
		},
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, "", nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	const apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"

	// Retry loop with exponential backoff for transient 503 / 429 errors.
	var respBody []byte
	maxRetries := 3
	backoff := 1 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBytes))
		if err != nil {
			return nil, "", nil, fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("x-goog-api-key", s.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := s.httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, "", nil, fmt.Errorf("gemini interactions request canceled: %w", ctx.Err())
			}
			if attempt == maxRetries {
				return nil, "", nil, fmt.Errorf("gemini interactions request failed after %d retries: %w", maxRetries, err)
			}
			log.Printf("Gemini interactions request error (attempt %d/%d): %v, retrying in %v...", attempt+1, maxRetries, err, backoff)
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, "", nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if resp.StatusCode == http.StatusServiceUnavailable || resp.StatusCode == http.StatusTooManyRequests {
			if attempt < maxRetries {
				log.Printf("Gemini interactions status %d (attempt %d/%d), retrying in %v...", resp.StatusCode, attempt+1, maxRetries, backoff)
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
		}

		if resp.StatusCode != http.StatusOK {
			var errResp struct {
				Error *struct {
					Code    any    `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if json.Unmarshal(body, &errResp) == nil && errResp.Error != nil {
				return nil, "", nil, fmt.Errorf("gemini API error (%v): %s", errResp.Error.Code, errResp.Error.Message)
			}
			return nil, "", nil, fmt.Errorf("gemini API returned status %d: %s", resp.StatusCode, string(body))
		}

		respBody = body
		break
	}

	var intResp momInteractionsResponse
	if err = json.Unmarshal(respBody, &intResp); err != nil {
		return nil, "", nil, fmt.Errorf("failed to parse Gemini interactions response: %w", err)
	}

	if intResp.Error != nil {
		return nil, "", nil, fmt.Errorf("gemini API error (%v): %s", intResp.Error.Code, intResp.Error.Message)
	}

	// Extract generated text from steps[].content[].text or output_text
	var rawJSON string
	for _, step := range intResp.Steps {
		for _, content := range step.Content {
			if strings.TrimSpace(content.Text) != "" {
				rawJSON = strings.TrimSpace(content.Text)
				break
			}
		}
		if rawJSON != "" {
			break
		}
	}
	if rawJSON == "" && strings.TrimSpace(intResp.OutputText) != "" {
		rawJSON = strings.TrimSpace(intResp.OutputText)
	}

	if rawJSON == "" {
		return nil, "", nil, fmt.Errorf("no meeting minutes text returned by Gemini interactions API")
	}

	rawJSON = strings.TrimPrefix(rawJSON, "```json")
	rawJSON = strings.TrimPrefix(rawJSON, "```")
	rawJSON = strings.TrimSuffix(rawJSON, "```")
	rawJSON = strings.TrimSpace(rawJSON)

	// Clean stray markdown bullets or hyphens on their own line between JSON keys (e.g. "\n   -   \n")
	reStrayDash := regexp.MustCompile(`(?m)^\s*-\s*$`)
	rawJSON = reStrayDash.ReplaceAllString(rawJSON, "")

	// Clean literal unescaped tabs in string values that cause invalid character '\t' in string
	rawJSON = strings.ReplaceAll(rawJSON, "\t", "  ")

	var result ExtractionResult
	if err = json.Unmarshal([]byte(rawJSON), &result); err != nil || (len(result.MeetingMinutes.Attendees) == 0 && len(result.MeetingMinutes.Decisions) == 0 && len(result.MeetingMinutes.ActionItems) == 0 && len(result.MeetingMinutes.DiscussionPoints) == 0) {
		// Fallback 1: check if response directly contains MeetingMinutes root object
		var directMoM MeetingMinutes
		if directErr := json.Unmarshal([]byte(rawJSON), &directMoM); directErr == nil && (len(directMoM.Attendees) > 0 || len(directMoM.Decisions) > 0 || len(directMoM.ActionItems) > 0 || len(directMoM.DiscussionPoints) > 0) {
			result.MeetingMinutes = directMoM
			if result.NamedTranscript == "" {
				result.NamedTranscript = trimmedTranscript
			}
		} else {
			// Fallback 2: check if response is wrapped under any other key (e.g. meetingMinutes, data)
			var genericMap map[string]json.RawMessage
			if gErr := json.Unmarshal([]byte(rawJSON), &genericMap); gErr == nil {
				for k, v := range genericMap {
					if k == "named_transcript" || k == "namedTranscript" {
						var nt string
						_ = json.Unmarshal(v, &nt)
						if strings.TrimSpace(nt) != "" {
							result.NamedTranscript = nt
						}
						continue
					}
					var candidate MeetingMinutes
					if cErr := json.Unmarshal(v, &candidate); cErr == nil && (len(candidate.Attendees) > 0 || len(candidate.Decisions) > 0 || len(candidate.ActionItems) > 0 || len(candidate.DiscussionPoints) > 0) {
						result.MeetingMinutes = candidate
						break
					}
				}
			}
			if len(result.MeetingMinutes.Attendees) == 0 && len(result.MeetingMinutes.Decisions) == 0 && len(result.MeetingMinutes.ActionItems) == 0 && err != nil {
				return nil, "", nil, fmt.Errorf("failed to unmarshal meeting minutes JSON: %w (raw: %s)", err, rawJSON)
			}
		}
	}

	mom := &result.MeetingMinutes

	// Ensure slice fields are non-nil for JSON array serialization consistency
	if mom.Attendees == nil {
		mom.Attendees = []Attendee{}
	} else {
		mom.Attendees = filterValidAttendees(mom.Attendees, trimmedTranscript)
	}
	if mom.Agenda == nil {
		mom.Agenda = []string{}
	}
	if mom.DiscussionPoints == nil {
		mom.DiscussionPoints = []DiscussionPoint{}
	}
	if mom.Decisions == nil {
		mom.Decisions = []Decision{}
	}
	if mom.ActionItems == nil {
		mom.ActionItems = []ActionItem{}
	}
	if mom.RisksIssuesDependencies == nil {
		mom.RisksIssuesDependencies = []RiskIssueDependency{}
	}

	finalTranscript := trimmedTranscript

	// Calculate token usage and estimated costs in INR & USD
	usage := &TokenUsage{
		Model: s.model,
	}
	if intResp.UsageMetadata != nil {
		usage.InputTokens = intResp.UsageMetadata.PromptTokenCount
		if usage.InputTokens == 0 {
			usage.InputTokens = intResp.UsageMetadata.InputTokens
		}
		usage.OutputTokens = intResp.UsageMetadata.CandidatesTokenCount
		if usage.OutputTokens == 0 {
			usage.OutputTokens = intResp.UsageMetadata.OutputTokens
		}
		usage.TotalTokens = intResp.UsageMetadata.TotalTokenCount
		if usage.TotalTokens == 0 {
			usage.TotalTokens = usage.InputTokens + usage.OutputTokens
		}
	} else if intResp.Usage != nil {
		usage.InputTokens = intResp.Usage.PromptTokens
		usage.OutputTokens = intResp.Usage.CompletionTokens
		usage.TotalTokens = intResp.Usage.TotalTokens
	}

	// Heuristic fallback if API did not return token metrics (~4 characters per token)
	if usage.InputTokens <= 0 {
		usage.InputTokens = len(prompt) / 4
		if usage.InputTokens < 1 {
			usage.InputTokens = 1
		}
	}
	if usage.OutputTokens <= 0 {
		usage.OutputTokens = len(rawJSON) / 4
		if usage.OutputTokens < 1 {
			usage.OutputTokens = 1
		}
	}
	if usage.TotalTokens <= 0 {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	}

	// Gemini Pricing:
	// gemini-3.5-flash-lite: $0.30/1M input, $2.50/1M output
	// gemini-3.5-flash: $0.35/1M input, $2.50/1M output
	// gemini-3.6-flash: $0.75/1M input, $3.75/1M output
	// Conversion: 1 USD = 88.0 INR
	inputRateUSD := 0.30
	outputRateUSD := 2.50
	if strings.Contains(s.model, "3.6") {
		inputRateUSD = 0.75
		outputRateUSD = 3.75
	} else if !strings.Contains(s.model, "lite") && strings.Contains(s.model, "3.5") {
		inputRateUSD = 0.35
		outputRateUSD = 2.50
	}

	usage.EstimatedCostUSD = (float64(usage.InputTokens)*inputRateUSD + float64(usage.OutputTokens)*outputRateUSD) / 1000000.0
	usage.EstimatedCostINR = usage.EstimatedCostUSD * 88.0
	return mom, finalTranscript, usage, nil
}

// filterValidAttendees ensures only people who were actually present and participating
// in the conversation are included in attendees. People mentioned in passing, absent, or
// who sent apologies are strictly excluded.
func filterValidAttendees(attendees []Attendee, transcript string) []Attendee {
	if len(attendees) == 0 {
		return attendees
	}

	lowerTranscript := strings.ToLower(transcript)
	var valid []Attendee

	for _, att := range attendees {
		name := strings.TrimSpace(att.Name)
		if name == "" {
			continue
		}
		lowerName := strings.ToLower(name)

		// Exclude generic non-person roles or labels
		if lowerName == "narrator" || lowerName == "all" || lowerName == "all attendees" || lowerName == "all staff" || lowerName == "everyone" {
			continue
		}

		// Check if explicitly noted as absent, sick, or apologies without participating
		isAbsent := strings.Contains(lowerTranscript, lowerName+" is absent") ||
			strings.Contains(lowerTranscript, lowerName+" was absent") ||
			strings.Contains(lowerTranscript, lowerName+" is off sick") ||
			strings.Contains(lowerTranscript, lowerName+" was off sick") ||
			strings.Contains(lowerTranscript, lowerName+" has flu") ||
			strings.Contains(lowerTranscript, "apologies from "+lowerName) ||
			strings.Contains(lowerTranscript, "apologies for "+lowerName) ||
			strings.Contains(lowerTranscript, lowerName+" sent apologies") ||
			strings.Contains(lowerTranscript, lowerName+" sends apologies") ||
			strings.Contains(lowerTranscript, lowerName+" couldn't make it") ||
			strings.Contains(lowerTranscript, lowerName+" couldn't join") ||
			strings.Contains(lowerTranscript, lowerName+" is unavailable") ||
			strings.Contains(lowerTranscript, lowerName+" on leave")

		// If explicitly absent and never spoke in dialogue, exclude
		if isAbsent && !strings.Contains(transcript, name+":") {
			log.Printf("filterValidAttendees: excluding %s (marked absent and did not speak in conversation)", name)
			continue
		}

		// If the person is not in the transcript at all, exclude
		if !strings.Contains(lowerTranscript, lowerName) {
			log.Printf("filterValidAttendees: excluding %s (name never appears in transcript)", name)
			continue
		}

		valid = append(valid, att)
	}

	return valid
}

// SaveMeetingMinutes persists extracted Minutes of Meeting into the meeting_minutes table with the given conversation_id.
// If a record already exists for the conversation, it replaces it (upsert).
func (s *GeminiMeetingMinutesService) SaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, mom *MeetingMinutes, usage *TokenUsage) (*MeetingMinutesRecord, error) {
	if mom == nil {
		return nil, fmt.Errorf("meeting minutes cannot be nil")
	}

	dataJSON, err := json.Marshal(mom)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal meeting minutes for database: %w", err)
	}

	recordID := uuid.New()
	now := time.Now()

	modelName := s.model
	inputTokens := 0
	outputTokens := 0
	totalTokens := 0
	costINR := 0.0
	costUSD := 0.0

	if usage != nil {
		if usage.Model != "" {
			modelName = usage.Model
		}
		inputTokens = usage.InputTokens
		outputTokens = usage.OutputTokens
		totalTokens = usage.TotalTokens
		costINR = usage.EstimatedCostINR
		costUSD = usage.EstimatedCostUSD
	}

	query := `INSERT INTO meeting_minutes (
		id, conversation_id, data, model, input_tokens, output_tokens, total_tokens, estimated_cost_inr, estimated_cost_usd, created_at
	) VALUES (
		$1, $2, $3, $4, $5, $6, $7, $8, $9, $10
	)
	ON CONFLICT (conversation_id) 
	DO UPDATE SET 
		data = EXCLUDED.data,
		model = EXCLUDED.model,
		input_tokens = EXCLUDED.input_tokens,
		output_tokens = EXCLUDED.output_tokens,
		total_tokens = EXCLUDED.total_tokens,
		estimated_cost_inr = EXCLUDED.estimated_cost_inr,
		estimated_cost_usd = EXCLUDED.estimated_cost_usd,
		created_at = EXCLUDED.created_at
	RETURNING id, created_at`

	err = s.db.QueryRow(ctx, query, recordID, conversationID, dataJSON, modelName, inputTokens, outputTokens, totalTokens, costINR, costUSD, now).Scan(&recordID, &now)
	if err != nil {
		return nil, fmt.Errorf("failed to insert/update meeting minutes in database: %w", err)
	}

	return &MeetingMinutesRecord{
		ID:               recordID,
		ConversationID:   conversationID,
		Data:             *mom,
		Model:            modelName,
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		TotalTokens:      totalTokens,
		EstimatedCostINR: costINR,
		EstimatedCostUSD: costUSD,
		CreatedAt:        now,
	}, nil
}

// ExtractAndSaveMeetingMinutes coordinates MoM extraction, transcript name resolution, and database insertion.
func (s *GeminiMeetingMinutesService) ExtractAndSaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, transcript string) (*MeetingMinutes, string, *TokenUsage, error) {
	mom, namedTranscript, usage, err := s.ExtractMeetingMinutes(ctx, transcript)
	if err != nil {
		return nil, "", nil, fmt.Errorf("extract meeting minutes: %w", err)
	}

	_, err = s.SaveMeetingMinutes(ctx, conversationID, mom, usage)
	if err != nil {
		return nil, "", nil, fmt.Errorf("save meeting minutes: %w", err)
	}

	return mom, namedTranscript, usage, nil
}
