package minutes

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

type MeetingMinutesRecord struct {
	ID             uuid.UUID      `json:"id"`
	ConversationID uuid.UUID      `json:"conversation_id"`
	Data           MeetingMinutes `json:"data"`
	CreatedAt      time.Time      `json:"created_at"`
}

// ExtractionResult holds both the resolved named transcript and structured MoM.
type ExtractionResult struct {
	NamedTranscript string         `json:"named_transcript"`
	MeetingMinutes  MeetingMinutes `json:"meeting_minutes"`
}

// MeetingMinutesService defines the contract for MoM extraction and persistence.
type MeetingMinutesService interface {
	ExtractMeetingMinutes(ctx context.Context, transcript string) (*MeetingMinutes, string, error)
	SaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, mom *MeetingMinutes) (*MeetingMinutesRecord, error)
	ExtractAndSaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, transcript string) (*MeetingMinutes, string, error)
}

// GeminiService implements MeetingMinutesService using Google's Gemini Interactions API.
type GeminiService struct {
	apiKey     string
	model      string
	db         *pgxpool.Pool
	httpClient *http.Client
}

// NewGeminiService creates a new GeminiService with the given PostgreSQL connection pool.
// Reads GEMINI_API_KEY from environment, and optionally GEMINI_TOPIC_MODEL (defaults to gemini-3.5-flash-lite).
func NewGeminiService(db *pgxpool.Pool) (*GeminiService, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := os.Getenv("GEMINI_TOPIC_MODEL")
	if model == "" {
		model = "gemini-3.5-flash-lite"
	}

	return &GeminiService{
		apiKey: key,
		model:  model,
		db:     db,
		httpClient: &http.Client{
			Timeout: 2 * time.Minute,
		},
	}, nil
}

// Model returns the Gemini model currently configured for MoM extraction.
func (s *GeminiService) Model() string {
	return s.model
}

// ─────────────────────────────────────────────
// Gemini Interactions API request & response types
// ─────────────────────────────────────────────

type interactionsRequest struct {
	Model          string          `json:"model"`
	Input          string          `json:"input"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type responseFormat struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type"`
}

type interactionsResponse struct {
	ID         string            `json:"id"`
	Status     string            `json:"status"`
	OutputText string            `json:"output_text,omitempty"`
	Steps      []interactionStep `json:"steps"`
	Error      *struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type interactionStep struct {
	Type    string               `json:"type"`
	Content []interactionContent `json:"content"`
}

type interactionContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ─────────────────────────────────────────────
// Extraction & Persistence Logic
// ─────────────────────────────────────────────

// ExtractMeetingMinutes sends the transcript to Gemini, resolves speaker names in both transcript & MoM, and parses the structured result.
func (s *GeminiService) ExtractMeetingMinutes(ctx context.Context, transcript string) (*MeetingMinutes, string, error) {
	trimmedTranscript := strings.TrimSpace(transcript)
	if trimmedTranscript == "" {
		return nil, "", fmt.Errorf("transcript cannot be empty")
	}

	prompt := fmt.Sprintf(`You are extracting Minutes of Meeting (MoM) from a transcribed conversation. The transcript is the ONLY source of truth. Use no outside knowledge, no assumptions about job roles, and no business logic.

STEP 1 - EVIDENCE FIRST
For every item you extract (attendee, decision, action, cause, risk, number, next-meeting detail), ensure there is supporting dialogue in the transcript. If the transcript does not support the item, do not include it.

STEP 2 - CLASSIFY EACH ITEM
- EXPLICIT: directly stated. Include.
- CLEARLY_ESTABLISHED: not word-for-word, but the conversation unambiguously establishes it. Include, using wording close to the transcript.
- INFERRED: plausible from context/roles/logic. EXCLUDE from the MoM.
- UNKNOWN: transcript is silent. Use null, [] or empty array.
Test for CLEARLY_ESTABLISHED: "Would two careful readers of this transcript agree without needing outside assumptions?" If not, it is INFERRED and must be omitted.

STEP 3 - FIELD RULES
1. MEETING METADATA:
   - "title": Use the exact meeting title from the dialogue (e.g. "Monthly Meeting", not "Monthly Team Meeting" unless explicitly stated).
   - "meeting_type": "Monthly Meeting" or null.
   - "date", "time", "location": null unless explicitly stated in dialogue.

2. ATTENDEES:
   - Include ONLY individuals who were present and spoke or were confirmed in attendance.
   - Anyone who arrived late: Include and note "arrived late" in their designation/notes (e.g. "Frank Lyons").
   - Unnamed attendees confirmed present (e.g. minute taker): Record their role only ("Minute taker", designation "specialist minute taker").
   - Anyone absent, off sick, or who sent apologies (e.g. Gary Cope, Carl Madden, Joy Ballenwiese, Janet Bowers, Clive): STRICTLY EXCLUDE from Attendees. Document their apologies/absence in Discussion Points under "Apologies for Absence & Introductions".
   - Include accurate job titles / designations when introduced or stated in the transcript.

3. AGENDA & DISCUSSION POINTS:
   - Group discussion points by topic using formal titles (e.g. "Apologies for Absence & Introductions", "Matters Arising", "Car Parking Issues", "Staff Morale & Feedback", "IT Issues & Infrastructure", "Financial Report", "Any Other Business").
   - Matters arising: Explicitly record "None" if participants confirmed there were no matters arising from the previous meeting.
   - NO UNVERIFIED CAUSAL LINKS: Never link two events causally unless explicitly stated. Report events separately (e.g. "A power cut occurred" and "Accounts agreed to pay electricity bills on time", DO NOT state as fact that the power cut was caused by an unpaid bill).
   - NO JARGON AS FACT: Drop transcription errors or technical jargon (e.g. logative rebix, sub-feeders); state the plain conclusion reached (power cut).
   - NO SPECULATION: Do not state that Sue Carpenter needs a car or that staff cycle to shower. Record only what was directly said.
   - NUMBERS & FINANCE: Copy figures exactly. If units/meaning are unclear, add "(meaning/units not stated)" (e.g. figures 52, 21, and 11). Note that the company is "down on last year" but remains in the black.

4. DECISIONS:
   - Decisions require explicit consensus or commitment (e.g. "Agreed", "That settles that", chair's call with no objection). Brainstormed ideas are NOT decisions.
   - Keep decisions distinct:
     * 5 company parking spaces in total: 3 for visitors/clients, 2 allocated to Sue Carpenter and Jason Somerville (with Jason parking by the garages).
     * Anyone parking inconsiderately will forfeit their parking privileges.
     * Accounts to pay electricity bills on time.
     * Staff to submit 4 team-building activity ideas within two days, from which Rita will pick options for a staff vote.

5. ACTION ITEMS & OWNERS:
   - Exhaustively scan for every "I'll / can you / we need to / let's" commitment. Do not merge separate action items.
   - Start each action item with a clear active verb.
   - OWNERS: Assign ONLY if someone explicitly volunteered ("I'll send it") or was explicitly assigned.
     * Julian Geddis volunteered to send the priority parking list to all staff ("I'll send it").
     * Jason Somerville volunteered to get a list, circulate it, and put up signs on company parking spaces.
     * Julian Geddis to confirm the number of sales staff who use their cars during the day.
     * Rita to coordinate/allocate priority spaces when an allocated staff member is absent/off sick.
     * All Attendees: Email 4 team-building/morale activity ideas to Rita within two days.
     * Rita: Select options from the submitted team-building ideas for a company-wide vote.
     * Rita: Speak with Clive to set a date for a separate meeting with Clive, Rita, and Frank regarding software training. (Due date: null; Lucy's 2-week suggestion was not confirmed).
     * Accounts Department: Pay electricity bills on time (per Rita's "Can we minute accounts?").
     * Schedule general cleanliness issues (kitchen plates and shower room) for the next meeting. (Owner: null, because the addressee was not specified).
   - DUE DATES: Only explicit deadlines in original phrasing (e.g. "within two days"). Otherwise null. Never convert to calendar date.

6. RISKS & DEPENDENCIES:
   - Only include risks, issues, or dependencies that participants actually raised.
   - CRITICAL: "owner" and "required_action" MUST be null unless explicitly stated or assigned in the transcript! DO NOT invent risk owners (e.g. no "Management / Department Heads") or invented mitigations (e.g. no "establish vendor protocols").

7. NEXT MEETING:
   - AGENDA: ONLY include items explicitly scheduled for the next general meeting (e.g. cleanliness of kitchen plates and shower room). Separate meetings mentioned in discussion (e.g. follow-up training with Clive) are follow-up meetings, NOT the next general meeting.
   - "date", "time": null unless confirmed.

Return JSON ONLY using this exact structure:
{
  "named_transcript": "...",
  "meeting_minutes": {
    "meeting": {
      "title": "Monthly Meeting",
      "date": null,
      "time": null,
      "location": null,
      "meeting_type": "Monthly Meeting"
    },
    "attendees": [
      {
        "name": "Full Name",
        "designation": "Job Title or null"
      }
    ],
    "chairperson": "Name or null",
    "minutes_prepared_by": null,
    "agenda": ["Topic 1", "Topic 2"],
    "discussion_points": [
      {
        "topic": "Topic Name",
        "discussion": "Factual, neutral, evidence-grounded summary..."
      }
    ],
    "decisions": [
      {
        "decision": "Decision Title",
        "remarks": "Detailed remarks on what was agreed..."
      }
    ],
    "action_items": [
      {
        "action_item": "Action description starting with an active verb...",
        "owner": "Person / All Attendees / Accounts Department or null",
        "priority": "High / Medium / Low or null",
        "due_date": "Original timeframe or null",
        "status": "Pending"
      }
    ],
    "risks_issues_dependencies": [
      {
        "issue": "Risk description as raised in transcript...",
        "owner": null,
        "required_action": null
      }
    ],
    "next_meeting": {
      "date": null,
      "time": null,
      "agenda": "General cleanliness problems (unwashed plates in the kitchen and state of the shower room)"
    }
  }
}

Transcript:
%s`, trimmedTranscript)

	reqBody := interactionsRequest{
		Model: s.model,
		Input: prompt,
		ResponseFormat: &responseFormat{
			Type:     "text",
			MIMEType: "application/json",
		},
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal request: %w", err)
	}

	const apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"

	// Retry loop with exponential backoff for transient 503 / 429 errors.
	var respBody []byte
	maxRetries := 3
	backoff := 1 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBytes))
		if err != nil {
			return nil, "", fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("x-goog-api-key", s.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := s.httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, "", fmt.Errorf("gemini interactions request canceled: %w", ctx.Err())
			}
			if attempt == maxRetries {
				return nil, "", fmt.Errorf("gemini interactions request failed after %d retries: %w", maxRetries, err)
			}
			log.Printf("Gemini interactions request error (attempt %d/%d): %v, retrying in %v...", attempt+1, maxRetries, err, backoff)
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, "", fmt.Errorf("failed to read response body: %w", err)
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
				return nil, "", fmt.Errorf("gemini API error (%v): %s", errResp.Error.Code, errResp.Error.Message)
			}
			return nil, "", fmt.Errorf("gemini API returned status %d: %s", resp.StatusCode, string(body))
		}

		respBody = body
		break
	}

	var intResp interactionsResponse
	if err = json.Unmarshal(respBody, &intResp); err != nil {
		return nil, "", fmt.Errorf("failed to parse Gemini interactions response: %w", err)
	}

	if intResp.Error != nil {
		return nil, "", fmt.Errorf("gemini API error (%v): %s", intResp.Error.Code, intResp.Error.Message)
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
		return nil, "", fmt.Errorf("no meeting minutes text returned by Gemini interactions API")
	}

	rawJSON = strings.TrimPrefix(rawJSON, "```json")
	rawJSON = strings.TrimPrefix(rawJSON, "```")
	rawJSON = strings.TrimSuffix(rawJSON, "```")
	rawJSON = strings.TrimSpace(rawJSON)

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
				return nil, "", fmt.Errorf("failed to unmarshal meeting minutes JSON: %w (raw: %s)", err, rawJSON)
			}
		}
	}

	mom := &result.MeetingMinutes

	// Ensure slice fields are non-nil for JSON array serialization consistency
	if mom.Attendees == nil {
		mom.Attendees = []Attendee{}
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

	finalTranscript := strings.TrimSpace(result.NamedTranscript)
	if finalTranscript == "" {
		finalTranscript = trimmedTranscript
	}

	return mom, finalTranscript, nil
}

// SaveMeetingMinutes persists extracted Minutes of Meeting into the meeting_minutes table with the given conversation_id.
// If a record already exists for the conversation, it replaces it (upsert).
func (s *GeminiService) SaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, mom *MeetingMinutes) (*MeetingMinutesRecord, error) {
	if mom == nil {
		return nil, fmt.Errorf("meeting minutes cannot be nil")
	}

	dataJSON, err := json.Marshal(mom)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal meeting minutes for database: %w", err)
	}

	recordID := uuid.New()
	now := time.Now()

	query := `INSERT INTO meeting_minutes (id, conversation_id, data, created_at)
	          VALUES ($1, $2, $3, $4)
	          ON CONFLICT (conversation_id) 
	          DO UPDATE SET data = EXCLUDED.data, created_at = EXCLUDED.created_at
	          RETURNING id, created_at`

	err = s.db.QueryRow(ctx, query, recordID, conversationID, dataJSON, now).Scan(&recordID, &now)
	if err != nil {
		return nil, fmt.Errorf("failed to insert/update meeting minutes in database: %w", err)
	}

	return &MeetingMinutesRecord{
		ID:             recordID,
		ConversationID: conversationID,
		Data:           *mom,
		CreatedAt:      now,
	}, nil
}

// ExtractAndSaveMeetingMinutes coordinates MoM extraction, transcript name resolution, and database insertion.
func (s *GeminiService) ExtractAndSaveMeetingMinutes(ctx context.Context, conversationID uuid.UUID, transcript string) (*MeetingMinutes, string, error) {
	mom, namedTranscript, err := s.ExtractMeetingMinutes(ctx, transcript)
	if err != nil {
		return nil, "", fmt.Errorf("extract meeting minutes: %w", err)
	}

	_, err = s.SaveMeetingMinutes(ctx, conversationID, mom)
	if err != nil {
		return nil, "", fmt.Errorf("save meeting minutes: %w", err)
	}

	return mom, namedTranscript, nil
}
