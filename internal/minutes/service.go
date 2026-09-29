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

	prompt := fmt.Sprintf(`You are an expert transcript and meeting minutes analyst.

Analyze the complete conversation transcript where speakers may currently be labeled as "Speaker 1", "Speaker 2", etc.

TASKS:
1. Identify the real names of speakers if their name is introduced, spoken, or confirmed in the dialogue (for example, if someone is greeted with "Hello is this Sarah?" and replies "Yes it is Sarah", then that speaker is "Sarah").
2. If a speaker's real name is never mentioned or remains unknown, keep their label as "Speaker 1", "Speaker 2", etc.
3. Replace speaker labels throughout the transcript with their identified names (e.g. replace "Speaker 2:" with "Sarah:"). Keep unknown speakers as "Speaker 1:".
4. Generate structured Minutes of Meeting (MoM).
   - In the "attendees" list, DO NOT list both a speaker number and their real name when they are the same person.
   - Use their real name if known (e.g. "Sarah"), or their speaker label if unknown (e.g. "Speaker 1").
   - Extract meaningful discussion points with topic and concise factual summary.
   - Extract explicit decisions.
   - Extract clear action items with owner, due date, and status.
   - For any unstated metadata (location, time, priority, chairperson), strictly use null.

IMPORTANT DATA RULES:
1. NEVER invent information.
2. If the transcript does not provide a meeting title, date, time, location, meeting type, designation, chairperson, or minutes preparer, return null.
3. Do not infer an exact date from the current date.
4. Do not invent attendees.
5. Extract explicit decisions from the conversation. Only include decisions that were actually made or clearly agreed upon.
6. Extract action items when the conversation contains a clear task/request/commitment, owner, due date, priority and status when supported by the transcript.
   - Do not confuse a general statement with an action item.
   - Only assign an owner when the transcript provides enough evidence.
   - Only assign a due date when the transcript explicitly provides one.
   - If priority is not discussed, return null.
   - If status is not discussed, use "Pending" only when an action item clearly exists and has not been completed. Otherwise use null.
7. Do not create risks/issues/dependencies unless they are actually discussed.
8. Do not create a next meeting unless the transcript discusses one.
9. Return JSON ONLY using the exact requested structure.

Return JSON ONLY using this exact structure:
{
  "named_transcript": "Speaker 1: Hello, is this Sarah?\nSarah: Hello, yes it is Sarah.\n...",
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
        "name": "Sarah",
        "designation": null
      },
      {
        "name": "Speaker 1",
        "designation": null
      }
    ],
    "chairperson": null,
    "minutes_prepared_by": null,
    "agenda": [],
    "discussion_points": [
      {
        "topic": "...",
        "discussion": "..."
      }
    ],
    "decisions": [
      {
        "decision": "...",
        "remarks": "..."
      }
    ],
    "action_items": [
      {
        "action_item": "...",
        "owner": null,
        "priority": null,
        "due_date": null,
        "status": "Pending"
      }
    ],
    "risks_issues_dependencies": [],
    "next_meeting": {
      "date": null,
      "time": null,
      "agenda": null
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
	if err = json.Unmarshal([]byte(rawJSON), &result); err != nil {
		// Fallback check if response directly contains MeetingMinutes root object
		var directMoM MeetingMinutes
		if directErr := json.Unmarshal([]byte(rawJSON), &directMoM); directErr == nil {
			result.MeetingMinutes = directMoM
			result.NamedTranscript = trimmedTranscript
		} else {
			return nil, "", fmt.Errorf("failed to unmarshal meeting minutes JSON: %w (raw: %s)", err, rawJSON)
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
