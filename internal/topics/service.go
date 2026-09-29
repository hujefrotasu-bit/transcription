package topics

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

// Topic represents an extracted topic record stored in PostgreSQL.
type Topic struct {
	ID             uuid.UUID `json:"id"`
	ConversationID uuid.UUID `json:"conversation_id"`
	Title          string    `json:"title"`
	Summary        string    `json:"summary"`
	CreatedAt      time.Time `json:"created_at"`
}

// ExtractedTopic represents the structured topic returned by the LLM.
type ExtractedTopic struct {
	Title   string `json:"title"`
	Summary string `json:"summary"`
}

// TopicService defines the contract for topic extraction and persistence.
type TopicService interface {
	ExtractTopics(ctx context.Context, transcript string) ([]ExtractedTopic, error)
	SaveTopics(ctx context.Context, conversationID uuid.UUID, extracted []ExtractedTopic) ([]Topic, error)
	ExtractAndSave(ctx context.Context, conversationID uuid.UUID, transcript string) ([]Topic, error)
}

// GeminiService implements TopicService using Google's Gemini generateContent API.
type GeminiService struct {
	apiKey     string
	model      string
	db         *pgxpool.Pool
	httpClient *http.Client
}

// NewGeminiService creates a new GeminiService with the given PostgreSQL connection pool.
// Reads GEMINI_API_KEY from environment, and optionally GEMINI_TOPIC_MODEL (defaults to gemini-3.8-flash).
func NewGeminiService(db *pgxpool.Pool) (*GeminiService, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := os.Getenv("GEMINI_TOPIC_MODEL")
	if model == "" {
		model = "gemini-3.8-flash"
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

// Model returns the Gemini model currently configured for topic extraction.
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

type topicsPayload struct {
	Topics []ExtractedTopic `json:"topics"`
}

// ─────────────────────────────────────────────
// Topic Extraction & Persistence
// ─────────────────────────────────────────────

// ExtractTopics sends the transcript to Gemini via the Interactions API and parses the structured topics.
func (s *GeminiService) ExtractTopics(ctx context.Context, transcript string) ([]ExtractedTopic, error) {
	trimmedTranscript := strings.TrimSpace(transcript)
	if trimmedTranscript == "" {
		return nil, fmt.Errorf("transcript cannot be empty")
	}

	prompt := fmt.Sprintf(`Analyze the complete conversation transcript.
Identify the important/main topics discussed.
Do not summarize every sentence.
Return concise useful topics.
Each topic must contain a title and a summary.

Return JSON ONLY matching this exact structure:
{
  "topics": [
    {
      "title": "Topic Title",
      "summary": "Brief summary of what was discussed about this topic."
    }
  ]
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
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	const maxAttempts = 3
	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		topics, err := s.callInteractions(ctx, reqBytes)
		if err == nil {
			return topics, nil
		}
		lastErr = err

		if !isRetryable(err) || attempt == maxAttempts {
			break
		}

		backoff := time.Duration(attempt) * 1500 * time.Millisecond
		log.Printf("ExtractTopics: Gemini transient error on attempt %d/%d (%v); retrying in %v...", attempt, maxAttempts, err, backoff)

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
	}

	return nil, lastErr
}

func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "service_unavailable") ||
		strings.Contains(msg, "high demand") ||
		strings.Contains(msg, "spikes in demand") ||
		strings.Contains(msg, "status 503") ||
		strings.Contains(msg, "status 429") ||
		strings.Contains(msg, "status 502") ||
		strings.Contains(msg, "status 504") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "resource_exhausted") ||
		strings.Contains(msg, "temporarily")
}

func (s *GeminiService) callInteractions(ctx context.Context, reqBytes []byte) ([]ExtractedTopic, error) {
	const apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("x-goog-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini interactions request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error *struct {
				Code    any    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(respBody, &errResp) == nil && errResp.Error != nil {
			return nil, fmt.Errorf("gemini API error (%v): %s", errResp.Error.Code, errResp.Error.Message)
		}
		return nil, fmt.Errorf("gemini API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var intResp interactionsResponse
	if err = json.Unmarshal(respBody, &intResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini interactions response: %w", err)
	}

	if intResp.Error != nil {
		return nil, fmt.Errorf("gemini API error (%v): %s", intResp.Error.Code, intResp.Error.Message)
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
		return nil, fmt.Errorf("no topic text returned by Gemini interactions API")
	}

	rawJSON = strings.TrimPrefix(rawJSON, "```json")
	rawJSON = strings.TrimPrefix(rawJSON, "```")
	rawJSON = strings.TrimSuffix(rawJSON, "```")
	rawJSON = strings.TrimSpace(rawJSON)

	var payload topicsPayload
	if err = json.Unmarshal([]byte(rawJSON), &payload); err != nil {
		return nil, fmt.Errorf("failed to unmarshal topics JSON: %w (raw: %s)", err, rawJSON)
	}

	// Validate topics
	var validTopics []ExtractedTopic
	for _, t := range payload.Topics {
		title := strings.TrimSpace(t.Title)
		summary := strings.TrimSpace(t.Summary)
		if title != "" && summary != "" {
			validTopics = append(validTopics, ExtractedTopic{
				Title:   title,
				Summary: summary,
			})
		}
	}

	if len(validTopics) == 0 {
		return nil, fmt.Errorf("LLM returned no valid topics with non-empty title and summary")
	}

	return validTopics, nil
}

// SaveTopics persists extracted topics into the topics table with the given conversation_id.
func (s *GeminiService) SaveTopics(ctx context.Context, conversationID uuid.UUID, extracted []ExtractedTopic) ([]Topic, error) {
	if len(extracted) == 0 {
		return nil, nil
	}

	var saved []Topic
	now := time.Now()

	for _, t := range extracted {
		topic := Topic{
			ID:             uuid.New(),
			ConversationID: conversationID,
			Title:          t.Title,
			Summary:        t.Summary,
			CreatedAt:      now,
		}

		query := `INSERT INTO topics (id, conversation_id, title, summary, created_at)
		          VALUES ($1, $2, $3, $4, $5)`

		_, err := s.db.Exec(ctx, query, topic.ID, topic.ConversationID, topic.Title, topic.Summary, topic.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to insert topic %q into database: %w", topic.Title, err)
		}

		saved = append(saved, topic)
	}

	return saved, nil
}

// ExtractAndSave coordinates topic extraction and database insertion for a conversation.
func (s *GeminiService) ExtractAndSave(ctx context.Context, conversationID uuid.UUID, transcript string) ([]Topic, error) {
	extracted, err := s.ExtractTopics(ctx, transcript)
	if err != nil {
		return nil, fmt.Errorf("extract topics: %w", err)
	}

	saved, err := s.SaveTopics(ctx, conversationID, extracted)
	if err != nil {
		return nil, fmt.Errorf("save topics: %w", err)
	}

	return saved, nil
}
