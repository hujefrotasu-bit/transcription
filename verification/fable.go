package verification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// FableClient interacts with the CodeCraft API for claude-fable models.
type FableClient struct {
	APIKey     string
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// NewFableClient initializes a FableClient with sanitized URL and model ID.
func NewFableClient(apiKey, baseURL, model string) *FableClient {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = "https://codecraftapi.com/v1"
	}
	baseURL = strings.TrimRight(baseURL, "/")

	model = strings.TrimSpace(model)
	if model == "" || strings.EqualFold(model, "claude-fable-5-1") {
		// CodeCraft requires the exact dot-formatted identifier
		model = "claude-fable-5.1"
	}

	return &FableClient{
		APIKey:  strings.TrimSpace(apiKey),
		BaseURL: baseURL,
		Model:   model,
		HTTPClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type chatRequest struct {
	Model          string          `json:"model"`
	Temperature    float64         `json:"temperature"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Messages       []chatMessage   `json:"messages"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

// CompleteChat sends a chat completion request to the Fable / CodeCraft endpoint.
func (c *FableClient) CompleteChat(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	if c.APIKey == "" {
		return "", fmt.Errorf("FABLE_API_KEY is not configured")
	}

	endpoint := c.BaseURL + "/chat/completions"

	var messages []chatMessage
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, chatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, chatMessage{
		Role:    "user",
		Content: userPrompt,
	})

	reqBody := chatRequest{
		Model:          c.Model,
		Temperature:    0.0,
		ResponseFormat: &responseFormat{Type: "json_object"},
		Messages:       messages,
	}

	rawJSON, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawJSON))
	if err != nil {
		return "", fmt.Errorf("failed to create http request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	maxRetries := 2
	backoff := 1 * time.Second
	var respBody []byte

	for attempt := 0; attempt <= maxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawJSON))
		if err != nil {
			return "", fmt.Errorf("failed to create http request: %w", err)
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("fable request canceled: %w", ctx.Err())
			}
			if attempt == maxRetries {
				return "", fmt.Errorf("fable request failed after %d retries: %w", maxRetries, err)
			}
			log.Printf("Fable request error (attempt %d/%d): %v, retrying in %v...", attempt+1, maxRetries, err, backoff)
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		bodyBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("failed to read fable response body: %w", err)
		}

		// Retry on transient gateway / upstream errors (502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout, 429 Rate Limit)
		if resp.StatusCode == http.StatusBadGateway ||
			resp.StatusCode == http.StatusServiceUnavailable ||
			resp.StatusCode == http.StatusGatewayTimeout ||
			resp.StatusCode == http.StatusTooManyRequests {
			if attempt < maxRetries {
				log.Printf("Fable API status %d (attempt %d/%d), retrying in %v...", resp.StatusCode, attempt+1, maxRetries, backoff)
				time.Sleep(backoff)
				backoff *= 2
				continue
			}
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return "", fmt.Errorf("fable API status %d: %s", resp.StatusCode, string(bodyBytes))
		}

		respBody = bodyBytes
		break
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", fmt.Errorf("failed to decode fable response JSON: %w (raw: %s)", err, string(respBody))
	}

	if chatResp.Error != nil && chatResp.Error.Message != "" {
		return "", fmt.Errorf("fable API error: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 || chatResp.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("fable returned empty completion choices")
	}

	return chatResp.Choices[0].Message.Content, nil
}
