package transcription

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TranscriptionService is the interface any speech-to-text provider must satisfy.
type TranscriptionService interface {
	Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error)
	ResolveSpeakers(ctx context.Context, rawTranscript string) string
	Model() string
}

// ─────────────────────────────────────────────
// Gemini implementation
// ─────────────────────────────────────────────

// GeminiService transcribes audio using the Google Gemini Interactions API.
// It uploads audio via the File API, then calls the /v1beta/interactions endpoint
// with speaker diarization configured via transcription_config.
type GeminiTranscriptionService struct {
	apiKey     string
	model      string
	nameModel  string
	httpClient *http.Client
}

// NewGeminiService reads GEMINI_API_KEY from the environment and constructs
// a GeminiService. GEMINI_TRANSCRIPTION_MODEL overrides the default transcription model,
// and GEMINI_TOPIC_MODEL overrides the default name resolution model.
func NewGeminiTranscriptionService() (*GeminiTranscriptionService, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY is not set")
	}

	model := os.Getenv("GEMINI_TRANSCRIPTION_MODEL")
	if model == "" {
		model = "gemini-3.5-transcribe"
	}

	nameModel := os.Getenv("GEMINI_SPEAKER_MODEL")
	if nameModel == "" {
		nameModel = os.Getenv("GEMINI_TOPIC_MODEL")
	}
	if nameModel == "" {
		nameModel = "gemini-3.5-flash-lite"
	}

	return &GeminiTranscriptionService{
		apiKey:    key,
		model:     model,
		nameModel: nameModel,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute, // audio upload + transcription can take time
		},
	}, nil
}

// Model returns the Gemini transcription model name in use.
func (s *GeminiTranscriptionService) Model() string {
	return s.model
}

// ─────────────────────────────────────────────
// File API types
// ─────────────────────────────────────────────

type uploadFileResponse struct {
	File struct {
		URI      string `json:"uri"`
		MIMEType string `json:"mimeType"`
	} `json:"file"`
}

// ─────────────────────────────────────────────
// Interactions API request types
// ─────────────────────────────────────────────

// audioInteractionsRequest is the body sent to POST /v1beta/interactions.
// This is the correct endpoint for gemini-3.5-transcribe (not generateContent).
type audioInteractionsRequest struct {
	Model            string           `json:"model"`
	Input            []inputItem      `json:"input"`
	GenerationConfig generationConfig `json:"generation_config"`
}

type inputItem struct {
	Type     string `json:"type"`
	URI      string `json:"uri"`
	MIMEType string `json:"mime_type"`
}

type generationConfig struct {
	TranscriptionConfig transcriptionConfig `json:"transcription_config"`
}

type transcriptionConfig struct {
	Mode transcriptionMode `json:"mode"`
}

// transcriptionMode sets verbatim mode with speaker diarization enabled.
// Per the official Gemini 3.5 Transcribe docs:
//   - type must be "verbatim" when using diarization_mode or timestamp_granularities
//   - diarization_mode "speaker" enables multi-speaker identification
type transcriptionMode struct {
	Type                   string   `json:"type"`
	DiarizationMode        string   `json:"diarization_mode"`
	TimestampGranularities []string `json:"timestamp_granularities,omitempty"`
}

// audioInteractionsResponse is the raw REST JSON returned by POST /v1beta/interactions.
//
// Unlike generateContent, the Interactions API wraps everything inside an
// "interaction" object. The transcript text lives at:
//
//	interaction.steps[N].content.parts[M].text
//
// where role == "model". The SDK convenience property output_text walks
// this path automatically; here we do it manually.
type audioInteractionsResponse struct {
	Steps []audioInteractionStep `json:"steps"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type audioInteractionStep struct {
	Content []audioInteractionContent `json:"content"`
	Type    string                    `json:"type"`
	Speaker string                    `json:"speaker,omitempty"`
}

type audioInteractionContent struct {
	Text         string                       `json:"text"`
	Type         string                       `json:"type"`
	Speaker      string                       `json:"speaker,omitempty"`
	SpeakerLabel string                       `json:"speaker_label,omitempty"`
	Annotations  []audioInteractionAnnotation `json:"annotations,omitempty"`
}

type audioInteractionAnnotation struct {
	Type         string `json:"type"`
	Text         string `json:"text"`
	Speaker      string `json:"speaker,omitempty"`
	SpeakerLabel string `json:"speaker_label,omitempty"`
	StartOffset  string `json:"start_offset,omitempty"`
	EndOffset    string `json:"end_offset,omitempty"`
}

// ─────────────────────────────────────────────
// Transcribe (public interface implementation)
// ─────────────────────────────────────────────

// Transcribe uploads the audio to the Gemini File API, transcribes with
// speaker diarization enabled, and resolves speaker labels to real names
// whenever names are mentioned or introduced in the conversation.
func (s *GeminiTranscriptionService) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	mimeType := mimeTypeForFilename(filename)

	// Step 1: upload the audio file to get a persistent URI.
	fileURI, err := s.uploadFile(ctx, audio, filename, mimeType)
	if err != nil {
		return "", fmt.Errorf("file upload failed: %w", err)
	}

	// Step 2: transcribe with official speaker diarization config.
	transcript, err := s.transcribeWithDiarization(ctx, fileURI, mimeType)
	if err != nil {
		return "", fmt.Errorf("transcription failed: %w", err)
	}

	// Step 3: resolve speaker names from introductions and dialogue.
	// If names are mentioned or speakers are introduced, replace generic "Speaker X:" labels with their actual names.
	// If names are not mentioned or remain unknown, keep "Speaker X:".
	namedTranscript := s.ResolveSpeakers(ctx, transcript)
	if strings.TrimSpace(namedTranscript) != "" {
		return strings.TrimSpace(namedTranscript), nil
	}

	return transcript, nil
}

// ResolveSpeakers uses Gemini to analyze meeting dialogue discourse, introductions,
// direct address, questions, answers, and role context to assign accurate speaker names
// to every dialogue turn. Handles unlabelled raw text, generic "Speaker 1:" tags, and mixed transcripts.
// If a person's name cannot be confirmed, it assigns consistent speaker identifiers (e.g. "Speaker 1:").
func (s *GeminiTranscriptionService) ResolveSpeakers(ctx context.Context, rawTranscript string) string {
	trimmed := strings.TrimSpace(rawTranscript)
	if trimmed == "" {
		return rawTranscript
	}

	prompt := fmt.Sprintf(`You are an expert conversational discourse and speaker-attribution analyzer for meeting transcripts.

You are given a meeting transcript. The input might:
A) Have NO speaker tags at all (just lines or paragraphs of dialogue spoken by different participants),
B) Contain generic tags (e.g. "Speaker 1:", "Speaker 2:"), OR
C) Have a mix of tags or partial speaker labels.

YOUR CORE OBJECTIVE:
Accurately identify WHO is speaking every single line/turn in the transcript and format each turn as:
"SpeakerName: Utterance..."

SPEAKER IDENTIFICATION & DEDUCTION RULES:
1. DIRECT ADDRESS VS SPEAKER (CRITICAL):
   - When an utterance addresses someone by name (e.g. "[Name], you said yesterday...", "[Name], what do you think?", "Okay, then [Name], you'll need to...", "right, [Name]?"), the SPEAKER of that line is NEVER [Name]. It is someone else speaking to [Name]. A person NEVER addresses themselves in the second person ("you")!
   - When someone is directly asked a question or assigned a task ("[Name], you said yesterday that..."), the IMMEDIATE NEXT turn answering ("Yeah, that's what I found...") is spoken by [Name].
   - A participant NEVER thanks themselves: when an utterance says "Thanks, [Name]" or "Thank you, [Name]", the speaker is someone thanking [Name], NEVER [Name].
   - When someone says "Because the last time I updated it, [Name] changed half the tasks...", the immediate rebuttal "Because half the tasks were wrong." is spoken by [Name] defending their action!

2. FACILITATOR & TASK OWNERSHIP REASONING:
   - Identify the meeting facilitator/lead (e.g. who opens the meeting, calls on people, keeps track of time to finish before six, summarizes tasks, closes meeting).
   - Trace functional ownership:
     * Cross-reference stated responsibilities, task assignments, and domain discussions (e.g. frontend, backend/API, QA/testing, analytics, numbers/dashboard, reviews, scheduling) to attribute speakers consistently.

3. UNKNOWN / UNCONFIRMED SPEAKERS:
   - If a speaker's specific real name cannot be determined from conversational evidence, assign a consistent speaker label (e.g. "Speaker 1:", "Speaker 2:", etc.) keeping the same speaker label for turns spoken by the same unknown individual.
   - NEVER invent or hallucinate names of people not mentioned in the dialogue. If unsure of their name, use "Speaker 1:", "Speaker 2:", etc.

4. STRICT VERBATIM PRESERVATION OF WORDS:
   - You MUST PRESERVE EVERY SINGLE SPOKEN WORD VERBATIM.
   - Do NOT edit, omit, rephrase, summarize, or alter any spoken text.
   - Every single turn from the input transcript must be present in the output in the exact same chronological order.

5. OUTPUT FORMAT:
   - Return ONLY the dialogue formatted with one speaker turn per paragraph (separated by double newlines), formatted as:
     Speaker: Text
   - Do NOT include any markdown code blocks (no `+"```"+`), no introductory comments, no explanations.

TRANSCRIPT:
%s`, trimmed)

	type textGenConfig struct {
		Temperature float64 `json:"temperature"`
	}

	type textReq struct {
		Model            string         `json:"model"`
		Input            string         `json:"input"`
		GenerationConfig *textGenConfig `json:"generation_config,omitempty"`
	}

	// Model candidates: primary configured model, followed by fallback model
	models := []string{s.nameModel}
	if s.nameModel != "gemini-3.5-flash-lite" {
		models = append(models, "gemini-3.5-flash-lite")
	} else {
		models = append(models, "gemini-3.5-flash")
	}

	const apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"

	for _, modelName := range models {
		reqBody := textReq{
			Model: modelName,
			Input: prompt,
			GenerationConfig: &textGenConfig{
				Temperature: 0.0,
			},
		}

		reqBytes, err := json.Marshal(reqBody)
		if err != nil {
			log.Printf("ResolveSpeakers (%s): failed to marshal request: %v", modelName, err)
			continue
		}

		var respBody []byte
		maxRetries := 2
		backoff := 1 * time.Second

		for attempt := 0; attempt <= maxRetries; attempt++ {
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBytes))
			if err != nil {
				log.Printf("ResolveSpeakers (%s): failed to create request: %v", modelName, err)
				break
			}

			req.Header.Set("x-goog-api-key", s.apiKey)
			req.Header.Set("Content-Type", "application/json")

			resp, err := s.httpClient.Do(req)
			if err != nil {
				if ctx.Err() != nil {
					return rawTranscript
				}
				if attempt == maxRetries {
					log.Printf("ResolveSpeakers (%s): request failed after %d retries: %v", modelName, maxRetries, err)
					break
				}
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				log.Printf("ResolveSpeakers (%s): read body error: %v", modelName, err)
				break
			}

			if resp.StatusCode == http.StatusServiceUnavailable || resp.StatusCode == http.StatusTooManyRequests {
				if attempt < maxRetries {
					log.Printf("ResolveSpeakers (%s): status %d, retrying in %v...", modelName, resp.StatusCode, backoff)
					time.Sleep(backoff)
					backoff *= 2
					continue
				}
				// If 503/429 persists on this model, break to try fallback model
				log.Printf("ResolveSpeakers (%s): persistent status %d, trying fallback model...", modelName, resp.StatusCode)
				break
			}

			if resp.StatusCode != http.StatusOK {
				log.Printf("ResolveSpeakers (%s): API returned status %d: %s", modelName, resp.StatusCode, string(body))
				break
			}

			respBody = body
			break
		}

		if len(respBody) == 0 {
			continue
		}

		var intResp struct {
			OutputText string `json:"output_text,omitempty"`
			Steps      []struct {
				Content []struct {
					Text string `json:"text"`
				} `json:"content"`
			} `json:"steps"`
		}

		if err := json.Unmarshal(respBody, &intResp); err != nil {
			log.Printf("ResolveSpeakers (%s): parse response error: %v", modelName, err)
			continue
		}

		var output string
		if strings.TrimSpace(intResp.OutputText) != "" {
			output = strings.TrimSpace(intResp.OutputText)
		} else {
			var sb strings.Builder
			for _, step := range intResp.Steps {
				for _, c := range step.Content {
					sb.WriteString(c.Text)
				}
			}
			output = strings.TrimSpace(sb.String())
		}

		// Clean any markdown code fences if wrapped
		output = strings.TrimPrefix(output, "```transcript")
		output = strings.TrimPrefix(output, "```text")
		output = strings.TrimPrefix(output, "```")
		output = strings.TrimSuffix(output, "```")
		output = strings.TrimSpace(output)

		if output != "" && len(output) >= len(trimmed)/2 {
			log.Printf("ResolveSpeakers: successfully resolved speaker attribution using %s (%d chars)", modelName, len(output))
			return output
		}
	}

	return rawTranscript
}

// ─────────────────────────────────────────────
// uploadFile
// ─────────────────────────────────────────────

// uploadFile sends the audio bytes to the Gemini File API and returns the file URI.
func (s *GeminiTranscriptionService) uploadFile(ctx context.Context, audio io.Reader, filename, mimeType string) (string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)

	// Part 1: JSON metadata.
	metaHeader := textproto.MIMEHeader{}
	metaHeader.Set("Content-Type", "application/json; charset=UTF-8")
	metaPart, err := mw.CreatePart(metaHeader)
	if err != nil {
		return "", fmt.Errorf("failed to create metadata part: %w", err)
	}
	metaJSON := fmt.Sprintf(`{"file":{"displayName":%q}}`, filename)
	if _, err = io.WriteString(metaPart, metaJSON); err != nil {
		return "", fmt.Errorf("failed to write metadata: %w", err)
	}

	// Part 2: audio data.
	audioHeader := textproto.MIMEHeader{}
	audioHeader.Set("Content-Type", mimeType)
	audioPart, err := mw.CreatePart(audioHeader)
	if err != nil {
		return "", fmt.Errorf("failed to create audio part: %w", err)
	}
	if _, err = io.Copy(audioPart, audio); err != nil {
		return "", fmt.Errorf("failed to write audio data: %w", err)
	}

	if err = mw.Close(); err != nil {
		return "", fmt.Errorf("failed to close multipart writer: %w", err)
	}

	const uploadURL = "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadURL, &body)
	if err != nil {
		return "", fmt.Errorf("failed to build upload request: %w", err)
	}
	req.Header.Set("x-goog-api-key", s.apiKey)
	req.Header.Set("Content-Type", "multipart/related; boundary="+mw.Boundary())

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read upload response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("file upload API returned %d: %s", resp.StatusCode, string(respBody))
	}

	var uploadResp uploadFileResponse
	if err = json.Unmarshal(respBody, &uploadResp); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %w", err)
	}

	if uploadResp.File.URI == "" {
		return "", fmt.Errorf("file upload returned empty URI")
	}

	return uploadResp.File.URI, nil
}

// ─────────────────────────────────────────────
// transcribeWithDiarization
// ─────────────────────────────────────────────

// transcribeWithDiarization calls the /v1beta/interactions endpoint with
// the official transcription_config.mode including speaker diarization.
// This is the correct endpoint for gemini-3.5-transcribe (NOT generateContent).
func (s *GeminiTranscriptionService) transcribeWithDiarization(ctx context.Context, fileURI, mimeType string) (string, error) {
	reqBody := audioInteractionsRequest{
		Model: s.model,
		Input: []inputItem{
			{
				Type:     "audio",
				URI:      fileURI,
				MIMEType: mimeType,
			},
		},
		GenerationConfig: generationConfig{
			TranscriptionConfig: transcriptionConfig{
				Mode: transcriptionMode{
					// "verbatim" is required when diarization_mode is set.
					// It disables smart formatting (filler-word removal, etc.)
					// in favour of accurate verbatim output with speaker labels.
					Type:                   "verbatim",
					DiarizationMode:        "speaker",
					TimestampGranularities: []string{"word"},
				},
			},
		},
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	const interactionsURL = "https://generativelanguage.googleapis.com/v1beta/interactions"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, interactionsURL, bytes.NewReader(reqBytes))
	if err != nil {
		return "", fmt.Errorf("failed to build interactions request: %w", err)
	}
	req.Header.Set("x-goog-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("interactions request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read interactions response: %w", err)
	}

	var iResp audioInteractionsResponse
	if err = json.Unmarshal(respBody, &iResp); err != nil {
		return "", fmt.Errorf("failed to parse interactions response: %w", err)
	}

	// Check for API-level errors before checking status code.
	if iResp.Error != nil {
		return "", fmt.Errorf("Gemini API error (%d): %s", iResp.Error.Code, iResp.Error.Message)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("interactions API returned %d: %s", resp.StatusCode, string(respBody))
	}

	transcript := formatDiarizedTranscript(iResp.Steps)
	if strings.TrimSpace(transcript) != "" {
		return strings.TrimSpace(transcript), nil
	}

	return "", fmt.Errorf("no transcript returned by Gemini")
}

// formatDiarizedTranscript formats the transcript with speaker labels (e.g., "Speaker 1: ...")
// using word-level annotations or content speaker metadata if available, falling back
// to plain text if no speaker attribution is present.
func formatDiarizedTranscript(steps []audioInteractionStep) string {
	speakerMap := make(map[string]string)
	nextSpeakerNum := 1

	getSpeakerLabel := func(raw string) string {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			return "Speaker"
		}
		if label, ok := speakerMap[raw]; ok {
			return label
		}
		label := fmt.Sprintf("Speaker %d", nextSpeakerNum)
		nextSpeakerNum++
		speakerMap[raw] = label
		return label
	}

	type turn struct {
		speaker string
		words   []string
	}

	var turns []turn

	// Path 1: Check for word-level annotations with speaker labels
	hasAnnotations := false
	for _, step := range steps {
		for _, content := range step.Content {
			for _, ann := range content.Annotations {
				spk := ann.Speaker
				if spk == "" {
					spk = ann.SpeakerLabel
				}
				if spk != "" && strings.TrimSpace(ann.Text) != "" {
					hasAnnotations = true
					if len(turns) == 0 || turns[len(turns)-1].speaker != spk {
						turns = append(turns, turn{
							speaker: spk,
							words:   []string{strings.TrimSpace(ann.Text)},
						})
					} else {
						turns[len(turns)-1].words = append(turns[len(turns)-1].words, strings.TrimSpace(ann.Text))
					}
				}
			}
		}
	}

	if hasAnnotations && len(turns) > 0 {
		var lines []string
		for _, t := range turns {
			line := fmt.Sprintf("%s: %s", getSpeakerLabel(t.speaker), strings.Join(t.words, " "))
			lines = append(lines, line)
		}
		return strings.Join(lines, "\n")
	}

	// Path 2: Check for per-content or per-step speaker attribution
	hasContentSpeakers := false
	for _, step := range steps {
		stepSpk := step.Speaker
		for _, content := range step.Content {
			spk := content.Speaker
			if spk == "" {
				spk = content.SpeakerLabel
			}
			if spk == "" {
				spk = stepSpk
			}
			if spk != "" && strings.TrimSpace(content.Text) != "" {
				hasContentSpeakers = true
				turns = append(turns, turn{
					speaker: spk,
					words:   []string{strings.TrimSpace(content.Text)},
				})
			}
		}
	}

	if hasContentSpeakers && len(turns) > 0 {
		var lines []string
		for _, t := range turns {
			line := fmt.Sprintf("%s: %s", getSpeakerLabel(t.speaker), strings.Join(t.words, " "))
			lines = append(lines, line)
		}
		return strings.Join(lines, "\n")
	}

	// Path 3: Plain text fallback if no speaker annotations were returned
	var sb strings.Builder
	for _, step := range steps {
		for _, content := range step.Content {
			if strings.TrimSpace(content.Text) != "" {
				if sb.Len() > 0 {
					sb.WriteString(" ")
				}
				sb.WriteString(strings.TrimSpace(content.Text))
			}
		}
	}
	return sb.String()
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

// mimeTypeForFilename returns the audio MIME type for the given filename.
func mimeTypeForFilename(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))

	known := map[string]string{
		".mp3":  "audio/mpeg",
		".mp4":  "audio/mp4",
		".m4a":  "audio/mp4",
		".wav":  "audio/wav",
		".flac": "audio/flac",
		".ogg":  "audio/ogg",
		".webm": "audio/webm",
		".aac":  "audio/aac",
		".aiff": "audio/aiff",
	}

	if mt, ok := known[ext]; ok {
		return mt
	}

	if mt := mime.TypeByExtension(ext); mt != "" {
		return mt
	}

	return "audio/mpeg"
}
