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

	nameModel := os.Getenv("GEMINI_TOPIC_MODEL")
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
	namedTranscript := s.resolveSpeakerNames(ctx, transcript)
	if strings.TrimSpace(namedTranscript) != "" {
		return strings.TrimSpace(namedTranscript), nil
	}

	return transcript, nil
}

// resolveSpeakerNames uses Gemini to identify real names from intros/dialogue
// and replaces generic "Speaker X:" labels with actual names. If a speaker's
// SpeakerMappingItem represents an identified speaker's voice mapping.
type SpeakerMappingItem struct {
	SpeakerID  string   `json:"speaker_id"`
	Name       *string  `json:"name"`
	Confidence float64  `json:"confidence"`
	Evidence   []string `json:"evidence"`
}

type SpeakerIdentityMap struct {
	Speakers []SpeakerMappingItem `json:"speakers"`
}

// resolveSpeakerNames uses Gemini to analyze diarized speakers and their dialogue
// to build a voice-to-identity map without rewriting or modifying the transcript.
func (s *GeminiTranscriptionService) resolveSpeakerNames(ctx context.Context, rawTranscript string) string {
	trimmed := strings.TrimSpace(rawTranscript)
	if trimmed == "" {
		return rawTranscript
	}

	prompt := fmt.Sprintf(`You are a speaker identity and voice-association system for a multi-person meeting transcript.

You are given a transcript produced by a speech-to-text system with speaker diarization.

The transcript contains speaker identifiers such as:
Speaker 1:
Speaker 2:
Speaker 3:

Your task is to determine which real person each diarized speaker represents by analyzing conversational evidence.

IMPORTANT: This is an identity-mapping task, NOT a transcription task.
DO NOT rewrite the transcript.
DO NOT correct the transcript.
DO NOT change any spoken words.
DO NOT generate a new transcript.
ONLY return the speaker-to-person mapping as JSON.

IDENTITY ANALYSIS RULES:
1. SELF-INTRODUCTION: If a speaker says "My name is...", "I'm...", etc., associate that speaker's diarized voice with that name.
2. DIRECT ADDRESS: If someone addresses a speaker ("What do you think?"), and the diarized speaker responds immediately as the addressed participant, use this as evidence.
3. INTRODUCTION BY ANOTHER PERSON: If someone introduces a colleague and the diarization shows which speaker responds, associate that speaker ID with that name.
4. VOICE CONSISTENCY: Treat each diarized speaker ID as representing one consistent voice throughout the recording.
5. DO NOT CONFUSE SPEAKERS: Do not assign a name to someone simply because they mentioned that name. Only assign the name to the voice that is addressed or introduces themselves.
6. UNKNOWN SPEAKERS: If you cannot establish a speaker's real name with sufficient evidence, DO NOT GUESS. Return "name": null.
7. NEVER INVENT PEOPLE: Only use names explicitly established in the conversation.
8. PRESERVE SPEAKER IDs: Never rename or modify Speaker 1, Speaker 2, etc. in your mapping keys.

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN, NO OTHER TEXT):
{
  "speakers": [
    {
      "speaker_id": "Speaker 1",
      "name": "Full Name Or null",
      "confidence": 0.95,
      "evidence": ["Exact conversational evidence from dialogue"]
    }
  ]
}

CONFIDENCE GUIDELINES:
- 0.95 - 1.00: Identity explicitly established and confirmed.
- 0.80 - 0.94: Identity strongly supported by conversational clues.
- Below 0.80: Do not assign a name. Return null.

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

	reqBody := textReq{
		Model: s.nameModel,
		Input: prompt,
		GenerationConfig: &textGenConfig{
			Temperature: 0.0,
		},
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		log.Printf("resolveSpeakerNames: failed to marshal request: %v", err)
		return rawTranscript
	}

	const apiURL = "https://generativelanguage.googleapis.com/v1beta/interactions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBytes))
	if err != nil {
		log.Printf("resolveSpeakerNames: failed to create request: %v", err)
		return rawTranscript
	}

	req.Header.Set("x-goog-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		log.Printf("resolveSpeakerNames: request error: %v", err)
		return rawTranscript
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("resolveSpeakerNames: read body error: %v", err)
		return rawTranscript
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("resolveSpeakerNames: API returned status %d: %s", resp.StatusCode, string(body))
		return rawTranscript
	}

	var intResp struct {
		OutputText string `json:"output_text,omitempty"`
		Steps      []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"steps"`
	}

	if err := json.Unmarshal(body, &intResp); err != nil {
		log.Printf("resolveSpeakerNames: parse response error: %v", err)
		return rawTranscript
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
	output = strings.TrimPrefix(output, "```json")
	output = strings.TrimPrefix(output, "```")
	output = strings.TrimSuffix(output, "```")
	output = strings.TrimSpace(output)

	var idMap SpeakerIdentityMap
	if err := json.Unmarshal([]byte(output), &idMap); err != nil {
		log.Printf("resolveSpeakerNames: failed to decode speaker map JSON: %v (raw: %s)", err, output)
		return rawTranscript
	}

	// Build verified lookup map (confidence >= 0.80 and non-empty name)
	resolvedMap := make(map[string]string)
	for _, spk := range idMap.Speakers {
		if spk.Name != nil && strings.TrimSpace(*spk.Name) != "" && spk.Confidence >= 0.80 {
			cleanName := strings.TrimSpace(*spk.Name)
			// Ensure speaker ID is normalized e.g. "Speaker 1"
			spkKey := strings.TrimSpace(spk.SpeakerID)
			resolvedMap[spkKey] = cleanName
			log.Printf("resolveSpeakerNames: mapped %s -> %s (conf: %.2f)", spkKey, cleanName, spk.Confidence)
		}
	}

	if len(resolvedMap) == 0 {
		log.Printf("resolveSpeakerNames: no high-confidence speaker names resolved; keeping generic speaker labels")
		return rawTranscript
	}

	// Deterministically replace "Speaker X:" labels line by line without altering spoken words
	lines := strings.Split(rawTranscript, "\n")
	for i, line := range lines {
		trimmedLine := strings.TrimSpace(line)
		for spkID, realName := range resolvedMap {
			prefix := spkID + ":"
			if strings.HasPrefix(trimmedLine, prefix) {
				restOfLine := strings.TrimPrefix(trimmedLine, prefix)
				lines[i] = realName + ":" + restOfLine
				break
			}
		}
	}

	finalTranscript := strings.Join(lines, "\n")
	log.Printf("resolveSpeakerNames: successfully replaced speaker labels with %d resolved names", len(resolvedMap))
	return finalTranscript
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
