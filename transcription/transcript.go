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
	"net"
	"net/http"
	"net/textproto"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
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
// sharedPooledTransport maintains a hot pool of persistent HTTP/2 connections.
// Go's DefaultTransport restricts MaxIdleConnsPerHost to 2, forcing concurrent workers
// to pay TLS 1.3 handshake overhead repeatedly. 32 idle connections keeps all workers warm.
var sharedPooledTransport = &http.Transport{
	Proxy: http.ProxyFromEnvironment,
	DialContext: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 60 * time.Second,
	}).DialContext,
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          100,
	MaxIdleConnsPerHost:   32,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
}

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
			Transport: sharedPooledTransport,
			Timeout:   5 * time.Minute, // audio upload + transcription can take time
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
		Code    any    `json:"code"`
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
// In Phase 2, long audio (>= 3 minutes) is safely sliced into 2-3 chunks
// with 2.5s overlap and transcribed in parallel within the strict 3 RPM limit,
// seamlessly stitched by the Overlap Seam Resolver.
func (s *GeminiTranscriptionService) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	mimeType := mimeTypeForFilename(filename)

	audioBytes, err := io.ReadAll(audio)
	if err != nil {
		return "", fmt.Errorf("failed to read audio data: %w", err)
	}

	enableChunking := os.Getenv("ENABLE_AUDIO_CHUNKING") != "false"
	isWAV := strings.HasSuffix(strings.ToLower(filename), ".wav")

	// Phase 2: Parallel Audio Chunking for long files (>= 180s)
	if enableChunking && isWAV {
		chunks, splitErr := SplitWAV(audioBytes, 3)
		if splitErr == nil && len(chunks) > 1 {
			log.Printf("[AudioChunker] Slicing %s into %d chunks with 2.5s overlap for parallel processing (3 RPM safe)...", filename, len(chunks))

			chunkTranscripts := make([]string, len(chunks))
			chunkErrs := make([]error, len(chunks))
			var wg sync.WaitGroup

			for i, c := range chunks {
				wg.Add(1)
				go func(idx int, chunk AudioChunk) {
					defer wg.Done()
					cFilename := fmt.Sprintf("chunk_%d_%s", idx, filename)
					uri, uErr := s.uploadFile(ctx, bytes.NewReader(chunk.Data), cFilename, "audio/wav")
					if uErr != nil {
						chunkErrs[idx] = fmt.Errorf("chunk %d upload failed: %w", idx, uErr)
						return
					}
					txt, tErr := s.transcribeWithDiarization(ctx, uri, "audio/wav")
					if tErr != nil {
						chunkErrs[idx] = fmt.Errorf("chunk %d transcribe failed: %w", idx, tErr)
						return
					}
					chunkTranscripts[idx] = txt
				}(i, c)
			}
			wg.Wait()

			for idx, cErr := range chunkErrs {
				if cErr != nil {
					log.Printf("[AudioChunker] Chunk %d failed: %v", idx, cErr)
					return "", fmt.Errorf("transcription failed on chunk %d: %w", idx, cErr)
				}
			}

			stitched := ResolveSeams(chunkTranscripts)
			if strings.TrimSpace(stitched) == "" {
				return "", fmt.Errorf("transcription failed: seam resolution yielded empty result")
			}
			log.Printf("[AudioChunker] Successfully stitched %d chunks for %s", len(chunks), filename)
			return stitched, nil
		}
	}

	// Single-pass transcription (used for short files under duration threshold, non-WAV formats, or when chunking is disabled)
	fileURI, err := s.uploadFile(ctx, bytes.NewReader(audioBytes), filename, mimeType)
	if err != nil {
		return "", fmt.Errorf("file upload failed: %w", err)
	}

	transcript, err := s.transcribeWithDiarization(ctx, fileURI, mimeType)
	if err != nil {
		return "", fmt.Errorf("transcription failed: %w", err)
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
1. DIRECT ADDRESS, FLOOR HANDOFFS & RESPONDENTS (CRITICAL EVIDENCE):
   - A PERSON NEVER ADDRESSES OR THANKS THEMSELVES IN THE SECOND PERSON ("you"):
     * When an utterance addresses someone by name (e.g. "thanks, [Name]", "[Name], could you...", "Okay, then [Name], you'll need to..."), the SPEAKER of that line is NEVER [Name]. It is someone else speaking to [Name]! A person NEVER says "Thanks, [Name]" or "[Name], could you..." to themselves!
     * If an utterance thanks [Name 1] and invites the next speaker ("Okay, thanks, [Name 1]. That was all useful information. So, [Name 2], can we turn to you now?"), it is the CHAIR / FACILITATOR speaking, NOT [Name 1]!
     * When someone asks "[Name], could you look into this?", and the immediate response is "Sure", the speaker of "Sure" is [Name] accepting the task!
   - FLOOR HANDOFFS & AGENDA PRESENTERS: When a speaker calls upon a participant by name to speak, update, present, or begin an agenda topic (e.g. "[Name], could you take us through the update?", "[Name], over to you", "Could you update us on this, [Name]?", "Let's hear from [Name]"), the IMMEDIATE NEXT turn responding or taking the floor (even if brief: "Sure.", "Thanks.", "Yeah, so...", "Right.", "Okay.") is spoken by [Name]! Attribute that turn as "[Name]:".
   - QUESTIONS & TASKS: When someone is directly asked a question or assigned a task ("[Name], what is the status?"), the IMMEDIATE NEXT turn answering is spoken by [Name].
   - DEFENDING ACTIONS & REBUTTALS: When someone says "Because the last time I updated it, [Name] changed half the tasks...", the immediate rebuttal "Because half the tasks were wrong." is spoken by [Name] defending their action!
   - THIRD-PARTY MENTIONS VS ACTIVE PARTICIPANTS: Distinguish between non-attending third parties mentioned in passing (e.g. an external contact who will arrive later or sent apologies) versus participants directly addressed in the room who actively respond (e.g. "[Name], could you start?" -> "Sure, happy to" - [Name] is present and speaks!).

2. FACILITATOR & TASK OWNERSHIP REASONING:
   - Identify the meeting facilitator/lead (e.g. who opens the meeting, calls on people, keeps track of time to finish before six, summarizes tasks, closes meeting).
   - Trace functional ownership:
     * Cross-reference stated responsibilities, task assignments, and domain discussions (e.g. frontend, backend/API, QA/testing, analytics, numbers/dashboard, reviews, scheduling) to attribute speakers consistently.

3. UNKNOWN / UNCONFIRMED SPEAKERS:
   - Only use generic speaker labels (e.g. "Speaker 1:", "Speaker 2:", etc.) if a speaker's real name genuinely cannot be determined from conversational evidence, introductions, direct address, or handoffs.
   - When conversational evidence reveals a speaker's identity (such as being called on by name and responding), ALWAYS replace the generic tag with their real name.
   - NEVER invent or hallucinate names of people not mentioned in the dialogue.

4. STRICT VERBATIM PRESERVATION OF WORDS:
   - You MUST PRESERVE EVERY SINGLE SPOKEN WORD VERBATIM.
   - Do NOT edit, omit, rephrase, summarize, or alter any spoken text.
   - Every single turn from the input transcript must be present in the output in the exact same chronological order.

5. STRICT OUTPUT FORMAT (NO SQUARE BRACKETS, NO EDITORIAL NOTES, NO RUN-ON PARAGRAPHS):
   - Format each turn strictly as:
     SpeakerName: Spoken text
   - Place each speaker turn on its OWN SEPARATE LINE separated by double newlines (\n\n). NEVER merge turns into a single paragraph!
   - NEVER put square brackets around speaker names (write "Anna:" NOT "[Anna]:", write "Marcus:" NOT "[Marcus]:").
   - NEVER output editorial notes, commentary, justifications, or bracketed explanations (NEVER write "[Speaker 5 speaking error...]" or "[Marcus role/chairperson slip...]").
   - Do NOT include markdown code blocks, introductory text, or explanations. Only the clean formatted dialogue.

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
			output = NormalizeTranscriptDisplay(sanitizeDiscourseAttributions(output, "", nil))
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

	payloadBytes := body.Bytes()
	contentType := "multipart/related; boundary=" + mw.Boundary()

	maxRetries := 3
	for attempt := 0; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadURL, bytes.NewReader(payloadBytes))
		if err != nil {
			return "", fmt.Errorf("failed to build upload request: %w", err)
		}
		req.Header.Set("x-goog-api-key", s.apiKey)
		req.Header.Set("Content-Type", contentType)

		resp, err := s.httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("upload request canceled: %w", ctx.Err())
			}
			if attempt == maxRetries {
				return "", fmt.Errorf("upload request failed: %w", err)
			}
			time.Sleep(5 * time.Second)
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("failed to read upload response: %w", err)
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			if attempt < maxRetries {
				log.Printf("[Gemini File Upload] Rate limit 429 hit. Worker waiting 30s before retry %d/%d...", attempt+1, maxRetries)
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-time.After(30 * time.Second):
				}
				continue
			}
			return "", fmt.Errorf("file upload API returned 429 rate limit after retries: %s", string(respBody))
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

	return "", fmt.Errorf("file upload failed: exceeded max retries")
}

// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// Global 3 RPM Rate Limiter for Audio Transcription
// ─────────────────────────────────────────────

var (
	geminiAudioSlotMu    sync.Mutex
	geminiAudioSlotsUsed []time.Time
)

// acquireGeminiTranscriptionSlot ensures outgoing audio transcription requests
// strictly adhere to the Free Tier limit of 3 Requests Per Minute (3 RPM).
// If 3 requests were already issued within the last 62 seconds, it blocks safely until
// an available slot opens up, completely eliminating HTTP 429 rate limit errors.
func acquireGeminiTranscriptionSlot(ctx context.Context) error {
	geminiAudioSlotMu.Lock()
	defer geminiAudioSlotMu.Unlock()

	const maxRPM = 3
	const window = 62 * time.Second

	for {
		now := time.Now()
		valid := make([]time.Time, 0, len(geminiAudioSlotsUsed))
		for _, t := range geminiAudioSlotsUsed {
			if now.Sub(t) < window {
				valid = append(valid, t)
			}
		}
		geminiAudioSlotsUsed = valid

		if len(geminiAudioSlotsUsed) < maxRPM {
			geminiAudioSlotsUsed = append(geminiAudioSlotsUsed, now)
			return nil
		}

		oldest := geminiAudioSlotsUsed[0]
		waitDuration := window - now.Sub(oldest) + 500*time.Millisecond
		log.Printf("[Gemini Transcribe] 3 RPM pacing: %d requests active in last 60s. Waiting %v for rate limit slot...", len(geminiAudioSlotsUsed), waitDuration)

		geminiAudioSlotMu.Unlock()
		select {
		case <-ctx.Done():
			geminiAudioSlotMu.Lock()
			return ctx.Err()
		case <-time.After(waitDuration):
		}
		geminiAudioSlotMu.Lock()
	}
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

	maxRetries := 4
	backoff := 15 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Acquire 3 RPM rate limit slot before dispatching request
		if err := acquireGeminiTranscriptionSlot(ctx); err != nil {
			return "", err
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, interactionsURL, bytes.NewReader(reqBytes))
		if err != nil {
			return "", fmt.Errorf("failed to build interactions request: %w", err)
		}
		req.Header.Set("x-goog-api-key", s.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := s.httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("interactions request canceled: %w", ctx.Err())
			}
			if attempt == maxRetries {
				return "", fmt.Errorf("interactions request failed after %d retries: %w", maxRetries, err)
			}
			log.Printf("[Gemini Transcribe] Network error (attempt %d/%d): %v, retrying in %v...", attempt+1, maxRetries, err, backoff)
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("failed to read interactions response: %w", err)
		}

		var iResp audioInteractionsResponse
		_ = json.Unmarshal(respBody, &iResp)

		isRateLimit := resp.StatusCode == http.StatusTooManyRequests ||
			(iResp.Error != nil && (fmt.Sprintf("%v", iResp.Error.Code) == "429" ||
				strings.Contains(strings.ToLower(iResp.Error.Message), "rate limit") ||
				strings.Contains(strings.ToLower(iResp.Error.Message), "too_many_requests")))

		if isRateLimit {
			errMsg := ""
			if iResp.Error != nil {
				errMsg = strings.ToLower(iResp.Error.Message)
			} else {
				errMsg = strings.ToLower(string(respBody))
			}

			// Distinguish temporary per-minute rate limits from genuine daily quota exhaustion
			isPerMinuteLimit := strings.Contains(errMsg, "per minute") ||
				strings.Contains(errMsg, "tokens per minute") ||
				(strings.Contains(errMsg, "retry in") && strings.Contains(errMsg, "s") && !strings.Contains(errMsg, "h") && !strings.Contains(errMsg, "d"))

			isDailyQuota := !isPerMinuteLimit && (strings.Contains(errMsg, "per day") || strings.Contains(errMsg, "daily limit"))

			if isDailyQuota {
				if iResp.Error != nil {
					return "", fmt.Errorf("Gemini API error (%v): %s", iResp.Error.Code, iResp.Error.Message)
				}
				return "", fmt.Errorf("Gemini daily quota exceeded: %s", string(respBody))
			}

			if attempt < maxRetries {
				waitDuration := 45 * time.Second
				if attempt >= 1 {
					waitDuration = 62 * time.Second
				}
				// Parse explicit retry duration if provided (e.g. "retry in 22s")
				if strings.Contains(errMsg, "retry in") {
					re := regexp.MustCompile(`retry in (\d+)s`)
					if m := re.FindStringSubmatch(errMsg); len(m) >= 2 {
						if sVal, err := strconv.Atoi(m[1]); err == nil && sVal > 0 {
							waitDuration = time.Duration(sVal+2) * time.Second
						}
					}
				}
				log.Printf("[Gemini Transcribe] Rate limit hit. Worker waiting %v before retry %d/%d...", waitDuration, attempt+1, maxRetries)
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-time.After(waitDuration):
				}
				continue
			}
			if iResp.Error != nil {
				return "", fmt.Errorf("Gemini API error (%v): %s", iResp.Error.Code, iResp.Error.Message)
			}
			return "", fmt.Errorf("interactions API returned 429 rate limit after %d retries", maxRetries)
		}

		// Check for API-level errors before checking status code.
		if iResp.Error != nil {
			return "", fmt.Errorf("Gemini API error (%v): %s", iResp.Error.Code, iResp.Error.Message)
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

	return "", fmt.Errorf("transcription failed: exceeded maximum retries")
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
