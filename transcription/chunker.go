package transcription

import (
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strings"
	"unicode"
)

// AudioChunk represents a sliced audio segment with its header and metadata.
type AudioChunk struct {
	Index    int
	StartSec float64
	EndSec   float64
	Data     []byte
	Filename string
}

// WAVInfo contains parsed WAV metadata.
type WAVInfo struct {
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
	DataOffset    int
	DataLength    int
	DurationSec   float64
}

// ParseWAV inspects and validates a WAV byte buffer, extracting audio parameters and PCM data range.
func ParseWAV(buf []byte) (*WAVInfo, error) {
	if len(buf) < 44 {
		return nil, fmt.Errorf("WAV data too small (%d bytes)", len(buf))
	}

	if string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid RIFF/WAVE header")
	}

	info := &WAVInfo{}
	hasFmt := false
	hasData := false

	offset := 12
	for offset+8 <= len(buf) {
		chunkID := string(buf[offset : offset+4])
		chunkSize := int(binary.LittleEndian.Uint32(buf[offset+4 : offset+8]))
		offset += 8

		if offset+chunkSize > len(buf) {
			// Malformed chunk size, cap to available buffer
			chunkSize = len(buf) - offset
		}

		switch chunkID {
		case "fmt ":
			if chunkSize < 14 {
				return nil, fmt.Errorf("fmt chunk too small (%d bytes)", chunkSize)
			}
			info.NumChannels = binary.LittleEndian.Uint16(buf[offset+2 : offset+4])
			info.SampleRate = binary.LittleEndian.Uint32(buf[offset+4 : offset+8])
			info.ByteRate = binary.LittleEndian.Uint32(buf[offset+8 : offset+12])
			info.BlockAlign = binary.LittleEndian.Uint16(buf[offset+12 : offset+14])
			if chunkSize >= 16 {
				info.BitsPerSample = binary.LittleEndian.Uint16(buf[offset+14 : offset+16])
			} else {
				info.BitsPerSample = 16
			}
			hasFmt = true

		case "data":
			info.DataOffset = offset
			info.DataLength = chunkSize
			hasData = true
		}

		offset += chunkSize
		// RIFF chunks are padded to 2-byte boundaries
		if chunkSize%2 != 0 {
			offset++
		}
	}

	if !hasFmt || !hasData {
		return nil, fmt.Errorf("WAV missing fmt or data chunk (hasFmt=%t, hasData=%t)", hasFmt, hasData)
	}

	if info.ByteRate == 0 {
		info.ByteRate = info.SampleRate * uint32(info.NumChannels) * uint32(info.BitsPerSample) / 8
	}

	if info.ByteRate > 0 {
		info.DurationSec = float64(info.DataLength) / float64(info.ByteRate)
	}

	return info, nil
}

// FindQuietFrame searches for the lowest RMS energy frame within [startSec, endSec] of the PCM data.
// This ensures split cuts occur during natural pauses/silence between words or sentences.
func FindQuietFrame(pcm []byte, info *WAVInfo, startSec, endSec float64) float64 {
	if info.ByteRate == 0 || len(pcm) == 0 {
		return (startSec + endSec) / 2.0
	}

	startByte := int(startSec * float64(info.ByteRate))
	endByte := int(endSec * float64(info.ByteRate))

	if startByte < 0 {
		startByte = 0
	}
	if endByte > len(pcm) {
		endByte = len(pcm)
	}

	align := int(info.BlockAlign)
	if align <= 0 {
		align = 4
	}
	startByte = (startByte / align) * align
	endByte = (endByte / align) * align

	if startByte >= endByte {
		return startSec
	}

	// 100ms analysis frame
	frameBytes := int(0.100 * float64(info.ByteRate))
	frameBytes = (frameBytes / align) * align
	if frameBytes <= 0 {
		frameBytes = align * 16
	}

	stepBytes := frameBytes / 2
	if stepBytes < align {
		stepBytes = align
	}

	minRMS := math.MaxFloat64
	bestOffset := startByte

	for cur := startByte; cur+frameBytes <= endByte; cur += stepBytes {
		var sumSq float64
		sampleCount := 0

		if info.BitsPerSample == 16 {
			for i := cur; i+2 <= cur+frameBytes; i += 2 {
				val := int16(binary.LittleEndian.Uint16(pcm[i : i+2]))
				sumSq += float64(val) * float64(val)
				sampleCount++
			}
		} else {
			for i := cur; i < cur+frameBytes; i++ {
				val := int(pcm[i]) - 128
				sumSq += float64(val) * float64(val)
				sampleCount++
			}
		}

		if sampleCount > 0 {
			rms := math.Sqrt(sumSq / float64(sampleCount))
			if rms < minRMS {
				minRMS = rms
				bestOffset = cur
			}
		}
	}

	return float64(bestOffset+frameBytes/2) / float64(info.ByteRate)
}

// BuildWAV constructs a valid 44-byte standard PCM WAV file from raw PCM audio bytes.
func BuildWAV(pcm []byte, info *WAVInfo) []byte {
	buf := make([]byte, 44+len(pcm))

	copy(buf[0:4], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:8], uint32(36+len(pcm)))
	copy(buf[8:12], "WAVE")

	copy(buf[12:16], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:20], 16) // Subchunk1Size for PCM
	binary.LittleEndian.PutUint16(buf[20:22], 1)  // AudioFormat = 1 (PCM)
	binary.LittleEndian.PutUint16(buf[22:24], info.NumChannels)
	binary.LittleEndian.PutUint32(buf[24:28], info.SampleRate)
	binary.LittleEndian.PutUint32(buf[28:32], info.ByteRate)
	binary.LittleEndian.PutUint16(buf[32:34], info.BlockAlign)
	binary.LittleEndian.PutUint16(buf[34:36], info.BitsPerSample)

	copy(buf[36:40], "data")
	binary.LittleEndian.PutUint32(buf[40:44], uint32(len(pcm)))
	copy(buf[44:], pcm)

	return buf
}

// SplitWAV splits a WAV byte slice into 2 or 3 overlapping chunks based on silence boundaries.
// Strict constraint: maxChunks is capped at 3 to strictly obey the 3 RPM limit of gemini-3.5-transcribe.
func SplitWAV(audioBytes []byte, maxChunks int) ([]AudioChunk, error) {
	info, err := ParseWAV(audioBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse WAV: %w", err)
	}

	// Short audio (<= 180s = 3 minutes) is processed in single-pass with no chunking.
	const minChunkDurationSec = 180.0
	if info.DurationSec <= minChunkDurationSec {
		return []AudioChunk{
			{
				Index:    0,
				StartSec: 0,
				EndSec:   info.DurationSec,
				Data:     audioBytes,
				Filename: "chunk_0.wav",
			},
		}, nil
	}

	// For files > 7 minutes, split into 2 or 3 chunks (capped at 3 to protect 3 RPM limit)
	numChunks := 2
	if info.DurationSec > 600.0 { // > 10 minutes
		numChunks = 3
	}
	if maxChunks > 0 && numChunks > maxChunks {
		numChunks = maxChunks
	}
	if numChunks > 3 {
		numChunks = 3
	}

	pcmData := audioBytes[info.DataOffset : info.DataOffset+info.DataLength]
	align := int(info.BlockAlign)
	if align <= 0 {
		align = 4
	}

	// Overlap window: 2.5 seconds at seam boundaries guarantees zero syllables are truncated
	overlapSec := 2.5

	type splitPoint struct {
		targetSec float64
		splitSec  float64
	}

	splitPoints := make([]splitPoint, numChunks-1)
	for i := 0; i < numChunks-1; i++ {
		target := info.DurationSec * float64(i+1) / float64(numChunks)
		// Search +/- 5 seconds around target for quietest RMS frame
		searchStart := target - 5.0
		searchEnd := target + 5.0
		if searchStart < 0 {
			searchStart = 0
		}
		if searchEnd > info.DurationSec {
			searchEnd = info.DurationSec
		}
		quietSec := FindQuietFrame(pcmData, info, searchStart, searchEnd)
		splitPoints[i] = splitPoint{
			targetSec: target,
			splitSec:  quietSec,
		}
	}

	chunks := make([]AudioChunk, numChunks)
	for i := 0; i < numChunks; i++ {
		var startSec, endSec float64

		if i == 0 {
			startSec = 0
			endSec = splitPoints[0].splitSec + overlapSec
		} else if i == numChunks-1 {
			startSec = splitPoints[i-1].splitSec
			endSec = info.DurationSec
		} else {
			startSec = splitPoints[i-1].splitSec
			endSec = splitPoints[i].splitSec + overlapSec
		}

		if endSec > info.DurationSec {
			endSec = info.DurationSec
		}

		startByte := int(startSec * float64(info.ByteRate))
		endByte := int(endSec * float64(info.ByteRate))

		startByte = (startByte / align) * align
		endByte = (endByte / align) * align

		if startByte < 0 {
			startByte = 0
		}
		if endByte > len(pcmData) {
			endByte = len(pcmData)
		}
		if startByte >= endByte {
			startByte = 0
			endByte = len(pcmData)
		}

		chunkPCM := pcmData[startByte:endByte]
		chunkWAV := BuildWAV(chunkPCM, info)

		chunks[i] = AudioChunk{
			Index:    i,
			StartSec: startSec,
			EndSec:   endSec,
			Data:     chunkWAV,
			Filename: fmt.Sprintf("chunk_%d.wav", i),
		}
	}

	return chunks, nil
}

// ─────────────────────────────────────────────
// Overlap Seam Resolver
// ─────────────────────────────────────────────

type speakerTurn struct {
	Speaker string
	Text    string
}

var (
	speakerTurnRegex          = regexp.MustCompile(`^(?:\[)?([A-Za-z0-9 _.-]{1,35})(?:\s+role/[^\]]+|\s+speaking error[^\]]*|\s+slip[^\]]*)?(?:\])?:\s*(.*)$`)
	reSpeakerPrefixBracket    = regexp.MustCompile(`(?i)\bSpeaker:\s*\[([A-Za-z0-9 _.-]{1,35})\]:\s*`)
	reSpeakerPrefixNarrator   = regexp.MustCompile(`(?i)\bSpeaker:\s*\[(?:Speaker\s*1|Narrator)\]:\s*`)
	reBracketSpeakerLine      = regexp.MustCompile(`(?m)^\s*\[([A-Za-z0-9 _.-]{1,35})(?:\s+role/[^\]]+|\s+slip[^\]]*)?\]:\s*`)
	reInlineBracketSpeaker    = regexp.MustCompile(`(?i)\s*\[([A-Za-z0-9 _.-]{1,35})(?:\s+role/[^\]]+|\s+slip[^\]]*)?\]:\s*`)
	reEditorialWithText       = regexp.MustCompile(`(?i)\[(?:Spoken by|Speaker\s*\d+|[A-Za-z0-9 _.-]+)\s*(?:speaking error|slip|based on|role)[^:]*:\s*([\s\S]*?)\]`)
	reEditorialStandalone     = regexp.MustCompile(`(?i)\[(?:Speaker\s*\d+\s+speaking error|role/chairperson slip|[A-Za-z0-9 _.-]+\s+slip|speaking error in transcript/attribution context)[^\]]*\]\s*`)
	reGenericBracketsInText   = regexp.MustCompile(`(?i)^\s*\[(?:Note|Editorial|Context)[^:]*:\s*([\s\S]*?)\]\s*$`)
	reSpeakerInSpeaker        = regexp.MustCompile(`(?i)^\s*\[(Speaker\s*\d+)\]:\s*([\s\S]*)$`)
	reInlineSpeakerAfterPunct = regexp.MustCompile(`([.!?])\s+([A-Z][a-zA-Z0-9 _.-]{1,25}):\s+`)
)

func cleanTurnEditorialText(speaker, text string) (string, string) {
	speaker = strings.TrimSpace(speaker)
	speaker = strings.TrimPrefix(speaker, "[")
	speaker = strings.TrimSuffix(speaker, "]")
	speaker = strings.TrimSpace(speaker)

	// Clean any "[Marcus role/chairperson slip in transcript text]:" lingering in speaker name
	reSlipInSpeaker := regexp.MustCompile(`(?i)^([A-Za-z0-9 _.-]{1,25})\s+(?:role/[^\]]+|slip[^\]]*|speaking error[^\]]*)`)
	if m := reSlipInSpeaker.FindStringSubmatch(speaker); len(m) >= 2 {
		speaker = strings.TrimSpace(m[1])
	}

	text = strings.TrimSpace(text)

	// Strip standalone editorial brackets like "[Speaker 5 speaking error in transcript/attribution context]"
	text = reEditorialStandalone.ReplaceAllString(text, "")

	// Unwrap bracketed meta text: [Spoken by X based on...: text]
	text = reEditorialWithText.ReplaceAllString(text, "$1")

	// Unwrap full note text: [Note: text]
	if m := reGenericBracketsInText.FindStringSubmatch(text); len(m) >= 2 {
		text = strings.TrimSpace(m[1])
	}

	// Case: Speaker: [Speaker 1]: Hello -> Speaker 1: Hello
	if strings.EqualFold(speaker, "Speaker") {
		if m := reSpeakerInSpeaker.FindStringSubmatch(text); len(m) >= 3 {
			speaker = strings.TrimSpace(m[1])
			text = strings.TrimSpace(m[2])
		} else if strings.HasPrefix(strings.ToLower(text), "speaker 1:") {
			speaker = "Speaker 1"
			text = strings.TrimSpace(text[10:])
		}
	}

	// Strip any lingering leading [Speaker N]: or [Name]: prefix in text
	if m := reSpeakerInSpeaker.FindStringSubmatch(text); len(m) >= 3 {
		text = strings.TrimSpace(m[2])
	}
	reLeadingBracketSpk := regexp.MustCompile(`^\s*\[([A-Za-z0-9 _.-]{1,35})\]:\s*`)
	if m := reLeadingBracketSpk.FindStringSubmatch(text); len(m) >= 2 {
		text = strings.TrimSpace(text[len(m[0]):])
	}

	return speaker, strings.TrimSpace(text)
}

// normalizeInlineSpeakerTurns unpacks any squished single-paragraph transcripts
// and inline speaker tags into clean, distinct lines separated by \n\n without square brackets.
func normalizeInlineSpeakerTurns(transcript string) string {
	t := strings.TrimSpace(transcript)
	if t == "" {
		return t
	}

	// 1. Clean "Speaker: [Speaker 1]: " -> "Speaker 1: "
	t = reSpeakerPrefixNarrator.ReplaceAllString(t, "Speaker 1: ")
	t = reSpeakerPrefixBracket.ReplaceAllString(t, "$1: ")

	// 2. Strip standalone editorial notes inside brackets
	t = reEditorialStandalone.ReplaceAllString(t, "")

	// 3. Unwrap editorial text in brackets
	t = reEditorialWithText.ReplaceAllString(t, "$1")

	// 4. Unpack inline [Speaker]: tags into newline-separated turns
	// e.g. "Okay, Maya? [Maya]: Fine. [Marcus]: And then..." ->
	// "Okay, Maya?\n\nMaya: Fine.\n\nMarcus: And then..."
	t = reInlineBracketSpeaker.ReplaceAllString(t, "\n\n$1: ")

	// 5. Clean any remaining bracketed speaker names at line starts
	// e.g. "[Marcus]: Hello" -> "Marcus: Hello"
	t = reBracketSpeakerLine.ReplaceAllString(t, "$1: ")

	// 6. Unpack inline unbracketed speaker turns following punctuation
	// e.g. "Okay, Maya? Maya: Fine." -> "Okay, Maya?\n\nMaya: Fine."
	t = reInlineSpeakerAfterPunct.ReplaceAllString(t, "$1\n\n$2: ")

	// 7. Collapse excessive blank lines
	t = regexp.MustCompile(`\n{3,}`).ReplaceAllString(t, "\n\n")

	return strings.TrimSpace(t)
}

// NormalizeTranscriptDisplay unpacks all inline tags and formats each speaker turn on its own separate line (\n\n).
func NormalizeTranscriptDisplay(transcript string) string {
	turns := parseSpeakerTurns(transcript)
	if len(turns) > 0 {
		return formatTurns(turns)
	}
	return normalizeInlineSpeakerTurns(transcript)
}

func parseSpeakerTurns(transcript string) []speakerTurn {
	transcript = normalizeInlineSpeakerTurns(transcript)
	rawBlocks := strings.Split(transcript, "\n")
	var turns []speakerTurn

	var currentSpeaker string
	var currentText strings.Builder

	flush := func() {
		if currentText.Len() > 0 {
			text := strings.TrimSpace(currentText.String())
			if text != "" {
				spk := currentSpeaker
				if spk == "" {
					spk = "Speaker 1"
				}
				spk, text = cleanTurnEditorialText(spk, text)
				turns = append(turns, speakerTurn{
					Speaker: spk,
					Text:    text,
				})
			}
			currentText.Reset()
		}
	}

	for _, line := range rawBlocks {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		if matches := speakerTurnRegex.FindStringSubmatch(trimmed); matches != nil && len(matches) >= 3 {
			// Found new speaker turn
			flush()
			currentSpeaker = strings.TrimSpace(matches[1])
			currentText.WriteString(matches[2])
		} else {
			if currentText.Len() > 0 {
				currentText.WriteString(" ")
			}
			currentText.WriteString(trimmed)
		}
	}
	flush()

	return turns
}

func formatTurns(turns []speakerTurn) string {
	var sb strings.Builder
	for i, t := range turns {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(fmt.Sprintf("%s: %s", t.Speaker, t.Text))
	}
	return sb.String()
}

func normalizeToken(word string) string {
	var sb strings.Builder
	for _, r := range strings.ToLower(word) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// ResolveTwoSeams stitches two contiguous transcript chunks (chunkA and chunkB)
// by finding the overlapping verbal sequence at the boundary and deduplicating it.
func ResolveTwoSeams(transcriptA, transcriptB string) string {
	tA := strings.TrimSpace(transcriptA)
	tB := strings.TrimSpace(transcriptB)

	if tA == "" {
		return tB
	}
	if tB == "" {
		return tA
	}

	turnsA := parseSpeakerTurns(tA)
	turnsB := parseSpeakerTurns(tB)

	if len(turnsA) == 0 {
		return tB
	}
	if len(turnsB) == 0 {
		return tA
	}

	// Collect normalized words from the tail of turnsA (up to last 60 words)
	type wordRef struct {
		TurnIndex int
		WordIndex int
		Raw       string
		Norm      string
	}

	var wordsA []wordRef
	for tIdx := len(turnsA) - 1; tIdx >= 0 && len(wordsA) < 60; tIdx-- {
		rawWords := strings.Fields(turnsA[tIdx].Text)
		var turnRefs []wordRef
		for wIdx, rw := range rawWords {
			norm := normalizeToken(rw)
			if norm != "" {
				turnRefs = append(turnRefs, wordRef{
					TurnIndex: tIdx,
					WordIndex: wIdx,
					Raw:       rw,
					Norm:      norm,
				})
			}
		}
		// Prepend so order is chronological
		wordsA = append(turnRefs, wordsA...)
	}

	// Collect normalized words from the head of turnsB (up to first 60 words)
	var wordsB []wordRef
	for tIdx := 0; tIdx < len(turnsB) && len(wordsB) < 60; tIdx++ {
		rawWords := strings.Fields(turnsB[tIdx].Text)
		for wIdx, rw := range rawWords {
			norm := normalizeToken(rw)
			if norm != "" {
				wordsB = append(wordsB, wordRef{
					TurnIndex: tIdx,
					WordIndex: wIdx,
					Raw:       rw,
					Norm:      norm,
				})
			}
		}
	}

	// Find the longest common sub-sequence matching a suffix of wordsA with a prefix of wordsB
	bestMatchLen := 0
	bestMatchBEndTurn := -1
	bestMatchBEndWord := -1

	// Minimum overlap threshold: 3 words
	minMatch := 3

	for startB := 0; startB < len(wordsB) && startB < 15; startB++ {
		for startA := 0; startA < len(wordsA); startA++ {
			matchLen := 0
			for startA+matchLen < len(wordsA) && startB+matchLen < len(wordsB) {
				if wordsA[startA+matchLen].Norm == wordsB[startB+matchLen].Norm {
					matchLen++
				} else {
					break
				}
			}

			// If match extends to near the end of wordsA, it's a valid seam overlap
			if matchLen >= minMatch && matchLen > bestMatchLen {
				bestMatchLen = matchLen
				endRef := wordsB[startB+matchLen-1]
				bestMatchBEndTurn = endRef.TurnIndex
				bestMatchBEndWord = endRef.WordIndex
			}
		}
	}

	// If no match >= 3, try 2 words if words are distinct
	if bestMatchLen < minMatch {
		for startB := 0; startB < len(wordsB) && startB < 8; startB++ {
			for startA := len(wordsA) - 6; startA >= 0 && startA < len(wordsA); startA++ {
				if startA+1 < len(wordsA) && startB+1 < len(wordsB) {
					if wordsA[startA].Norm == wordsB[startB].Norm &&
						wordsA[startA+1].Norm == wordsB[startB+1].Norm &&
						len(wordsA[startA].Norm) >= 4 {
						bestMatchLen = 2
						endRef := wordsB[startB+1]
						bestMatchBEndTurn = endRef.TurnIndex
						bestMatchBEndWord = endRef.WordIndex
						break
					}
				}
			}
			if bestMatchLen > 0 {
				break
			}
		}
	}

	// If a seam match was found, splice turnsB starting after the overlap point
	if bestMatchLen >= 2 && bestMatchBEndTurn >= 0 {
		var remainderTurns []speakerTurn

		for tIdx := bestMatchBEndTurn; tIdx < len(turnsB); tIdx++ {
			rawWords := strings.Fields(turnsB[tIdx].Text)
			if tIdx == bestMatchBEndTurn {
				if bestMatchBEndWord+1 < len(rawWords) {
					remWords := rawWords[bestMatchBEndWord+1:]
					remText := strings.TrimSpace(strings.Join(remWords, " "))
					if remText != "" {
						remainderTurns = append(remainderTurns, speakerTurn{
							Speaker: turnsB[tIdx].Speaker,
							Text:    remText,
						})
					}
				}
			} else {
				remainderTurns = append(remainderTurns, turnsB[tIdx])
			}
		}

		if len(remainderTurns) == 0 {
			return formatTurns(turnsA)
		}

		// Check if first turn of remainder can merge into last turn of turnsA
		lastA := &turnsA[len(turnsA)-1]
		firstRem := remainderTurns[0]

		if lastA.Speaker == firstRem.Speaker {
			lastA.Text = strings.TrimSpace(lastA.Text + " " + firstRem.Text)
			turnsA = append(turnsA, remainderTurns[1:]...)
		} else {
			turnsA = append(turnsA, remainderTurns...)
		}

		return formatTurns(turnsA)
	}

	// Fallback when no overlap was detected (e.g. cut occurred during complete silence):
	// Combine turns cleanly without duplication
	lastA := &turnsA[len(turnsA)-1]
	firstB := turnsB[0]
	if lastA.Speaker == firstB.Speaker {
		lastA.Text = strings.TrimSpace(lastA.Text + " " + firstB.Text)
		turnsA = append(turnsA, turnsB[1:]...)
	} else {
		turnsA = append(turnsA, turnsB...)
	}

	return formatTurns(turnsA)
}

// ResolveSeams resolves multiple chunk transcripts sequentially.
func ResolveSeams(chunkTranscripts []string) string {
	if len(chunkTranscripts) == 0 {
		return ""
	}
	current := chunkTranscripts[0]
	for i := 1; i < len(chunkTranscripts); i++ {
		current = ResolveTwoSeams(current, chunkTranscripts[i])
	}
	return current
}
