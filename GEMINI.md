# Backend Rules: Transcript AI Project

## CRITICAL RULE: Mandatory Live LLM Execution (Zero Mocking / No Pre-Made Data)

**NEVER run benchmarks, audio transcription, Minutes of Meeting (MoM) generation, or verification audits with mocked, hardcoded, synthetic, or programmatic rubric fallback data.**

Every operation across the application, database, and benchmark suite MUST use live LLM calls:
1. **Transcription**: Must run live on `gemini-3.5-transcribe` via Gemini Interactions API.
2. **MoM Extraction**: Must run live on `gemini-3.5-flash-lite` through the structured extraction prompt.
3. **Verification Audit**: Must run live on `claude-fable-5.1` via Fable / CodeCraft API.
   - **PERMANENTLY REMOVED**: The programmatic `fast` rubric and any manual score calculation have been completely removed from the codebase.
   - If Claude Fable fails, times out, or runs out of balance (`insufficient_quota`), the audit MUST fail honestly (score 0, status FAIL) and show the exact error. NEVER substitute programmatic checks or synthetic scores.
4. **Benchmark Suite**: Must benchmark all files end-to-end with real live LLM execution, tracking actual latency, actual token usage, and actual model-evaluated scores.

### Protocol for API Errors & Rate Limits
If an API rate limit, quota exhaustion (e.g. CodeCraft insufficient balance or Gemini daily limit), or service error occurs:
- **DO NOT** fake data or scores.
- **DO NOT** fall back to programmatic score generation.
- **DO NOT** seed pre-calculated or dataset values into the database or benchmark reports.
- **HALT IMMEDIATELY** and inform the user of the exact error, or request an updated API key / balance top-up.

---

## CRITICAL RULE: Absolute Output Quality Invariance (Quality Over Speed)

**Output accuracy, transcription fidelity, speaker attribution precision, and comprehensive audit-grade extraction ALWAYS take precedence over latency or execution speed. Zero compromises on quality are permitted under any circumstance.**

1. **Strict Prohibition on Quality-Degrading Speed Hacks:**
   - **NEVER** propose, recommend, or implement "optimizations" that compromise output depth, drop model stages, approximate diarization, or degrade reasoning quality to shave off seconds.
   - **NEVER** replace acoustic speech-to-text with single-pass multimodal approximations that drop dialogue, hallucinate speaker turns, or paraphrase real-world meetings.
   - **NEVER** bypass or eliminate the dedicated speaker resolution pass or the complete 12-section Minutes of Meeting extraction rubric.

2. **Mandatory Preservation of Acoustic Diarization Payloads:**
   - `gemini-3.5-transcribe` requires `TimestampGranularities: []string{"word"}` to emit word-level speaker metadata annotations in Google's Interactions API. **NEVER** disable, remove, or alter this configuration; doing so strips speaker labels and corrupts acoustic diarization.
   - `DiarizationMode: "speaker"` and `Type: "verbatim"` are strictly mandatory.

3. **Zero Hardcoding & Zero Test-Dataset Leakage:**
   - All prompt instructions, linguistic rules, and backend algorithms MUST remain 100% generalized, abstract, and linguistic-based (e.g., using `[Name]` placeholders).
   - **NEVER** hardcode sample/test person names (e.g., "Paul"), verbatim dialogue snippets, or file-specific assumptions in prompts, rubrics, or codebase.

4. **Permissible Optimizations ONLY:**
   - Optimizations are ONLY allowed if they have **0.0% effect on output quality and 100% identical outputs**:
     * HTTP/2 connection pooling (`MaxIdleConnsPerHost: 32`).
     * Package-level precompiled regexes to reduce CPU cycles.
     * UI streaming / progressive rendering (displaying verified transcript early while MoM completes in background).
     * Cloud infrastructure proximity to reduce international network transit latency.
