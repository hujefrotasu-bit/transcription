# Transcription & Topic Extraction Service

An audio transcription, speaker diarization, and topic extraction backend built with Go, PostgreSQL, and Google Gemini APIs.

## Features

- **Audio Transcription & Diarization**: Transcribes uploaded audio files (`.ogg`, `.wav`, `.mp3`, etc.) using Gemini `gemini-3.5-transcribe` Interactions API with speaker diarization (`Speaker 1:`, `Speaker 2:`).
- **Topic Extraction**: Extracts concise main topics and summaries from transcripts using Gemini `gemini-3.8-flash` Interactions API with structured JSON output.
- **Persistent Storage**: Stores conversations and topics in PostgreSQL.
- **REST API**:
  - `POST /api/conversations/audio`: Upload multipart audio to transcribe, diarize, extract topics, and save.
  - `POST /api/conversations`: Create conversation from raw text transcript.
  - `GET /api/conversations`: Retrieve all conversations with their topics.
  - `GET /api/conversations/:id`: Retrieve a single conversation and its topics by UUID.

## Prerequisites

- Go 1.22+
- PostgreSQL
- Google Gemini API Key

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_NAME=conversation_parser

   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_TRANSCRIPTION_MODEL=gemini-3.5-transcribe
   GEMINI_TOPIC_MODEL=gemini-3.8-flash
   ```

2. Run database migrations:
   Apply SQL files in `migrations/` to your PostgreSQL database.

3. Run the service:
   ```bash
   go run .
   ```
