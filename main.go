package main

import (
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"

	"transcript/benchmark"
	"transcript/database"
	"transcript/routes"
	"transcript/transcription"
	"transcript/verification"
)

func main() {
	// Load environment variables.
	if err := godotenv.Load(); err != nil {
		log.Println("Note: .env file not loaded:", err)
	}

	// Connect to PostgreSQL.
	db, err := database.Connect()
	if err != nil {
		log.Fatal("Failed to connect to PostgreSQL:", err)
	}
	defer db.Close()
	log.Println("Connected to PostgreSQL!")

	// 1. Initialise the Gemini audio transcription service.
	audioSvc, err := transcription.NewGeminiTranscriptionService()
	if err != nil {
		log.Fatal("Failed to initialise audio transcription service:", err)
	}
	log.Println("Transcription service ready (model:", audioSvc.Model(), ")")

	// 2. Initialise the Gemini Minutes of Meeting service.
	momSvc, err := transcription.NewGeminiMeetingMinutesService(db)
	if err != nil {
		log.Fatal("Failed to initialise meeting minutes service:", err)
	}
	log.Println("Meeting minutes service ready (model:", momSvc.Model(), ")")

	// 3. Initialise Claude Fable verification service.
	fableKey := os.Getenv("FABLE_API_KEY")
	fableBaseURL := os.Getenv("FABLE_BASE_URL")
	fableModel := os.Getenv("FABLE_MODEL")
	fableClient := verification.NewFableClient(fableKey, fableBaseURL, fableModel)
	verifier := verification.NewVerifier(fableClient)
	verifHandler := verification.NewHandler(db, verifier)
	log.Println("Fable verification service ready (model:", fableClient.Model, ")")

	// 4. Wire up conversation handler with DB + transcription + meeting minutes + verifier services.
	conversationHandler := &transcription.ConversationHandler{
		DB:                   db,
		TranscriptionService: audioSvc,
		MinutesService:       momSvc,
		Verifier:             verifier,
	}

	// 5. Wire up bulk benchmarking handler.
	benchHandler := benchmark.NewHandler(audioSvc, momSvc, verifier)

	// 6. Initialise HTTP API routes using dedicated ServeMux.
	router := routes.NewRouter(conversationHandler, verifHandler, benchHandler)

	log.Println("Server running on http://localhost:8080")
	if err = http.ListenAndServe(":8080", router); err != nil {
		log.Fatal(err)
	}
}
