package main

import (
	"log"
	"net/http"

	"github.com/joho/godotenv"

	"transcript/database"
	"transcript/routes"
	"transcript/transcription"
)

func main() {
	// Load environment variables.
	if err := godotenv.Load(); err != nil {
		log.Fatal("Failed to load .env:", err)
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

	// 3. Wire up handler with DB + transcription + meeting minutes services.
	conversationHandler := &transcription.ConversationHandler{
		DB:                   db,
		TranscriptionService: audioSvc,
		MinutesService:       momSvc,
	}

	// 4. Initialise HTTP API routes using dedicated ServeMux.
	router := routes.NewRouter(conversationHandler)

	log.Println("Server running on http://localhost:8080")
	if err = http.ListenAndServe(":8080", router); err != nil {
		log.Fatal(err)
	}
}
