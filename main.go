package main

import (
	"log"
	"net/http"

	"github.com/joho/godotenv"

	"transcript/internal/database"
	"transcript/internal/handlers"
	"transcript/internal/minutes"
	"transcript/internal/transcription"
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

	// Initialise the Gemini transcription service.
	svc, err := transcription.NewGeminiService()
	if err != nil {
		log.Fatal("Failed to initialise transcription service:", err)
	}
	log.Println("Transcription service ready (model:", svc.Model(), ")")

	// Initialise the Gemini Minutes of Meeting service.
	minutesSvc, err := minutes.NewGeminiService(db)
	if err != nil {
		log.Fatal("Failed to initialise meeting minutes service:", err)
	}
	log.Println("Meeting minutes service ready (model:", minutesSvc.Model(), ")")

	// Wire up handler with DB + transcription + meeting minutes services.
	conversationHandler := &handlers.ConversationHandler{
		DB:                   db,
		TranscriptionService: svc,
		MinutesService:       minutesSvc,
	}

	// Routes.
	http.HandleFunc("/api/conversations", conversationHandler.HandleConversations)
	http.HandleFunc("/api/conversations/", conversationHandler.GetConversationByID)
	http.HandleFunc("/api/conversations/audio", conversationHandler.UploadAudio)

	log.Println("Server running on http://localhost:8080")
	if err = http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
