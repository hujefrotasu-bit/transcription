package main

import (
	"log"
	"net/http"

	"github.com/joho/godotenv"

	"transcript/internal/database"
	"transcript/internal/handlers"
	"transcript/internal/topics"
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

	// Initialise the Gemini topic extraction service.
	topicSvc, err := topics.NewGeminiService(db)
	if err != nil {
		log.Fatal("Failed to initialise topic service:", err)
	}
	log.Println("Topic extraction service ready (model:", topicSvc.Model(), ")")

	// Wire up handler with DB + transcription + topic services.
	conversationHandler := &handlers.ConversationHandler{
		DB:                   db,
		TranscriptionService: svc,
		TopicService:         topicSvc,
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
