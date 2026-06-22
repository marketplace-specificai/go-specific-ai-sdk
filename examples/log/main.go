package main

import (
	"context"
	"log"

	specificai "github.com/marketplace-specificai/go-specific-ai-sdk"
	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

// Example of logging interactions with an external (non-SpecificAI) model.
//
// Use the task-specific Log* methods when you already have a prompt/response
// pair from any model outside SpecificAI (e.g. an in-house model or a
// third-party provider) and you just want to record it in the SpecificAI
// platform. Each method creates the usecase with the correct task type. The
// records are added to an existing Task or used to bootstrap a new one with the
// same taskName and projectName.
func main() {
	client, err := specificai.NewClient(
		specificai.WithBaseURL("https://dev.specific.ai"),
	)
	if err != nil {
		log.Fatal(err)
	}
	// Logging is fire-and-forget; Close flushes pending records before exit.
	defer client.Close()

	ctx := context.Background()
	projectName := "my_project" // Project (group) name in the platform

	// Classification: pass the predicted labels. Optional settings use functional
	// options: the task defaults to single-label (add WithMultilabel() for
	// multi-label) and the model name defaults to null (set it with WithModelName).
	if err := client.LogClassification(
		ctx,
		"Classify this review as Positive or Negative:\nThis movie was terrible!",
		[]string{"Negative"}, // labels; pass nil to log only the example
		"sentiment_analysis", // Task (usecase) name in the platform
		projectName,
		specificai.WithModelName("my-external-model"),
	); err != nil {
		log.Fatal(err)
	}

	// Summarization: pass the generated text. Model name omitted => left null.
	if err := client.LogSummarization(
		ctx,
		"Specific AI is an auto-distillation platform for LLMs...", // Replace with your own example
		"Specific AI auto-distills LLMs.",
		"doc_summary",
		projectName,
	); err != nil {
		log.Fatal(err)
	}

	// Entity recognition: pass the recognized entities.
	person := "PERSON"
	obama := "Barack Obama"
	start := 0
	if err := client.LogEntityRecognition(
		ctx,
		"Barack Obama was born in Hawaii.",
		[]inference.Entity{{Label: &person, Content: &obama, StartIndex: &start}},
		"entity_extraction",
		projectName,
		specificai.WithModelName("my-external-model"),
	); err != nil {
		log.Fatal(err)
	}

	// Example-only: pass nil labels to log just the prompt for later labeling.
	if err := client.LogClassification(
		ctx,
		"Classify this review:\nThe fit was fine but the quality was poor.",
		nil,
		"sentiment_analysis",
		projectName,
	); err != nil {
		log.Fatal(err)
	}

	log.Println("Logged external interactions to SpecificAI.")
}
