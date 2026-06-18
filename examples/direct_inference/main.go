package main

import (
	"context"
	"fmt"
	"log"

	specificai "github.com/marketplace-specificai/go-specific-ai-sdk"
	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

func main() {
	client, err := specificai.NewClient(
		specificai.WithBaseURL("https://dev.specific.ai"),
		specificai.WithTrace(true),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	resp, err := client.Create(
		context.Background(),
		"The product quality is excellent and delivery was fast",
		"sentiment_analysis",
		"my_project",
	)
	if err != nil {
		log.Fatal(err)
	}

	switch r := resp.(type) {
	case *inference.ClassificationResponse:
		fmt.Printf("Labels: %v\n", r.Labels)
		fmt.Printf("Confidences: %v\n", r.Confidences)
	case *inference.GenerationResponse:
		fmt.Printf("Response: %s\n", r.Response)
	case *inference.EntityRecognitionResponse:
		for _, e := range r.Entities {
			fmt.Printf("Entity: %s (%s)\n", *e.Content, *e.Label)
		}
	}
}
