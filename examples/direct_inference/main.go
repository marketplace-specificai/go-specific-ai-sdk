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
		specificai.WithAPIKey("sk-your-api-key"),
		specificai.WithWorkspaceID("your-workspace-id"),
		specificai.WithTrace(true),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Gateway inference by model ID.
	resp, err := client.CreateByModelID(
		context.Background(),
		"The product quality is excellent and delivery was fast",
		"6890bd049414abe1d38d3bf7.v1", // modelID: {usecase_id}.v{version}
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
