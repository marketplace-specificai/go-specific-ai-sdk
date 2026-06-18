package main

import (
	"context"
	"fmt"
	"log"

	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

func main() {
	client, err := inference.NewOpenAIClient(inference.OpenAIClientConfig{
		APIKey:        "sk-your-openai-key",
		SpecificAIURL: "https://dev.specific.ai",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	resp, err := client.CreateCompletion(
		context.Background(),
		"gpt-4o-mini",
		[]inference.OpenAIMessage{
			{Role: "user", Content: "Classify this: The product is great!"},
		},
		"sentiment_analysis",
		"my_project",
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) > 0 {
		fmt.Println(resp.Choices[0].Message.Content)
	}
}
