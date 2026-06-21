package main

import (
	"context"
	"fmt"
	"log"

	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

func main() {
	client, err := inference.NewAnthropicClient(inference.AnthropicClientConfig{
		APIKey:        "sk-ant-your-key",
		SpecificAIURL: "https://dev.specific.ai",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	resp, err := client.CreateMessage(
		context.Background(),
		"claude-sonnet-4-20250514",
		[]inference.AnthropicMessage{
			{Role: "user", Content: "Classify this: Delivery was terrible"},
		},
		"sentiment_analysis",
		"my_project",
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Content) > 0 {
		fmt.Println(resp.Content[0].Text)
	}
}
