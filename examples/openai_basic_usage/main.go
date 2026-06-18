// Example of using the SpecificAI Go SDK for text classification.
//
// This example runs in "trace" mode:
//   - UseSpecificAIInference=false -> the request is sent to the OpenAI API.
//   - The prompt/response pair is traced to the SpecificAI platform (dev.specific.ai)
//     under the given task/project so it can be used to build a dataset and later
//     train/deploy a SpecificAI model.
//
// Task:    trec-mini2
// Project: dyarden
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

// TREC question-type classification (coarse labels).
const promptTemplate = `
You are a question-type classifier. Given a question, classify it into exactly one
of the following categories:
ABBR - abbreviation
DESC - description and abstract concepts
ENTY - entities
HUM  - human beings
LOC  - locations
NUM  - numeric values

Return only the category code (one of: ABBR, DESC, ENTY, HUM, LOC, NUM). No other text.
Question:
{{example}}
`

func main() {
	example := "What is the capital of France?"
	prompt := strings.ReplaceAll(promptTemplate, "{{example}}", example)

	// Initialize the OpenAI client with SpecificAI integration.
	client, err := inference.NewOpenAIClient(inference.OpenAIClientConfig{
		SpecificAIURL:          "https://dev.specific.ai",
		APIKey:                 os.Getenv("OPENAI_API_KEY"), // required: the request is served by OpenAI in trace mode
		UseSpecificAIInference: false,                       // false -> OpenAI inference + trace to SpecificAI
	})
	if err != nil {
		log.Fatal(err)
	}
	// Close flushes pending traces to the SpecificAI platform before exit.
	defer client.Close()

	// Sends a request through the SDK.
	// The request is served by OpenAI, and the prompt/response is traced to SpecificAI.
	resp, err := client.CreateCompletion(
		context.Background(),
		"gpt-4o-mini",
		[]inference.OpenAIMessage{
			{Role: "user", Content: prompt},
		},
		"trec-mini2", // task_name: the deployed Task name in the SpecificAI platform
		"dyarden",    // project_name: the deployed Project name in the SpecificAI platform
		map[string]any{"temperature": 0},
	)
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) > 0 {
		fmt.Printf("Classification result: %s\n", resp.Choices[0].Message.Content)
	}
}
