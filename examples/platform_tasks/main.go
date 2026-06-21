package main

import (
	"context"
	"fmt"
	"log"

	specificai "github.com/marketplace-specificai/go-specific-ai-sdk"
	"github.com/marketplace-specificai/go-specific-ai-sdk/platform"
)

func main() {
	client, err := specificai.NewClient(
		specificai.WithBaseURL("https://dev.specific.ai"),
		specificai.WithAPIKey("sk_your_key_here"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()

	// Create a task
	resp, err := client.Tasks.Create(ctx, "my_project", []platform.TaskCreate{
		{
			TaskName: "sentiment_analysis",
			TaskType: "ClassificationResponse",
		},
	}, "Classify customer sentiment")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Created task IDs: %v\n", resp.CreatedTaskIDs)

	// List all tasks
	groups, err := client.Tasks.List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, g := range groups {
		fmt.Printf("Project: %s\n", g.ProjectName)
		for _, task := range g.IterTasks() {
			fmt.Printf("  Task: %s (type: %s)\n", task.TaskName, task.TaskType)
		}
	}
}
