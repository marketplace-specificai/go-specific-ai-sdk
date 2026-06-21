package main

import (
	"context"
	"fmt"
	"log"
	"time"

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

	// Configure model setup
	_, err = client.Tasks.SaveTeacherAndPrompt(ctx,
		"your_task_id",
		"gpt-4o-mini",
		"Classify the sentiment of the following text as positive, negative, or neutral.",
		platform.ModelProviderOpenAI,
		"",
	)
	if err != nil {
		log.Fatal(err)
	}

	// Start training
	resp, err := client.Trainings.Start(ctx, platform.StartParams{
		TaskID:        "your_task_id",
		BaseModelName: "bert-base-uncased",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Training started: distillation_event_id=%s\n", resp.DistillationEventID)

	// Poll training status
	for {
		job, err := client.Trainings.Get(ctx, resp.DistillationEventID)
		if err != nil {
			log.Fatal(err)
		}
		if job == nil {
			break
		}
		fmt.Printf("Status: %s\n", job.Status)
		if job.Status == "completed" || job.Status == "failed" {
			break
		}
		time.Sleep(10 * time.Second)
	}

	// Get metrics
	metrics, err := client.Models.GetMetrics(ctx, platform.GetMetricsParams{
		TaskID: "your_task_id",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Metrics: %v\n", metrics.AllMetrics)
}
