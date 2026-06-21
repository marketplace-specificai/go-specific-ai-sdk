// Package specificai provides a Go client for the SpecificAI platform.
//
// The SDK supports three modes of operation:
//
//   - Platform mode: manage tasks, datasets, trainings, and models via the SpecificAI REST API.
//   - Inference mode: run predictions against deployed SpecificAI/Triton models.
//   - Tracing mode: collect LLM call data for model improvement.
//
// Quick start (platform):
//
//	client, err := specificai.NewClient(
//	    specificai.WithBaseURL("https://platform.specific.ai"),
//	    specificai.WithAPIKey("sk_..."),
//	)
//	if err != nil { log.Fatal(err) }
//	defer client.Close()
//
//	tasks, err := client.Tasks.List(context.Background())
//
// Quick start (inference with tracing):
//
//	client, err := specificai.NewClient(
//	    specificai.WithBaseURL("https://platform.specific.ai"),
//	    specificai.WithTrace(true),
//	    specificai.WithInferenceURL("http://triton:8000"),
//	)
//	if err != nil { log.Fatal(err) }
//	defer client.Close()
//
//	resp, err := client.Create(context.Background(), "Classify this text", "my_task", "my_project")
package specificai
