//go:build integration

package specificai

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/optune-ai/optune/go-sdk/inference"
)

func requiredEnv(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

func integrationClient(t *testing.T) *Client {
	t.Helper()
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	apiKey := requiredEnv(t, "SPECIFIC_AI_API_KEY")

	client, err := NewClient(
		WithBaseURL(baseURL),
		WithAPIKey(apiKey),
		WithTimeout(30*time.Second),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return client
}

// --- Client initialization ---

func TestIntegration_ClientInit_BaseURLOnly(t *testing.T) {
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	client, err := NewClient(WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("NewClient with base URL only: %v", err)
	}
	defer client.Close()

	if client.Tasks == nil {
		t.Error("expected Tasks manager to be initialized")
	}
}

func TestIntegration_ClientInit_WithTrace(t *testing.T) {
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	client, err := NewClient(WithBaseURL(baseURL), WithTrace(true))
	if err != nil {
		t.Fatalf("NewClient with trace: %v", err)
	}
	defer client.Close()
}

func TestIntegration_ClientInit_InferenceURLOnly(t *testing.T) {
	inferenceURL := os.Getenv("SPECIFIC_AI_INFERENCE_URL")
	if inferenceURL == "" {
		t.Skip("skipping: SPECIFIC_AI_INFERENCE_URL not set")
	}
	client, err := NewClient(WithInferenceURL(inferenceURL))
	if err != nil {
		t.Fatalf("NewClient with inference URL: %v", err)
	}
	defer client.Close()
}

// --- Direct inference ---

func TestIntegration_Inference_Classification(t *testing.T) {
	client := integrationClient(t)
	defer client.Close()

	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.Create(ctx, "This is a test input for classification", taskName, projectName)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	cr, ok := resp.(*inference.ClassificationResponse)
	if !ok {
		t.Fatalf("expected ClassificationResponse, got %T", resp)
	}
	if len(cr.Labels) == 0 {
		t.Error("expected at least one label")
	}
	if len(cr.Confidences) == 0 {
		t.Error("expected at least one confidence score")
	}
	t.Logf("Labels: %v, Confidences: %v", cr.Labels, cr.Confidences)
}

func TestIntegration_Inference_WithTracing(t *testing.T) {
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	apiKey := requiredEnv(t, "SPECIFIC_AI_API_KEY")
	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	client, err := NewClient(
		WithBaseURL(baseURL),
		WithAPIKey(apiKey),
		WithTrace(true),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.Create(ctx, "Tracing integration test", taskName, projectName)
	if err != nil {
		t.Fatalf("Create with tracing: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	t.Logf("Response type: %T", resp)
}

// --- Platform: Tasks ---

func TestIntegration_Platform_ListTasks(t *testing.T) {
	client := integrationClient(t)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	tasks, err := client.Tasks.List(ctx)
	if err != nil {
		t.Fatalf("Tasks.List: %v", err)
	}
	t.Logf("Found %d tasks", len(tasks))
}

func TestIntegration_Platform_TaskCRUD(t *testing.T) {
	client := integrationClient(t)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	projectName := os.Getenv("INTEGRATION_PROJECT_NAME")
	if projectName == "" {
		projectName = "integration-test"
	}

	taskName := "go-sdk-integration-test"
	created, err := client.Tasks.Create(ctx, taskName, "classification", projectName, nil)
	if err != nil {
		t.Fatalf("Tasks.Create: %v", err)
	}
	t.Logf("Created task: %v", created)

	fetched, err := client.Tasks.Get(ctx, taskName, projectName)
	if err != nil {
		t.Fatalf("Tasks.Get: %v", err)
	}
	if fetched == nil {
		t.Fatal("expected non-nil task from Get")
	}

	err = client.Tasks.Delete(ctx, taskName, projectName)
	if err != nil {
		t.Fatalf("Tasks.Delete: %v", err)
	}
	t.Log("Task CRUD cycle completed successfully")
}

// --- Platform: Trainings ---

func TestIntegration_Platform_ListTrainings(t *testing.T) {
	client := integrationClient(t)
	defer client.Close()

	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	trainings, err := client.Trainings.List(ctx, taskName, projectName)
	if err != nil {
		t.Fatalf("Trainings.List: %v", err)
	}
	t.Logf("Found %d trainings", len(trainings))
}

// --- Platform: Models ---

func TestIntegration_Platform_GetModelMetrics(t *testing.T) {
	client := integrationClient(t)
	defer client.Close()

	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	metrics, err := client.Models.GetMetrics(ctx, taskName, projectName)
	if err != nil {
		t.Logf("Models.GetMetrics returned error (may be expected if no model deployed): %v", err)
		return
	}
	t.Logf("Model metrics: %+v", metrics)
}
