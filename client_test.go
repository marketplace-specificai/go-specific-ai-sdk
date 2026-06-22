package specificai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewClient_RequiresConfig(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")
	_, err := NewClient()
	if err == nil {
		t.Fatal("expected error when no config provided")
	}
}

func TestNewClient_TraceRequiresBaseURL(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")
	_, err := NewClient(WithInferenceURL("http://triton:8000"), WithTrace(true))
	if err == nil {
		t.Fatal("expected error when trace=true without base URL")
	}
}

func TestNewClient_PlatformMode(t *testing.T) {
	client, err := NewClient(
		WithBaseURL("http://localhost:8000"),
		WithAPIKey("sk_test"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if client.Tasks == nil {
		t.Fatal("expected Tasks manager to be initialized")
	}
	if client.Assets == nil {
		t.Fatal("expected Assets manager to be initialized")
	}
	if client.Trainings == nil {
		t.Fatal("expected Trainings manager to be initialized")
	}
	if client.Models == nil {
		t.Fatal("expected Models manager to be initialized")
	}
}

func TestNewClient_InferenceMode(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")
	client, err := NewClient(WithInferenceURL("http://triton:8000"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if client.inference == nil {
		t.Fatal("expected inference client to be initialized")
	}
	if client.Tasks != nil {
		t.Fatal("expected Tasks to be nil in inference-only mode")
	}
}

func TestNewClient_CombinedMode(t *testing.T) {
	client, err := NewClient(
		WithBaseURL("http://localhost:8000"),
		WithAPIKey("sk_test"),
		WithInferenceURL("http://triton:8000"),
		WithTrace(true),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if client.Tasks == nil {
		t.Fatal("expected Tasks manager in combined mode")
	}
	if client.inference == nil {
		t.Fatal("expected inference client in combined mode")
	}
	if client.tracer == nil {
		t.Fatal("expected tracer in combined mode")
	}
}

func TestNewClient_TracerCreatedWithoutTrace(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")
	// Log must work without WithTrace, so the tracer is created whenever a
	// base URL is configured.
	client, err := NewClient(WithBaseURL("http://localhost:8000"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if client.tracer == nil {
		t.Fatal("expected tracer to be initialized when base URL is set, even without trace")
	}
	if client.trace {
		t.Fatal("expected auto-trace to remain disabled when WithTrace was not passed")
	}
}

func TestLog_RequiresBaseURL(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")
	client, err := NewClient(WithInferenceURL("http://triton:8000"))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if err := client.LogClassification(context.Background(), "p", []string{"x"}, "task", "proj"); err == nil {
		t.Fatal("expected error when logging without a base URL")
	}
}

func TestLogClassification_PostsExternalRecord(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/public/api/collect_raw_data" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&body)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	client, err := NewClient(WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if err := client.LogClassification(
		context.Background(),
		"the prompt",
		[]string{"Negative"},
		"my_task",
		"my_project",
		WithMultilabel(),
		WithModelName("my-external-model"),
	); err != nil {
		t.Fatal(err)
	}
	// Close flushes pending records synchronously.
	client.Close()

	if body["is_from_optune_model"] != false {
		t.Fatalf("expected is_from_optune_model=false, got %v", body["is_from_optune_model"])
	}
	if body["modelname"] != "my-external-model" {
		t.Fatalf("expected modelname=my-external-model, got %v", body["modelname"])
	}
	if body["prompt"] != "the prompt" {
		t.Fatalf("expected prompt='the prompt', got %v", body["prompt"])
	}
	if body["usecase_name"] != "my_task" {
		t.Fatalf("expected usecase_name=my_task, got %v", body["usecase_name"])
	}
	if body["usecase_group"] != "my_project" {
		t.Fatalf("expected usecase_group=my_project, got %v", body["usecase_group"])
	}
	if body["task_type"] != "ClassificationResponse" {
		t.Fatalf("expected task_type=ClassificationResponse, got %v", body["task_type"])
	}
	if body["is_multilabel"] != true {
		t.Fatalf("expected is_multilabel=true, got %v", body["is_multilabel"])
	}
	resp, ok := body["response"].(map[string]any)
	if !ok {
		t.Fatalf("expected structured response object, got %T", body["response"])
	}
	labels, ok := resp["labels"].([]any)
	if !ok || len(labels) != 1 || labels[0] != "Negative" {
		t.Fatalf("expected labels=[Negative], got %v", resp["labels"])
	}
}

func TestLogClassification_ExampleOnly(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	client, err := NewClient(WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	// nil labels => example-only log (no response). Options omitted => defaults.
	if err := client.LogClassification(context.Background(), "p", nil, "my_task", ""); err != nil {
		t.Fatal(err)
	}
	client.Close()

	if got, exists := body["response"]; exists && got != nil {
		t.Fatalf("expected nil response for example-only log, got %v", got)
	}
}

func TestLogSummarization_PostsGenerationResponse(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	client, err := NewClient(WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if err := client.LogSummarization(context.Background(), "p", "a short summary", "my_task", ""); err != nil {
		t.Fatal(err)
	}
	client.Close()

	if body["task_type"] != "Summarization" {
		t.Fatalf("expected task_type=Summarization, got %v", body["task_type"])
	}
	resp, ok := body["response"].(map[string]any)
	if !ok || resp["response"] != "a short summary" {
		t.Fatalf("expected generation response, got %v", body["response"])
	}
}

func TestLogClassification_DefaultsModelAndProject(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "")
	t.Setenv("SPECIFIC_AI_API_KEY", "")

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	client, err := NewClient(WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if err := client.LogClassification(context.Background(), "p", []string{"x"}, "my_task", ""); err != nil {
		t.Fatal(err)
	}
	client.Close()

	if body["usecase_group"] != "default" {
		t.Fatalf("expected usecase_group=default, got %v", body["usecase_group"])
	}
	// modelName was omitted, so it must be left null (field omitted from JSON).
	if got, exists := body["modelname"]; exists && got != nil {
		t.Fatalf("expected modelname to be null/omitted by default, got %v", got)
	}
	// isMultilabel was omitted, so it must default to false.
	if body["is_multilabel"] != false {
		t.Fatalf("expected is_multilabel=false by default, got %v", body["is_multilabel"])
	}
}

func TestNewClient_EnvFallback(t *testing.T) {
	t.Setenv("SPECIFIC_AI_BASE_URL", "http://env-host:8000")
	t.Setenv("SPECIFIC_AI_API_KEY", "sk_env_key")

	client, err := NewClient()
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if client.Tasks == nil {
		t.Fatal("expected platform mode from env vars")
	}
}
