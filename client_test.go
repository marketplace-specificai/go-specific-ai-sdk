package specificai

import (
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
	_, err := NewClient(WithTritonURL("http://triton:8000"), WithTrace(true))
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
	client, err := NewClient(WithTritonURL("http://triton:8000"))
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
		WithTritonURL("http://triton:8000"),
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
