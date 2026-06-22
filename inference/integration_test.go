//go:build integration

package inference

import (
	"context"
	"os"
	"testing"
	"time"
)

func requiredEnv(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}
	return v
}

// --- Direct Triton inference ---

func TestIntegration_DirectInference_Classification(t *testing.T) {
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	client := NewClient(baseURL, "")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, logits, err := client.Infer(ctx, "Test classification input", taskName, projectName)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	cr, ok := resp.(*ClassificationResponse)
	if !ok {
		t.Fatalf("expected ClassificationResponse, got %T", resp)
	}
	if len(cr.Labels) == 0 {
		t.Error("expected at least one label")
	}
	t.Logf("Labels: %v, Logits: %d values", cr.Labels, len(logits))
}

func TestIntegration_DirectInference_InferenceURL(t *testing.T) {
	inferenceURL := os.Getenv("SPECIFIC_AI_INFERENCE_URL")
	if inferenceURL == "" {
		t.Skip("skipping: SPECIFIC_AI_INFERENCE_URL not set")
	}
	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	client := NewClient("", inferenceURL)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, _, err := client.Infer(ctx, "Test with direct Triton URL", taskName, projectName)
	if err != nil {
		t.Fatalf("Infer via Triton URL: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	t.Logf("Response type: %T", resp)
}

func TestIntegration_DirectInference_GatewayRootStripping(t *testing.T) {
	baseURL := requiredEnv(t, "SPECIFIC_AI_BASE_URL")
	taskName := requiredEnv(t, "INTEGRATION_TASK_NAME")
	projectName := requiredEnv(t, "INTEGRATION_PROJECT_NAME")

	client := NewClient(baseURL+"/api", "")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, _, err := client.Infer(ctx, "Test gatewayRoot stripping", taskName, projectName)
	if err != nil {
		t.Fatalf("Infer with /api suffix: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	t.Log("gatewayRoot stripping works correctly")
}

// --- OpenAI wrapper ---

func TestIntegration_OpenAI_CreateCompletion(t *testing.T) {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		t.Skip("skipping: OPENAI_API_KEY not set")
	}
	specificAIURL := os.Getenv("SPECIFIC_AI_BASE_URL")

	client, err := NewOpenAIClient(OpenAIClientConfig{
		APIKey:        openaiKey,
		SpecificAIURL: specificAIURL,
	})
	if err != nil {
		t.Fatalf("NewOpenAIClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := client.CreateChatCompletion(
		ctx,
		"gpt-4o-mini",
		[]OpenAIMessage{{Role: "user", Content: "Say hello in one word"}},
		"integration-test",
		"go-sdk-test",
		nil,
	)
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
	t.Logf("OpenAI response: %s", resp.Choices[0].Message.Content)
}

// --- Anthropic wrapper ---

func TestIntegration_Anthropic_CreateMessage(t *testing.T) {
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	if anthropicKey == "" {
		t.Skip("skipping: ANTHROPIC_API_KEY not set")
	}
	specificAIURL := os.Getenv("SPECIFIC_AI_BASE_URL")

	client, err := NewAnthropicClient(AnthropicClientConfig{
		APIKey:        anthropicKey,
		SpecificAIURL: specificAIURL,
	})
	if err != nil {
		t.Fatalf("NewAnthropicClient: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := client.CreateMessage(
		ctx,
		"claude-sonnet-4-20250514",
		[]AnthropicMessage{{Role: "user", Content: "Say hello in one word"}},
		"integration-test",
		"go-sdk-test",
		nil,
	)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected at least one content block")
	}
	t.Logf("Anthropic response: %s", resp.Content[0].Text)
}
