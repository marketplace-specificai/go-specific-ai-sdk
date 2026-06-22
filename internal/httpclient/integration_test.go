//go:build integration

package httpclient

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestIntegration_HTTPClient_AuthenticatedRequest(t *testing.T) {
	baseURL := os.Getenv("SPECIFIC_AI_BASE_URL")
	apiKey := os.Getenv("SPECIFIC_AI_API_KEY")
	if baseURL == "" || apiKey == "" {
		t.Skip("skipping: SPECIFIC_AI_BASE_URL or SPECIFIC_AI_API_KEY not set")
	}

	client, err := New(baseURL,
		WithAPIKey(apiKey),
		WithTimeout(15*time.Second),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	resp, err := client.Do(ctx, "GET", "/llm_usecases", nil)
	if err != nil {
		t.Fatalf("GET /llm_usecases: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	t.Logf("Authenticated GET /llm_usecases returned %d", resp.StatusCode)
}

func TestIntegration_HTTPClient_UnauthenticatedReject(t *testing.T) {
	baseURL := os.Getenv("SPECIFIC_AI_BASE_URL")
	if baseURL == "" {
		t.Skip("skipping: SPECIFIC_AI_BASE_URL not set")
	}

	client, err := New(baseURL,
		WithTimeout(15*time.Second),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	resp, err := client.Do(ctx, "GET", "/llm_usecases", nil)
	if err != nil {
		t.Logf("Request failed as expected: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		t.Fatal("expected non-200 for unauthenticated request")
	}
	t.Logf("Unauthenticated request correctly returned %d", resp.StatusCode)
}
