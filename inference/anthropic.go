package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/marketplace-specificai/go-specific-ai-sdk/tracing"
)

// AnthropicMessage is a single message in an Anthropic Messages API request.
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicResponse is the response from the Anthropic Messages API.
type AnthropicResponse struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Model   string `json:"model"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage map[string]any `json:"usage"`
}

// AnthropicClientConfig configures the Anthropic wrapper.
type AnthropicClientConfig struct {
	APIKey                    string
	AnthropicURL              string
	SpecificAIURL             string
	UseSpecificAIInference    bool
	ParseSpecificAIResponse   ParseSpecificAIResponseFunc
}

// AnthropicClient wraps the Anthropic Messages API with tracing and optional SpecificAI inference.
type AnthropicClient struct {
	apiKey                 string
	anthropicURL           string
	specificAIURL          string
	useSpecificAIInference bool
	parseResponse          ParseSpecificAIResponseFunc
	inference              *Client
	tracer                 *tracing.Collector
	httpClient             *http.Client
}

// NewAnthropicClient creates a new Anthropic wrapper client.
func NewAnthropicClient(cfg AnthropicClientConfig) (*AnthropicClient, error) {
	if cfg.SpecificAIURL == "" {
		return nil, fmt.Errorf("specificAIURL is required")
	}
	cfg.SpecificAIURL = strings.TrimRight(cfg.SpecificAIURL, "/")

	if cfg.AnthropicURL == "" {
		cfg.AnthropicURL = "https://api.anthropic.com/v1/messages"
	}
	if cfg.ParseSpecificAIResponse == nil {
		cfg.ParseSpecificAIResponse = DefaultParseSpecificAIResponse
	}

	c := &AnthropicClient{
		apiKey:                 cfg.APIKey,
		anthropicURL:           cfg.AnthropicURL,
		specificAIURL:          cfg.SpecificAIURL,
		useSpecificAIInference: cfg.UseSpecificAIInference,
		parseResponse:          cfg.ParseSpecificAIResponse,
		tracer:                 tracing.New(cfg.SpecificAIURL),
		httpClient:             &http.Client{Timeout: 60 * time.Second},
	}
	if cfg.UseSpecificAIInference {
		c.inference = NewClient(cfg.SpecificAIURL, "")
	}
	return c, nil
}

// CreateMessage sends a messages request, optionally routing through SpecificAI inference.
func (c *AnthropicClient) CreateMessage(
	ctx context.Context,
	model string,
	messages []AnthropicMessage,
	taskName string,
	projectName string,
	extraParams map[string]any,
) (*AnthropicResponse, error) {
	if projectName == "" {
		projectName = "default"
	}

	prompt := extractAnthropicPrompt(messages)
	start := time.Now()

	var (
		result         *AnthropicResponse
		inferenceError *string
		isFromSpecific bool
		rawLogits      []float64
	)

	if c.useSpecificAIInference && c.inference != nil {
		isFromSpecific = true
		resp, logits, err := c.inference.Infer(ctx, prompt, taskName, projectName)
		if err != nil {
			errStr := err.Error()
			inferenceError = &errStr
			isFromSpecific = false
			log.Printf("specificai: SpecificAI inference failed, falling back to Anthropic: %v", err)
		} else {
			rawLogits = logits
			content := c.parseResponse(resp)
			result = &AnthropicResponse{
				ID:    fmt.Sprintf("response-%d", time.Now().Unix()),
				Type:  "message",
				Role:  "assistant",
				Model: "specific-ai",
				Content: []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}{
					{Type: "text", Text: content},
				},
			}
		}
	}

	if result == nil {
		var err error
		result, err = c.callAnthropic(ctx, model, messages, extraParams)
		if err != nil {
			return nil, err
		}
	}

	responseText := ""
	if len(result.Content) > 0 {
		responseText = result.Content[0].Text
	}

	c.tracer.Collect(tracing.Record{
		ModelName:         model,
		Prompt:            prompt,
		Response:          responseText,
		UsecaseName:       taskName,
		UsecaseGroup:      projectName,
		Datasets:          []string{},
		ResponseTime:      time.Since(start).Seconds(),
		IsFromOptuneModel: isFromSpecific,
		InferenceError:    inferenceError,
		RawLogits:         rawLogits,
	})

	return result, nil
}

// Close shuts down the tracing collector.
func (c *AnthropicClient) Close() {
	if c.tracer != nil {
		c.tracer.Close()
	}
}

func (c *AnthropicClient) callAnthropic(ctx context.Context, model string, messages []AnthropicMessage, extra map[string]any) (*AnthropicResponse, error) {
	payload := map[string]any{
		"model":    model,
		"messages": messages,
	}
	for k, v := range extra {
		payload[k] = v
	}
	if _, ok := payload["max_tokens"]; !ok {
		payload["max_tokens"] = 1024
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal anthropic request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.anthropicURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create anthropic request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read anthropic response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result AnthropicResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse anthropic response: %w", err)
	}
	return &result, nil
}

func extractAnthropicPrompt(messages []AnthropicMessage) string {
	var parts []string
	for _, m := range messages {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n")
}
