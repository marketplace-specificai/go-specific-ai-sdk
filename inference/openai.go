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

// OpenAICompletionRequest is the request body for OpenAI Chat Completions.
type OpenAICompletionRequest struct {
	Model    string           `json:"model"`
	Messages []OpenAIMessage  `json:"messages"`
	Extra    map[string]any   `json:"-"`
}

// OpenAIMessage is a single message in a chat completion request.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAICompletionResponse is the response from OpenAI Chat Completions.
type OpenAICompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

// ParseSpecificAIResponseFunc converts a SpecificAI response into a string for the OpenAI format.
type ParseSpecificAIResponseFunc func(LMResponse) string

// DefaultParseSpecificAIResponse joins classification labels with ";".
func DefaultParseSpecificAIResponse(resp LMResponse) string {
	if cr, ok := resp.(*ClassificationResponse); ok {
		return strings.Join(cr.Labels, ";")
	}
	if gr, ok := resp.(*GenerationResponse); ok {
		return gr.Response
	}
	b, _ := json.Marshal(resp)
	return string(b)
}

// OpenAIClientConfig configures the OpenAI wrapper.
type OpenAIClientConfig struct {
	APIKey                    string
	OpenAIURL                 string
	SpecificAIURL             string
	UseSpecificAIInference    bool
	ParseSpecificAIResponse   ParseSpecificAIResponseFunc
}

// OpenAIClient wraps OpenAI Chat Completions with tracing and optional SpecificAI inference.
type OpenAIClient struct {
	apiKey                  string
	openAIURL               string
	specificAIURL           string
	useSpecificAIInference  bool
	parseResponse           ParseSpecificAIResponseFunc
	inference               *Client
	tracer                  *tracing.Collector
	httpClient              *http.Client
}

// NewOpenAIClient creates a new OpenAI wrapper client.
func NewOpenAIClient(cfg OpenAIClientConfig) (*OpenAIClient, error) {
	if cfg.SpecificAIURL == "" {
		return nil, fmt.Errorf("specificAIURL is required")
	}
	cfg.SpecificAIURL = strings.TrimRight(cfg.SpecificAIURL, "/")

	if cfg.OpenAIURL == "" {
		cfg.OpenAIURL = "https://api.openai.com/v1/chat/completions"
	}
	if cfg.ParseSpecificAIResponse == nil {
		cfg.ParseSpecificAIResponse = DefaultParseSpecificAIResponse
	}

	c := &OpenAIClient{
		apiKey:                 cfg.APIKey,
		openAIURL:              cfg.OpenAIURL,
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

// CreateCompletion sends a chat completion request, optionally routing through SpecificAI inference.
func (c *OpenAIClient) CreateCompletion(
	ctx context.Context,
	model string,
	messages []OpenAIMessage,
	taskName string,
	projectName string,
	extraParams map[string]any,
) (*OpenAICompletionResponse, error) {
	if projectName == "" {
		projectName = "default"
	}

	prompt := extractPrompt(messages)
	start := time.Now()

	var (
		result         *OpenAICompletionResponse
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
			log.Printf("specificai: SpecificAI inference failed, falling back to OpenAI: %v", err)
		} else {
			rawLogits = logits
			content := c.parseResponse(resp)
			result = &OpenAICompletionResponse{
				ID:    fmt.Sprintf("specificai-%d", time.Now().UnixMilli()),
				Model: "specific-ai",
				Choices: []struct {
					Index   int `json:"index"`
					Message struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"message"`
					FinishReason string `json:"finish_reason"`
				}{
					{
						Index:        0,
						Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{Role: "assistant", Content: content},
						FinishReason: "stop",
					},
				},
			}
		}
	}

	if result == nil {
		var err error
		result, err = c.callOpenAI(ctx, model, messages, extraParams)
		if err != nil {
			return nil, err
		}
	}

	responseText := ""
	if len(result.Choices) > 0 {
		responseText = result.Choices[0].Message.Content
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
func (c *OpenAIClient) Close() {
	if c.tracer != nil {
		c.tracer.Close()
	}
}

func (c *OpenAIClient) callOpenAI(ctx context.Context, model string, messages []OpenAIMessage, extra map[string]any) (*OpenAICompletionResponse, error) {
	payload := map[string]any{
		"model":    model,
		"messages": messages,
	}
	for k, v := range extra {
		payload[k] = v
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal openai request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.openAIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create openai request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read openai response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result OpenAICompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse openai response: %w", err)
	}
	return &result, nil
}

func extractPrompt(messages []OpenAIMessage) string {
	var parts []string
	for _, m := range messages {
		parts = append(parts, m.Content)
	}
	return strings.Join(parts, "\n")
}
