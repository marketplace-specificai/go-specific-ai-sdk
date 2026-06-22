package specificai

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/marketplace-specificai/go-specific-ai-sdk/inference"
	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
	"github.com/marketplace-specificai/go-specific-ai-sdk/platform"
	"github.com/marketplace-specificai/go-specific-ai-sdk/tracing"
)

// gatewayRoot strips a trailing /api or /api/ suffix from the base URL so that
// public proxy paths (/public/triton, /public/api) resolve relative to the
// domain root. The Python SDK passes these as a separate "url" parameter; the
// Go SDK derives the gateway root from baseURL instead.
func gatewayRoot(baseURL string) string {
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/api") {
		u = strings.TrimSuffix(u, "/api")
	}
	return u
}

// Client is the unified entry point for the SpecificAI SDK.
//
// It supports two modes:
//   - Platform mode (base_url + api_key): manage tasks, datasets, trainings, and models.
//   - Inference mode (inference_url or base_url): run predictions against deployed models.
//
// Both modes can be active simultaneously.
type Client struct {
	// Platform managers (nil when base_url is not configured).
	Tasks     *platform.TaskManager
	Assets    *platform.AssetsManager
	Trainings *platform.TrainingManager
	Models    *platform.ModelManager

	platform  *platform.Client
	inference *inference.Client
	tracer    *tracing.Collector
	trace     bool
}

// NewClient creates a new SpecificAI client with the given options.
//
// At least one of WithBaseURL or WithInferenceURL must be provided (or set via
// SPECIFIC_AI_BASE_URL / SPECIFIC_AI_API_KEY environment variables).
func NewClient(opts ...Option) (*Client, error) {
	cfg := &clientConfig{
		apiKeyHeader: "Authorization",
		apiKeyPrefix: "Bearer",
		timeout:      30 * time.Second,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.baseURL == "" {
		cfg.baseURL = os.Getenv("SPECIFIC_AI_BASE_URL")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("SPECIFIC_AI_API_KEY")
	}

	c := &Client{trace: cfg.trace}

	if cfg.baseURL != "" {
		httpOpts := []httpclient.Option{
			httpclient.WithTimeout(cfg.timeout),
		}
		if cfg.apiKey != "" {
			httpOpts = append(httpOpts, httpclient.WithAPIKey(cfg.apiKey))
		}
		if cfg.apiKeyHeader != "Authorization" {
			httpOpts = append(httpOpts, httpclient.WithAPIKeyHeader(cfg.apiKeyHeader))
		}
		if cfg.apiKeyPrefix != "Bearer" {
			httpOpts = append(httpOpts, httpclient.WithAPIKeyPrefix(cfg.apiKeyPrefix))
		}
		if cfg.sessionCookie != "" {
			name := cfg.sessionCookieName
			if name == "" {
				name = "session"
			}
			httpOpts = append(httpOpts, httpclient.WithSessionCookie(name, cfg.sessionCookie))
		}

		httpClient, err := httpclient.New(cfg.baseURL, httpOpts...)
		if err != nil {
			return nil, fmt.Errorf("create HTTP client: %w", err)
		}

		c.platform = platform.NewClient(httpClient)
		c.Tasks = c.platform.Tasks
		c.Assets = c.platform.Assets
		c.Trainings = c.platform.Trainings
		c.Models = c.platform.Models
	}

	baseURL := strings.TrimRight(cfg.baseURL, "/")
	inferenceURL := strings.TrimRight(cfg.inferenceURL, "/")

	if baseURL != "" || inferenceURL != "" {
		c.inference = inference.NewClient(baseURL, inferenceURL)
	}

	if cfg.trace && baseURL == "" {
		return nil, fmt.Errorf("WithTrace requires WithBaseURL to be set")
	}

	// The tracer powers both auto-tracing (Create when trace is enabled) and
	// explicit logging via Log. It only needs the gateway root, so create it
	// whenever a base URL is available, independent of the trace flag.
	if baseURL != "" {
		c.tracer = tracing.New(gatewayRoot(baseURL))
	}

	if c.platform == nil && c.inference == nil {
		return nil, fmt.Errorf("provide either base_url or inference_url (or both)")
	}

	return c, nil
}

// Create runs inference on a deployed SpecificAI model.
//
// This requires the client to be initialized with WithInferenceURL or WithBaseURL
// for gateway-based inference.
func (c *Client) Create(ctx context.Context, message, taskName, projectName string) (inference.LMResponse, error) {
	if c.inference == nil {
		return nil, fmt.Errorf("inference not configured: initialize with WithInferenceURL or WithBaseURL")
	}
	if projectName == "" {
		projectName = "default"
	}

	start := time.Now()
	var (
		resp           inference.LMResponse
		rawLogits      []float64
		inferenceError *string
	)

	resp, rawLogits, err := c.inference.Infer(ctx, message, taskName, projectName)
	if err != nil {
		errStr := err.Error()
		inferenceError = &errStr
		if c.trace && c.tracer != nil {
			c.tracer.Collect(tracing.Record{
				ModelName:         "specific.ai",
				Prompt:            message,
				Response:          nil,
				UsecaseName:       taskName,
				UsecaseGroup:      projectName,
				Datasets:          []string{},
				ResponseTime:      time.Since(start).Seconds(),
				IsFromOptuneModel: true,
				InferenceError:    inferenceError,
				RawLogits:         rawLogits,
			})
		}
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	if c.trace && c.tracer != nil {
		c.tracer.Collect(tracing.Record{
			ModelName:         "specific.ai",
			Prompt:            message,
			Response:          resp,
			UsecaseName:       taskName,
			UsecaseGroup:      projectName,
			Datasets:          []string{},
			ResponseTime:      time.Since(start).Seconds(),
			IsFromOptuneModel: true,
			RawLogits:         rawLogits,
		})
	}

	return resp, nil
}

// Platform task type for summarization usecases. The inference TaskType enum
// models response types (generation == "GenerationResponse"), while the usecase
// task type is "Summarization"; send the latter so the usecase is created with
// the correct type.
const summarizationTaskType = "Summarization"

// LogOption configures optional fields of the Log* methods. Unset values use
// their defaults: isMultilabel is false and the model name is left null.
type LogOption func(*logConfig)

type logConfig struct {
	isMultilabel bool
	modelName    string
}

// WithMultilabel marks a classification task as multi-label. It has no effect on
// other task types.
func WithMultilabel() LogOption {
	return func(cfg *logConfig) { cfg.isMultilabel = true }
}

// WithModelName sets the name of the external model that produced the response.
// When omitted, the model name is left null.
func WithModelName(modelName string) LogOption {
	return func(cfg *logConfig) { cfg.modelName = modelName }
}

// log records an interaction with an external (non-SpecificAI) model into the
// platform. Records are added to an existing task or used to bootstrap a new one
// with the same taskName and projectName. A nil response logs only the example.
//
// Logging is fire-and-forget; call Close to flush pending records before exit.
// Requires the client to be initialized with WithBaseURL.
func (c *Client) log(example string, response any, taskName, projectName, taskType string, opts ...LogOption) error {
	if c.tracer == nil {
		return fmt.Errorf("logging not configured: initialize with WithBaseURL")
	}
	cfg := logConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if projectName == "" {
		projectName = "default"
	}

	c.tracer.Collect(tracing.Record{
		ModelName:         cfg.modelName,
		Prompt:            example,
		Response:          response,
		UsecaseName:       taskName,
		UsecaseGroup:      projectName,
		Datasets:          []string{},
		IsFromOptuneModel: false,
		TaskType:          taskType,
		IsMultilabel:      cfg.isMultilabel,
	})
	return nil
}

// LogClassification logs a classification interaction with an external model.
// Pass nil labels to log only the example (no response). Requires WithBaseURL.
//
// Optional settings are passed as functional options, e.g.
// WithMultilabel() and WithModelName("my-model"); isMultilabel defaults to
// false and the model name defaults to null when omitted.
func (c *Client) LogClassification(ctx context.Context, example string, labels []string, taskName, projectName string, opts ...LogOption) error {
	var response any
	if labels != nil {
		response = &inference.ClassificationResponse{Labels: labels}
	}
	return c.log(example, response, taskName, projectName, string(inference.TaskTypeClassification), opts...)
}

// LogSummarization logs a summarization (generation) interaction with an
// external model. Pass an empty summary to log only the example. Requires
// WithBaseURL. The model name defaults to null; set it with WithModelName.
func (c *Client) LogSummarization(ctx context.Context, example, summary, taskName, projectName string, opts ...LogOption) error {
	var response any
	if summary != "" {
		response = &inference.GenerationResponse{Response: summary}
	}
	return c.log(example, response, taskName, projectName, summarizationTaskType, opts...)
}

// LogEntityRecognition logs an entity recognition interaction with an external
// model. Pass nil entities to log only the example. Requires WithBaseURL. The
// model name defaults to null; set it with WithModelName.
func (c *Client) LogEntityRecognition(ctx context.Context, example string, entities []inference.Entity, taskName, projectName string, opts ...LogOption) error {
	var response any
	if entities != nil {
		response = &inference.EntityRecognitionResponse{Entities: entities}
	}
	return c.log(example, response, taskName, projectName, string(inference.TaskTypeEntityRecognition), opts...)
}

// Close flushes pending traces and releases resources.
// Always call Close when done with the client.
func (c *Client) Close() {
	if c.tracer != nil {
		c.tracer.Close()
	}
}
