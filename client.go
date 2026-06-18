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
//   - Inference mode (triton_url or base_url): run predictions against deployed models.
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
// At least one of WithBaseURL or WithTritonURL must be provided (or set via
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
	tritonURL := strings.TrimRight(cfg.tritonURL, "/")

	if baseURL != "" || tritonURL != "" {
		c.inference = inference.NewClient(baseURL, tritonURL)
	}

	if cfg.trace {
		if baseURL == "" {
			return nil, fmt.Errorf("WithTrace requires WithBaseURL to be set")
		}
		c.tracer = tracing.New(gatewayRoot(baseURL))
	}

	if c.platform == nil && c.inference == nil {
		return nil, fmt.Errorf("provide either base_url or triton_url (or both)")
	}

	return c, nil
}

// Create runs inference on a deployed SpecificAI model.
//
// This requires the client to be initialized with WithTritonURL or WithBaseURL
// for gateway-based inference.
func (c *Client) Create(ctx context.Context, message, taskName, projectName string) (inference.LMResponse, error) {
	if c.inference == nil {
		return nil, fmt.Errorf("inference not configured: initialize with WithTritonURL or WithBaseURL")
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

// Close flushes pending traces and releases resources.
// Always call Close when done with the client.
func (c *Client) Close() {
	if c.tracer != nil {
		c.tracer.Close()
	}
}
