# SpecificAI Go SDK — Installation Guide

## Prerequisites

- **Go 1.22+** — [Install Go](https://go.dev/doc/install)
- A SpecificAI account with a deployed model
- Your **SpecificAI service URL** and **API key** (from the platform Settings page)

## Installation

```bash
go get github.com/marketplace-specificai/go-specific-ai-sdk@latest
```

> The module path is `github.com/marketplace-specificai/go-specific-ai-sdk`. Imports use this
> path throughout your code.

## Quick Start

### 1. Create a new project

```bash
mkdir my-specificai-app && cd my-specificai-app
go mod init my-specificai-app
go get github.com/marketplace-specificai/go-specific-ai-sdk@latest
```

### 2. Create `main.go`

```go
package main

import (
    "context"
    "fmt"
    "log"

    specificai "github.com/marketplace-specificai/go-specific-ai-sdk"
    "github.com/marketplace-specificai/go-specific-ai-sdk/inference"
)

func main() {
    client, err := specificai.NewClient(
        specificai.WithBaseURL("https://your-instance.specific.ai/api"),
        specificai.WithAPIKey("your-api-key"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    resp, err := client.Create(
        context.Background(),
        "This product is amazing!",
        "your-task-name",
        "your-project-name",
    )
    if err != nil {
        log.Fatal(err)
    }

    switch r := resp.(type) {
    case *inference.ClassificationResponse:
        fmt.Printf("Labels: %v\n", r.Labels)
        fmt.Printf("Confidences: %v\n", r.Confidences)
    case *inference.EntityRecognitionResponse:
        for _, e := range r.Entities {
            fmt.Printf("Entity: %s (%s)\n", *e.Content, *e.Label)
        }
    case *inference.GenerationResponse:
        fmt.Printf("Response: %s\n", r.Response)
    }
}
```

### 3. Run

```bash
go run main.go
```

## Configuration

### Environment variables

Instead of passing options directly, you can set environment variables:

```bash
export SPECIFIC_AI_BASE_URL="https://your-instance.specific.ai/api"
export SPECIFIC_AI_API_KEY="your-api-key"
```

Then initialize without explicit options:

```go
client, err := specificai.NewClient()
```

### Client options

| Option | Env var | Description |
|--------|---------|-------------|
| `WithBaseURL(url)` | `SPECIFIC_AI_BASE_URL` | SpecificAI platform URL |
| `WithAPIKey(key)` | `SPECIFIC_AI_API_KEY` | API key for authentication |
| `WithTrace(true)` | — | Enable tracing to the platform |
| `WithInferenceURL(url)` | — | Direct inference (Triton) URL (bypasses gateway) |
| `WithTimeout(d)` | — | HTTP timeout (default: 30s) |

## SDK Modes

### Direct inference (deployed SpecificAI models)

Use `client.Create()` as shown above. The SDK routes requests through the
SpecificAI gateway to your deployed Triton models.

### OpenAI wrapper with SpecificAI tracing

```go
import "github.com/marketplace-specificai/go-specific-ai-sdk/inference"

client, _ := inference.NewOpenAIClient(inference.OpenAIClientConfig{
    APIKey:              "sk-...",
    SpecificAIURL:       "https://your-instance.specific.ai",
    UseSpecificInference: true,
})
defer client.Close()

resp, _ := client.CreateChatCompletion(ctx, "gpt-4o", messages, "task", "project", nil)
```

### Anthropic wrapper with SpecificAI tracing

```go
import "github.com/marketplace-specificai/go-specific-ai-sdk/inference"

client, _ := inference.NewAnthropicClient(inference.AnthropicClientConfig{
    APIKey:        "sk-ant-...",
    SpecificAIURL: "https://your-instance.specific.ai",
})
defer client.Close()

resp, _ := client.CreateMessage(ctx, "claude-sonnet-4-20250514", messages, "task", "project", nil)
```

### Platform automation

```go
client, _ := specificai.NewClient(
    specificai.WithBaseURL("https://your-instance.specific.ai/api"),
    specificai.WithAPIKey("your-api-key"),
)

tasks, _ := client.Tasks.List(ctx)
training, _ := client.Trainings.Start(ctx, taskID, config)
metrics, _ := client.Models.GetMetrics(ctx, taskID)
```

## Error handling

The SDK returns typed errors for common failure modes:

```go
import specificai "github.com/marketplace-specificai/go-specific-ai-sdk"

resp, err := client.Create(ctx, message, task, project)
if err != nil {
    var apiErr *specificai.APIError
    if errors.As(err, &apiErr) {
        fmt.Printf("API error %d: %s\n", apiErr.StatusCode, apiErr.Message)
    }
}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `405 Method Not Allowed` | Using gateway URL as `WithInferenceURL` | Remove `WithInferenceURL`; use `WithBaseURL` only |
| `404 Not Found` | Wrong model name | Check task/project names match the platform exactly |
| `401 Unauthorized` | Invalid or expired API key | Regenerate key in platform Settings |
| Connection refused | Wrong URL or service down | Verify the URL and that the service is running |

## Support

- Email: support@specific.ai
- Platform docs: Settings > Documentation in your SpecificAI instance
