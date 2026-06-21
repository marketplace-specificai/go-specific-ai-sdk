# SpecificAI Go SDK

Go client library for the [SpecificAI](https://specific.ai) platform. Manage tasks, datasets, trainings, and models programmatically, run inference against deployed models, and collect traces for continuous improvement.

## Installation

```bash
go get github.com/marketplace-specificai/go-specific-ai-sdk@latest
```

## Quick Start

### Environment Variables

The SDK reads these environment variables as defaults:

| Variable | Purpose |
|---|---|
| `SPECIFIC_AI_BASE_URL` | Platform backend URL (e.g. `https://platform.specific.ai`) |
| `SPECIFIC_AI_API_KEY` | API key (`sk_...`) |

### Platform API

```go
package main

import (
    "context"
    "fmt"
    "log"

    specificai "github.com/marketplace-specificai/go-specific-ai-sdk"
    "github.com/marketplace-specificai/go-specific-ai-sdk/platform"
)

func main() {
    client, err := specificai.NewClient(
        specificai.WithBaseURL("https://platform.specific.ai"),
        specificai.WithAPIKey("sk_your_key_here"),
    )
    if err != nil { log.Fatal(err) }
    defer client.Close()

    ctx := context.Background()

    // List tasks
    groups, _ := client.Tasks.List(ctx)
    for _, g := range groups {
        for _, t := range g.IterTasks() {
            fmt.Printf("%s: %s\n", t.TaskName, t.TaskType)
        }
    }

    // Create a task
    resp, _ := client.Tasks.Create(ctx, "my_project", []platform.TaskCreate{
        {TaskName: "sentiment", TaskType: "ClassificationResponse"},
    }, "Classify sentiment")
    fmt.Println("Created:", resp.CreatedTaskIDs)
}
```

### Inference with Tracing

```go
client, err := specificai.NewClient(
    specificai.WithBaseURL("https://platform.specific.ai"),
    specificai.WithTrace(true),
    specificai.WithInferenceURL("http://triton:8000"),
)
if err != nil { log.Fatal(err) }
defer client.Close()

resp, err := client.Create(ctx, "Great product!", "sentiment", "my_project")
// resp is an inference.LMResponse (ClassificationResponse, GenerationResponse, or EntityRecognitionResponse)
```

### Logging external model interactions

Use the task-specific `Log*` methods to record a prompt/response pair from an
external (non-SpecificAI) model into the platform, so the usecase is created with
the correct task type. The record is added to an existing Task or used to
bootstrap a new one with the same `taskName` and `projectName`. Logging requires
`WithBaseURL` and is fire-and-forget, so call `Close` to flush pending records
before exit. Pass nil labels / nil entities / an empty summary to log only the
example (no response).

Optional settings are passed as functional options: `WithMultilabel()` marks a
classification task as multi-label (defaults to single-label) and
`WithModelName("...")` sets the external model name (left null when omitted).

```go
client, err := specificai.NewClient(specificai.WithBaseURL("https://platform.specific.ai"))
if err != nil { log.Fatal(err) }
defer client.Close()

// Classification (multi-label and model name are optional):
err = client.LogClassification(ctx, "prompt sent to your model", []string{"Negative"},
    "sentiment", "my_project", specificai.WithMultilabel(), specificai.WithModelName("my-external-model"))

// Summarization (model name omitted => left null):
err = client.LogSummarization(ctx, "prompt", "a short summary", "doc_summary", "my_project")

// Entity recognition:
err = client.LogEntityRecognition(ctx, "prompt", []inference.Entity{ /* ... */ },
    "entity_extraction", "my_project", specificai.WithModelName("my-external-model"))
```

### OpenAI Wrapper

Wraps OpenAI Chat Completions with automatic tracing. Optionally routes through SpecificAI's optimized models with fallback to OpenAI.

```go
import "github.com/marketplace-specificai/go-specific-ai-sdk/inference"

client, _ := inference.NewOpenAIClient(inference.OpenAIClientConfig{
    APIKey:        "sk-your-openai-key",
    SpecificAIURL: "https://platform.specific.ai",
})
defer client.Close()

resp, _ := client.CreateCompletion(ctx, "gpt-4o-mini",
    []inference.OpenAIMessage{{Role: "user", Content: "Classify: Great product!"}},
    "sentiment", "my_project", nil,
)
fmt.Println(resp.Choices[0].Message.Content)
```

### Anthropic Wrapper

```go
client, _ := inference.NewAnthropicClient(inference.AnthropicClientConfig{
    APIKey:        "sk-ant-your-key",
    SpecificAIURL: "https://platform.specific.ai",
})
defer client.Close()

resp, _ := client.CreateMessage(ctx, "claude-sonnet-4-20250514",
    []inference.AnthropicMessage{{Role: "user", Content: "Classify: Terrible service"}},
    "sentiment", "my_project", nil,
)
fmt.Println(resp.Content[0].Text)
```

## Configuration Options

All options use the functional options pattern (`WithXxx`):

| Option | Default | Description |
|---|---|---|
| `WithBaseURL(url)` | `$SPECIFIC_AI_BASE_URL` | Backend URL |
| `WithAPIKey(key)` | `$SPECIFIC_AI_API_KEY` | Bearer API key |
| `WithTrace(bool)` | `false` | Enable trace collection (requires base URL) |
| `WithInferenceURL(url)` | — | Direct inference (Triton) URL |
| `WithTimeout(duration)` | `30s` | HTTP request timeout |
| `WithAPIKeyHeader(name)` | `"Authorization"` | Auth header name |
| `WithAPIKeyPrefix(prefix)` | `"Bearer"` | Auth header prefix |
| `WithSessionCookie(name, value)` | — | Cookie-based auth |

## Platform Managers

### Tasks (`client.Tasks`)

| Method | Description |
|---|---|
| `List(ctx)` | List all tasks grouped by project |
| `Get(ctx, taskID)` | Fetch a single task |
| `Create(ctx, projectName, tasks, projectGoal)` | Create tasks |
| `Edit(ctx, taskID, params)` | Update task fields |
| `SaveTeacherAndPrompt(ctx, ...)` | Set teacher model and prompt |
| `SetComparisonConfig(ctx, ...)` | Configure comparison mode |
| `Delete(ctx, taskID)` | Delete a task |

### Assets (`client.Assets`)

| Method | Description |
|---|---|
| `UploadDataset(ctx, params)` | Upload a local file as dataset/benchmark |
| `UploadHuggingFaceDataset(ctx, params)` | Import a HuggingFace dataset |
| `GetUploadStatus(ctx, statusID)` | Check upload processing status |
| `GetFileColumns(ctx, filePath)` | Infer columns from a local file |
| `GetFileLabels(ctx, ...)` | Infer labels from a file column |
| `DeleteDataset(ctx, ...)` | Delete a dataset/benchmark |
| `GetLowConfidenceSamples(ctx, ...)` | Fetch samples for re-annotation |

### Trainings (`client.Trainings`)

| Method | Description |
|---|---|
| `Start(ctx, params)` | Start a training run |
| `Stop(ctx, taskID)` | Stop the current training |
| `List(ctx)` | List all training jobs |
| `Get(ctx, distillationEventID)` | Get training by event ID |
| `GetLatest(ctx, taskID)` | Get the latest training for a task |

### Models (`client.Models`)

| Method | Description |
|---|---|
| `GetMetrics(ctx, params)` | Fetch evaluation metrics |
| `ListVersions(ctx, taskID)` | List model versions with metrics |
| `DeleteVersion(ctx, taskID, version)` | Delete a model version |
| `Deploy(ctx, ...)` | Deploy/undeploy a model |
| `Evaluate(ctx, taskID, version)` | Trigger re-evaluation |
| `GetBenchmarkPredictions(ctx, params)` | Fetch benchmark predictions |
| `DownloadModel(ctx, ...)` | Download a model artifact |

## Error Handling

The SDK returns typed errors for different HTTP status codes:

```go
import "errors"

_, err := client.Tasks.Get(ctx, "nonexistent")

var notFound *specificai.NotFoundError
if errors.As(err, &notFound) {
    fmt.Println("Task not found")
}

var authErr *specificai.AuthenticationError
if errors.As(err, &authErr) {
    fmt.Println("Invalid API key")
}
```

Error types: `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404), `RateLimitError` (429), `ValidationError` (400/422), `ServerError` (5xx), `NetworkError` (transport).

## Running Tests

```bash
cd go-sdk
go test -race ./...
```

## Examples

See the [examples/](examples/) directory for complete working examples:

- [Platform Tasks](examples/platform_tasks/) — Create and list tasks
- [Platform Training](examples/platform_training/) — Configure, train, and evaluate models
- [Direct Inference](examples/direct_inference/) — Run inference against deployed models
- [Log](examples/log/) — Log external (non-SpecificAI) model interactions into the platform
- [OpenAI Wrapper](examples/openai/) — OpenAI Chat Completions with tracing
- [Anthropic Wrapper](examples/anthropic/) — Anthropic Messages API with tracing

## License

See the repository root LICENSE file.
