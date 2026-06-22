package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"
)

var unsafeChars = regexp.MustCompile(`[^a-zA-Z0-9_\-.]`)

func sanitizeNamePart(part string) string {
	return unsafeChars.ReplaceAllString(part, "_")
}

// tritonRequest is the Triton inference protocol request payload.
type tritonRequest struct {
	Inputs  []tritonInput  `json:"inputs"`
	Outputs []tritonOutput `json:"outputs"`
}

type tritonInput struct {
	Name     string   `json:"name"`
	Shape    []int    `json:"shape"`
	Datatype string   `json:"datatype"`
	Data     []string `json:"data"`
}

type tritonOutput struct {
	Name       string            `json:"name"`
	Parameters map[string]any    `json:"parameters,omitempty"`
}

type tritonResponse struct {
	Outputs []struct {
		Name string `json:"name"`
		Data []any  `json:"data"`
	} `json:"outputs"`
}

// Client performs direct inference against a Triton endpoint.
type Client struct {
	baseURL      string
	inferenceURL string
	http         *http.Client
}

// NewClient creates a new inference client.
// baseURL is the SpecificAI gateway URL.
// inferenceURL is an optional direct inference (Triton) URL; when set it takes precedence.
func NewClient(baseURL, inferenceURL string) *Client {
	return &Client{
		baseURL:      strings.TrimRight(baseURL, "/"),
		inferenceURL: strings.TrimRight(inferenceURL, "/"),
		http:         &http.Client{Timeout: 30 * time.Second},
	}
}

// Infer runs inference and returns parsed responses plus raw logits.
func (c *Client) Infer(ctx context.Context, prompt, taskName, projectName string) (LMResponse, []float64, error) {
	modelName := fmt.Sprintf("model_%s_%s", sanitizeNamePart(projectName), sanitizeNamePart(taskName))
	rawLogits, modelResult, err := c.modelInference(ctx, modelName, prompt)
	if err != nil {
		return nil, nil, err
	}
	resp, err := parseModelResult(modelResult)
	if err != nil {
		return nil, rawLogits, err
	}
	return resp, rawLogits, nil
}

// gatewayRoot strips a trailing /api or /api/ suffix from the base URL so that
// the Triton proxy path (/public/triton) is resolved relative to the domain
// root. The Python SDK uses a separate "url" param for this; here we derive it.
func gatewayRoot(baseURL string) string {
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/api") {
		u = strings.TrimSuffix(u, "/api")
	}
	return u
}

func (c *Client) modelInference(ctx context.Context, modelName, inputText string) ([]float64, map[string]any, error) {
	tritonBase := c.inferenceURL
	if tritonBase == "" {
		tritonBase = gatewayRoot(c.baseURL) + "/public/triton"
	}
	url := fmt.Sprintf("%s/v2/models/%s/infer", tritonBase, modelName)

	payload := tritonRequest{
		Inputs: []tritonInput{{
			Name:     "input_text",
			Shape:    []int{1, 1},
			Datatype: "BYTES",
			Data:     []string{inputText},
		}},
		Outputs: []tritonOutput{
			{Name: "result"},
			{Name: "raw_logits", Parameters: map[string]any{"binary_data": false}},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal inference request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, nil, fmt.Errorf("create inference request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("inference request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("read inference response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("inference returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var tritonResp tritonResponse
	if err := json.Unmarshal(respBody, &tritonResp); err != nil {
		return nil, nil, fmt.Errorf("parse inference response: %w", err)
	}

	var rawLogits []float64
	var resultJSON string

	for _, out := range tritonResp.Outputs {
		switch out.Name {
		case "raw_logits":
			for _, v := range out.Data {
				if f, ok := v.(float64); ok {
					rawLogits = append(rawLogits, f)
				}
			}
		case "result":
			if len(out.Data) > 0 {
				if s, ok := out.Data[0].(string); ok {
					resultJSON = s
				}
			}
		}
	}

	if resultJSON == "" {
		return nil, nil, fmt.Errorf("no result output from inference")
	}

	var modelResult map[string]any
	if err := json.Unmarshal([]byte(resultJSON), &modelResult); err != nil {
		return nil, nil, fmt.Errorf("parse model result: %w", err)
	}

	return rawLogits, modelResult, nil
}

func parseModelResult(result map[string]any) (LMResponse, error) {
	taskType, _ := result["task_type"].(string)
	switch TaskType(taskType) {
	case TaskTypeClassification:
		return parseClassification(result)
	case TaskTypeEntityRecognition:
		return parseNER(result)
	case TaskTypeGeneration:
		return parseGeneration(result)
	default:
		return nil, fmt.Errorf("unknown task type: %s", taskType)
	}
}

func parseClassification(result map[string]any) (*ClassificationResponse, error) {
	confRaw, _ := result["confidences"].(map[string]any)
	confidences := make(map[string]float64, len(confRaw))
	for k, v := range confRaw {
		if f, ok := v.(float64); ok {
			confidences[k] = f
		}
	}

	labelsRaw, _ := result["labels"].([]any)
	labels := make([]string, 0, len(labelsRaw))
	for _, v := range labelsRaw {
		if s, ok := v.(string); ok {
			labels = append(labels, s)
		}
	}
	sort.Slice(labels, func(i, j int) bool {
		return confidences[labels[i]] > confidences[labels[j]]
	})

	var thresholds map[string]float64
	if thRaw, ok := result["thresholds"].(map[string]any); ok {
		thresholds = make(map[string]float64, len(thRaw))
		for k, v := range thRaw {
			if f, ok := v.(float64); ok {
				thresholds[k] = f
			}
		}
	}

	extraParams, _ := result["extra_params"].(map[string]any)

	return &ClassificationResponse{
		Labels:      labels,
		Confidences: confidences,
		Thresholds:  thresholds,
		ExtraParams: extraParams,
	}, nil
}

func parseNER(result map[string]any) (*EntityRecognitionResponse, error) {
	respRaw, _ := result["response"].(map[string]any)
	b, err := json.Marshal(respRaw)
	if err != nil {
		return nil, fmt.Errorf("marshal NER response: %w", err)
	}
	var resp EntityRecognitionResponse
	if err := json.Unmarshal(b, &resp); err != nil {
		return nil, fmt.Errorf("parse NER response: %w", err)
	}
	return &resp, nil
}

func parseGeneration(result map[string]any) (*GenerationResponse, error) {
	respRaw, _ := result["response"].(map[string]any)
	text, _ := respRaw["response"].(string)
	return &GenerationResponse{Response: text}, nil
}
