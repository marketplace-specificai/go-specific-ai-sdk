package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	neturl "net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// publicPrefix is the path a deployed model answers on, before the workspace
// segment the platform appends for its own tenants.
const publicPrefix = "/public/triton"

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
	Name       string         `json:"name"`
	Parameters map[string]any `json:"parameters,omitempty"`
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
	apiKey       string
	http         *http.Client

	// Looked up once from the platform and kept for the life of the client; see
	// inferenceBase. The mutex is held across that lookup so concurrent Infer
	// calls make one request rather than one each.
	mu       sync.Mutex
	basePath string
}

// NewClient creates a new inference client.
// baseURL is the SpecificAI gateway URL.
// inferenceURL is an optional direct inference (Triton) URL; when set it takes precedence.
func NewClient(baseURL, inferenceURL string) *Client {
	return NewClientWithAPIKey(baseURL, inferenceURL, "")
}

// NewClientWithAPIKey is NewClient with a SpecificAI API key sent as a Bearer token,
// falling back to SPECIFIC_AI_API_KEY when apiKey is empty.
func NewClientWithAPIKey(baseURL, inferenceURL, apiKey string) *Client {
	if apiKey == "" {
		apiKey = os.Getenv("SPECIFIC_AI_API_KEY")
	}
	return &Client{
		baseURL:      strings.TrimRight(baseURL, "/"),
		inferenceURL: strings.TrimRight(inferenceURL, "/"),
		apiKey:       apiKey,
		http: &http.Client{
			Timeout: 30 * time.Second,
			// A gateway that terminates TLS redirects plain HTTP to HTTPS, and Go's
			// default client answers a 301 or 302 on a POST by retrying it as a
			// bodiless GET. Hand the redirect back so postFollowingRedirects can
			// repeat it as a POST, the same way internal/httpclient does for the
			// platform routes.
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

// Hops an API key is carried across before the request is given up on, for
// inference and for the discovery call alike.
const maxRedirects = 3

// mayReplay reports whether the POST, its body and its API key may be repeated to
// target.
//
// Replaying by hand is what keeps the method, and it also skips what net/http does
// for a redirect that matters here: it drops sensitive headers when the host
// changes. Without a check of its own this would hand a customer's key and their
// prompt to whatever a Location names.
//
// Same host only, and never a step down from HTTPS. That still allows the hop this
// exists for -- plain HTTP answered with the HTTPS URL of the same deployment --
// and a port change within one host, which is how a Triton addressed directly
// moves between its own listeners.
func mayReplay(current, target *neturl.URL) bool {
	if target.Hostname() != current.Hostname() {
		return false
	}
	return target.Scheme == current.Scheme || target.Scheme == "https"
}

// postFollowingRedirects sends body to url and follows redirects as POSTs. Nothing is
// rewritten, so an endpoint that does not redirect -- a Triton reached directly over
// plain HTTP, say -- behaves exactly as before.
func (c *Client) postFollowingRedirects(ctx context.Context, url string, body []byte) (*http.Response, []byte, error) {
	for i := 0; i <= maxRedirects; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, nil, fmt.Errorf("create inference request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}

		resp, err := c.http.Do(req)
		if err != nil {
			return nil, nil, fmt.Errorf("inference request failed: %w", err)
		}
		respBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, nil, fmt.Errorf("read inference response: %w", readErr)
		}

		// 303 is left out on purpose: it means "GET the other thing", so repeating
		// the POST would be wrong.
		switch resp.StatusCode {
		case http.StatusMovedPermanently, http.StatusFound,
			http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
		default:
			return resp, respBody, nil
		}
		location := resp.Header.Get("Location")
		if location == "" {
			return resp, respBody, nil
		}
		next, err := req.URL.Parse(location)
		if err != nil {
			return resp, respBody, nil
		}
		if !mayReplay(req.URL, next) {
			return nil, nil, fmt.Errorf(
				"%s redirected to %s, which this will not repeat the request to: a redirect off the "+
					"host that was asked, or from HTTPS down to HTTP, would send the API key and the "+
					"request body somewhere the platform did not choose", url, next)
		}
		url = next.String()
	}
	return nil, nil, fmt.Errorf("inference request was redirected too many times")
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
// the public proxy paths are resolved relative to the domain root. The Python
// SDK uses a separate "url" param for this; here we derive it.
func gatewayRoot(baseURL string) string {
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/api") {
		u = strings.TrimSuffix(u, "/api")
	}
	return u
}

// fetchBasePath asks the platform which path prefix this API key's models answer
// on, returning "" when the answer cannot be had -- an older backend, or a blip --
// so the caller stays on the un-scoped prefix and asks again next time.
//
// This uses a throwaway client rather than c.http because that one hands redirects
// back for postFollowingRedirects to repeat as POSTs; a GET is safe to follow, and
// a plain-HTTP base URL redirected to HTTPS has to be followed or there is no answer.
// Followed under the same destination rule, because Go's own policy keeps
// Authorization on any same-host hop -- including one back down to plain HTTP,
// which would put the API key on the wire the caller did not choose.
func (c *Client) fetchBasePath(ctx context.Context, root string) string {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, root+"/public/api/inference_path", nil)
	if err != nil {
		return ""
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	client := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) > maxRedirects {
				return fmt.Errorf("discovery was redirected more than %d times", maxRedirects)
			}
			if from := via[len(via)-1].URL; !mayReplay(from, req.URL) {
				return fmt.Errorf("%s redirected to %s, which this will not carry the API key to", from, req.URL)
			}
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var target struct {
		BasePath string `json:"base_path"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&target); err != nil {
		return ""
	}
	return target.BasePath
}

// inferenceBase returns the URL a model's /v2/models/... hangs off.
//
// On hosted SaaS it ends in the workspace that deployed the model, which is what
// keeps two tenants' identically named models apart. Customers are never shown
// their workspace id, so the API key is what reveals it. An inferenceURL the
// caller configured points at a Triton of their own, which this platform neither
// routes nor scopes, and is used as given.
func (c *Client) inferenceBase(ctx context.Context) string {
	if c.inferenceURL != "" {
		return c.inferenceURL
	}
	root := gatewayRoot(c.baseURL)
	c.mu.Lock()
	cached := c.basePath
	c.mu.Unlock()
	if cached != "" {
		return root + cached
	}
	// The lookup runs unlocked on purpose. A mutex does not watch a context, so
	// holding one across a network call would keep concurrent Infer calls waiting
	// past their own deadlines, and a lookup that fails caches nothing -- so each
	// of them would then spend its own timeout in turn before any reached the
	// fallback. Unlocked, a first burst may look the prefix up more than once,
	// which costs one cheap GET each and happens once per process.
	basePath := c.fetchBasePath(ctx, root)
	if basePath == "" {
		return root + publicPrefix
	}
	c.mu.Lock()
	c.basePath = basePath
	c.mu.Unlock()
	return root + basePath
}

func (c *Client) modelInference(ctx context.Context, modelName, inputText string) ([]float64, map[string]any, error) {
	url := fmt.Sprintf("%s/v2/models/%s/infer", c.inferenceBase(ctx), modelName)

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

	resp, respBody, err := c.postFollowingRedirects(ctx, url, body)
	if err != nil {
		return nil, nil, err
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
