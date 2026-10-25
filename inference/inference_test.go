package inference

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	neturl "net/url"
	"strings"
	"testing"
)

func TestSanitizeNamePart(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello", "hello"},
		{"hello world", "hello_world"},
		{"test/path", "test_path"},
		{"a-b_c.d", "a-b_c.d"},
		{"special!@#$%chars", "special_____chars"},
	}
	for _, tt := range tests {
		got := sanitizeNamePart(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeNamePart(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestInfer_Classification(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"positive", "negative"},
			"confidences": map[string]float64{"positive": 0.9, "negative": 0.1},
			"thresholds":  map[string]float64{"positive": 0.5, "negative": 0.5},
		}
		resultJSON, _ := json.Marshal(result)

		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{0.9, 0.1}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient("", srv.URL)
	resp, logits, err := client.Infer(context.Background(), "test text", "my_task", "my_project")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cr, ok := resp.(*ClassificationResponse)
	if !ok {
		t.Fatalf("expected ClassificationResponse, got %T", resp)
	}
	if len(cr.Labels) != 2 {
		t.Fatalf("expected 2 labels, got %d", len(cr.Labels))
	}
	if cr.Labels[0] != "positive" {
		t.Fatalf("expected first label=positive (sorted by confidence), got %s", cr.Labels[0])
	}
	if cr.Confidences["positive"] != 0.9 {
		t.Fatalf("expected confidence=0.9, got %f", cr.Confidences["positive"])
	}
	if len(logits) != 2 {
		t.Fatalf("expected 2 logits, got %d", len(logits))
	}
}

func TestInfer_SendsAPIKey(t *testing.T) {
	t.Setenv("SPECIFIC_AI_API_KEY", "sk_env")
	var auth []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = append(auth, r.Header.Get("Authorization"))
		resultJSON, _ := json.Marshal(map[string]any{
			"task_type": "GenerationResponse",
			"response":  map[string]any{"response": "ok"},
		})
		json.NewEncoder(w).Encode(map[string]any{
			"outputs": []map[string]any{{"name": "result", "data": []string{string(resultJSON)}}},
		})
	}))
	defer srv.Close()

	for _, client := range []*Client{NewClientWithAPIKey("", srv.URL, "sk_explicit"), NewClient("", srv.URL)} {
		if _, _, err := client.Infer(context.Background(), "hi", "task", "project"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if len(auth) != 2 || auth[0] != "Bearer sk_explicit" || auth[1] != "Bearer sk_env" {
		t.Fatalf("expected explicit then env Bearer keys, got %v", auth)
	}
}

func TestInfer_Generation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		result := map[string]any{
			"task_type": "GenerationResponse",
			"response":  map[string]any{"response": "Generated text here"},
		}
		resultJSON, _ := json.Marshal(result)
		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient("", srv.URL)
	resp, _, err := client.Infer(context.Background(), "generate something", "gen_task", "project")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gr, ok := resp.(*GenerationResponse)
	if !ok {
		t.Fatalf("expected GenerationResponse, got %T", resp)
	}
	if gr.Response != "Generated text here" {
		t.Fatalf("expected 'Generated text here', got %q", gr.Response)
	}
}

func TestInfer_NER(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nerResp := map[string]any{
			"entities": []map[string]any{
				{"label": "PERSON", "content": "John", "start_index": 0},
			},
			"relations": []map[string]any{},
		}
		result := map[string]any{
			"task_type": "EntityRecognitionResponse",
			"response":  nerResp,
		}
		resultJSON, _ := json.Marshal(result)
		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient("", srv.URL)
	resp, _, err := client.Infer(context.Background(), "John went home", "ner_task", "project")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	nr, ok := resp.(*EntityRecognitionResponse)
	if !ok {
		t.Fatalf("expected EntityRecognitionResponse, got %T", resp)
	}
	if len(nr.Entities) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(nr.Entities))
	}
	if *nr.Entities[0].Label != "PERSON" {
		t.Fatalf("expected label=PERSON, got %s", *nr.Entities[0].Label)
	}
}

func TestInfer_UnknownTaskType(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		result := map[string]any{"task_type": "UnknownType"}
		resultJSON, _ := json.Marshal(result)
		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient("", srv.URL)
	_, _, err := client.Infer(context.Background(), "test", "task", "project")
	if err == nil {
		t.Fatal("expected error for unknown task type")
	}
}

func TestGatewayRoot(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"https://dev.specific.ai/api", "https://dev.specific.ai"},
		{"https://dev.specific.ai/api/", "https://dev.specific.ai"},
		{"https://dev.specific.ai", "https://dev.specific.ai"},
		{"https://dev.specific.ai/", "https://dev.specific.ai"},
		{"http://localhost:8080/api", "http://localhost:8080"},
		{"http://localhost:8080", "http://localhost:8080"},
		{"https://example.com/myapi", "https://example.com/myapi"},
	}
	for _, tt := range tests {
		got := gatewayRoot(tt.input)
		if got != tt.want {
			t.Errorf("gatewayRoot(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestInfer_BaseURLWithAPISuffix(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"a"},
			"confidences": map[string]float64{"a": 1.0},
		}
		resultJSON, _ := json.Marshal(result)
		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient(srv.URL+"/api", "")
	_, _, err := client.Infer(context.Background(), "test", "task", "proj")
	if err != nil {
		t.Fatal(err)
	}
	expected := "/public/triton/v2/models/model_proj_task/infer"
	if capturedPath != expected {
		t.Fatalf("expected path %s, got %s", expected, capturedPath)
	}
}

func TestInfer_AsksThePlatformWhichWorkspacePathToUse(t *testing.T) {
	// Two tenants can deploy the same model name, so the path carries the workspace.
	// Nobody tells the caller theirs: the API key is what the platform answers from.
	const workspace = "6ad761f4d2970b22cb1a569d"
	var capturedPath, lookupAuth string
	lookups := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/public/api/inference_path" {
			lookups++
			lookupAuth = r.Header.Get("Authorization")
			json.NewEncoder(w).Encode(map[string]string{
				"base_path":    "/public/triton/" + workspace,
				"workspace_id": workspace,
			})
			return
		}
		capturedPath = r.URL.Path
		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"a"},
			"confidences": map[string]float64{"a": 1.0},
		}
		resultJSON, _ := json.Marshal(result)
		json.NewEncoder(w).Encode(map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		})
	}))
	defer srv.Close()

	client := NewClientWithAPIKey(srv.URL, "", "sk-test")
	for range 2 {
		if _, _, err := client.Infer(context.Background(), "test", "task", "proj"); err != nil {
			t.Fatal(err)
		}
	}

	expected := "/public/triton/" + workspace + "/v2/models/model_proj_task/infer"
	if capturedPath != expected {
		t.Fatalf("expected path %s, got %s", expected, capturedPath)
	}
	if lookupAuth != "Bearer sk-test" {
		t.Fatalf("expected the API key on the lookup, got %q", lookupAuth)
	}
	if lookups != 1 {
		t.Fatalf("expected the answer to be kept, got %d lookups", lookups)
	}
}

func TestInfer_FallsBackToTheUnscopedPathWhenNobodyAnswers(t *testing.T) {
	// A self-hosted Triton, or a backend older than the workspace segment.
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/public/api/inference_path" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		capturedPath = r.URL.Path
		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"a"},
			"confidences": map[string]float64{"a": 1.0},
		}
		resultJSON, _ := json.Marshal(result)
		json.NewEncoder(w).Encode(map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "")
	if _, _, err := client.Infer(context.Background(), "test", "task", "proj"); err != nil {
		t.Fatal(err)
	}

	expected := "/public/triton/v2/models/model_proj_task/infer"
	if capturedPath != expected {
		t.Fatalf("expected path %s, got %s", expected, capturedPath)
	}
}

func TestModelName_Sanitization(t *testing.T) {
	var capturedURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"a"},
			"confidences": map[string]float64{"a": 1.0},
		}
		resultJSON, _ := json.Marshal(result)
		resp := map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
				{"name": "raw_logits", "data": []float64{}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewClient("", srv.URL)
	_, _, err := client.Infer(context.Background(), "test", "my task/v2", "project name")
	if err != nil {
		t.Fatal(err)
	}
	expected := "/v2/models/model_project_name_my_task_v2/infer"
	if capturedURL != expected {
		t.Fatalf("expected URL %s, got %s", expected, capturedURL)
	}
}

func TestInfer_RedirectOffTheHostIsRefused(t *testing.T) {
	// net/http drops sensitive headers across hosts; following redirects by hand
	// has to refuse the hop rather than hand the key and the prompt to it.
	var reached bool
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
	}))
	defer elsewhere.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Same port, different host: 127.0.0.1 and localhost are not the same name.
		w.Header().Set("Location", strings.Replace(elsewhere.URL, "127.0.0.1", "localhost", 1)+r.URL.Path)
		w.WriteHeader(http.StatusMovedPermanently)
	}))
	defer srv.Close()

	client := NewClientWithAPIKey("", srv.URL, "test-key")
	_, _, err := client.Infer(context.Background(), "test text", "my_task", "my_project")
	if err == nil {
		t.Fatal("expected the redirect to another host to be refused")
	}
	if !strings.Contains(err.Error(), "redirected to") {
		t.Fatalf("error does not explain the refused redirect: %v", err)
	}
	if reached {
		t.Fatal("the request was replayed to another host")
	}
}

func TestFetchBasePath_RedirectOffTheHostDoesNotCarryTheKey(t *testing.T) {
	// Go's own redirect policy keeps Authorization on any same-host hop and only
	// drops it when the host changes, so the discovery GET needs the destination
	// check too: otherwise a Location naming plain HTTP puts the key on the wire.
	var carried string
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		carried = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"base_path": "/public/triton/stolen"}`))
	}))
	defer elsewhere.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", strings.Replace(elsewhere.URL, "127.0.0.1", "localhost", 1)+r.URL.Path)
		w.WriteHeader(http.StatusMovedPermanently)
	}))
	defer srv.Close()

	client := NewClientWithAPIKey("", srv.URL, "test-key")
	if base := client.fetchBasePath(context.Background(), srv.URL); base != "" {
		t.Fatalf("a refused redirect must leave the caller on the un-scoped prefix, got %q", base)
	}
	if carried != "" {
		t.Fatalf("the API key was carried to another host: %q", carried)
	}
}

func TestMayReplay(t *testing.T) {
	for _, tc := range []struct {
		current, target string
		allowed         bool
	}{
		{"http://host/a", "https://host/a", true}, // the upgrade this exists for
		{"https://host/a", "https://host/b", true},
		{"http://host:8000/a", "http://host:9000/a", true}, // one Triton's own listeners
		{"https://host/a", "http://host/a", false},         // a step down to cleartext
		{"https://host/a", "https://elsewhere/a", false},
		{"http://host/a", "http://elsewhere/a", false},
	} {
		current, _ := neturl.Parse(tc.current)
		target, _ := neturl.Parse(tc.target)
		if got := mayReplay(current, target); got != tc.allowed {
			t.Errorf("mayReplay(%s, %s) = %v, want %v", tc.current, tc.target, got, tc.allowed)
		}
	}
}

func TestInfer_RedirectKeepsThePostAndItsCredential(t *testing.T) {
	// A gateway that terminates TLS redirects plain HTTP to HTTPS; Go's default
	// client would retry the POST as a bodiless GET with no Authorization header.
	var finalMethod, finalAuth, finalBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("moved") == "" {
			w.Header().Set("Location", r.URL.Path+"?moved=1")
			w.WriteHeader(http.StatusMovedPermanently)
			return
		}
		body, _ := io.ReadAll(r.Body)
		finalMethod, finalAuth, finalBody = r.Method, r.Header.Get("Authorization"), string(body)

		result, _ := json.Marshal(map[string]any{"task_type": "GenerationResponse", "response": map[string]any{"response": "hi"}})
		json.NewEncoder(w).Encode(map[string]any{
			"outputs": []map[string]any{{"name": "result", "data": []string{string(result)}}},
		})
	}))
	defer srv.Close()

	client := NewClientWithAPIKey("", srv.URL, "test-key")
	if _, _, err := client.Infer(context.Background(), "test text", "my_task", "my_project"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if finalMethod != http.MethodPost {
		t.Fatalf("redirect changed method to %s, expected POST to be preserved", finalMethod)
	}
	if finalAuth != "Bearer test-key" {
		t.Fatalf("redirect dropped the API key, got %q", finalAuth)
	}
	if finalBody == "" {
		t.Fatal("redirect dropped the request body")
	}
}
