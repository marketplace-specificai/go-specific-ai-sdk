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
	resp, logits, err := client.Infer(context.Background(), "test text", "my_task", "my_project", "")
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
		if _, _, err := client.Infer(context.Background(), "hi", "task", "project", ""); err != nil {
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
	resp, _, err := client.Infer(context.Background(), "generate something", "gen_task", "project", "")
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
	resp, _, err := client.Infer(context.Background(), "John went home", "ner_task", "project", "")
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
	_, _, err := client.Infer(context.Background(), "test", "task", "project", "")
	if err == nil {
		t.Fatal("expected error for unknown task type")
	}
}

func TestInfer_NoTargetForMissingWorkspaceReturnsError(t *testing.T) {
	// baseURL set but no workspaceID and no modelID — mode 2 would fail.
	var reached bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "")
	// No modelID, no taskName/projectName — none of the modes applies.
	_, _, err := client.Infer(context.Background(), "test", "", "", "")
	if err == nil {
		t.Fatal("expected error when no inference target is configured")
	}
	if !strings.Contains(err.Error(), "no inference target configured") {
		t.Fatalf("error does not explain the missing target: %v", err)
	}
	if reached {
		t.Fatal("unexpected request was made")
	}
}

func TestInfer_BaseURLWithWorkspaceAndModelIDRoutesToGateway(t *testing.T) {
	const (
		workspaceID = "507f1f77bcf86cd799439011"
		modelID     = "507f1f77bcf86cd799439011.v2"
	)
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		result := map[string]any{"task_type": "ClassificationResponse", "labels": []string{"a"}, "confidences": map[string]float64{"a": 1.0}}
		resultJSON, _ := json.Marshal(result)
		json.NewEncoder(w).Encode(map[string]any{"outputs": []map[string]any{{"name": "result", "data": []string{string(resultJSON)}}}})
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "")
	client.SetWorkspaceID(workspaceID)
	_, _, err := client.Infer(context.Background(), "test", "", "", modelID)
	if err != nil {
		t.Fatal(err)
	}
	want := "/" + workspaceID + "/models/" + modelID + "/infer"
	if capturedPath != want {
		t.Fatalf("expected %s, got %s", want, capturedPath)
	}
}

func TestInfer_NoTargetReturnsError(t *testing.T) {
	client := NewClient("", "")
	_, _, err := client.Infer(context.Background(), "test", "task", "proj", "")
	if err == nil {
		t.Fatal("expected error when no inference target is configured")
	}
	if !strings.Contains(err.Error(), "no inference target configured") {
		t.Fatalf("error does not explain the missing target: %v", err)
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
	_, _, err := client.Infer(context.Background(), "test", "my task/v2", "project name", "")
	if err != nil {
		t.Fatal(err)
	}
	expected := "/v2/models/model_project_name_my_task_v2/infer"
	if capturedURL != expected {
		t.Fatalf("expected URL %s, got %s", expected, capturedURL)
	}
}

func TestInfer_DifferentModelIDsOnSameClient(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		result := map[string]any{
			"task_type":   "ClassificationResponse",
			"labels":      []string{"a"},
			"confidences": map[string]float64{"a": 1.0},
		}
		resultJSON, _ := json.Marshal(result)
		json.NewEncoder(w).Encode(map[string]any{
			"outputs": []map[string]any{
				{"name": "result", "data": []string{string(resultJSON)}},
			},
		})
	}))
	defer srv.Close()

	const workspaceID = "507f1f77bcf86cd799439011"
	client := NewClient(srv.URL, "")
	client.SetWorkspaceID(workspaceID)
	if _, _, err := client.Infer(context.Background(), "v1", "", "", workspaceID+".v1"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := client.Infer(context.Background(), "v2", "", "", workspaceID+".v2"); err != nil {
		t.Fatal(err)
	}

	// Version is encoded in model_id; Triton path has no /versions/ segment.
	if strings.Contains(paths[0], "/versions/") {
		t.Fatalf("unexpected /versions/ in path: %s", paths[0])
	}
	if strings.Contains(paths[1], "/versions/") {
		t.Fatalf("unexpected /versions/ in path: %s", paths[1])
	}
	if !strings.HasSuffix(paths[0], "/infer") || !strings.HasSuffix(paths[1], "/infer") {
		t.Fatalf("expected paths ending in /infer, got %s and %s", paths[0], paths[1])
	}
}

func TestInfer_RedirectOffTheHostIsRefused(t *testing.T) {
	var reached bool
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
	}))
	defer elsewhere.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", strings.Replace(elsewhere.URL, "127.0.0.1", "localhost", 1)+r.URL.Path)
		w.WriteHeader(http.StatusMovedPermanently)
	}))
	defer srv.Close()

	client := NewClientWithAPIKey("", srv.URL, "test-key")
	_, _, err := client.Infer(context.Background(), "test text", "my_task", "my_project", "")
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

func TestMayReplay(t *testing.T) {
	for _, tc := range []struct {
		current, target string
		allowed         bool
	}{
		{"http://host/a", "https://host/a", true},
		{"https://host/a", "https://host/b", true},
		{"http://host:8000/a", "http://host:9000/a", true},
		{"https://host/a", "http://host/a", false},
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

func TestInfer_GatewayMode2UsesModelIDURL(t *testing.T) {
	// Mode 2: baseURL + workspaceID + modelID (per call) -> /{ws}/models/{id}/infer.
	const (
		workspaceID = "507f1f77bcf86cd799439011"
		modelID     = "507f1f77bcf86cd799439011.v2"
	)

	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	client.SetWorkspaceID(workspaceID)

	// taskName and projectName are ignored when modelID is set.
	if _, _, err := client.Infer(context.Background(), "hello", "", "", modelID); err != nil {
		t.Fatal(err)
	}

	want := "/" + workspaceID + "/models/" + modelID + "/infer"
	if capturedPath != want {
		t.Fatalf("expected mode-2 gateway URL %s, got %s", want, capturedPath)
	}
}

func TestInfer_TritonNamePathHasNoVersionsSegment(t *testing.T) {
	// Mode 2 (model_id): model ID encodes version; no /versions/ in path.
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		result := map[string]any{"task_type": "ClassificationResponse", "labels": []string{"a"}, "confidences": map[string]float64{"a": 1.0}}
		resultJSON, _ := json.Marshal(result)
		json.NewEncoder(w).Encode(map[string]any{"outputs": []map[string]any{{"name": "result", "data": []string{string(resultJSON)}}}})
	}))
	defer srv.Close()

	const workspaceID = "507f1f77bcf86cd799439011"
	client := NewClient(srv.URL, "")
	client.SetWorkspaceID(workspaceID)
	if _, _, err := client.Infer(context.Background(), "hello", "", "", workspaceID+".v3"); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(capturedPath, "/versions/") {
		t.Fatalf("unexpected /versions/ in gateway model-ID path: %s", capturedPath)
	}
}

func TestInfer_RedirectKeepsThePostAndItsCredential(t *testing.T) {
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
	if _, _, err := client.Infer(context.Background(), "test text", "my_task", "my_project", ""); err != nil {
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
