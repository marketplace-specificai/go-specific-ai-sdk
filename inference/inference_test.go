package inference

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
