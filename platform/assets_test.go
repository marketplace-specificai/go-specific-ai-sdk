package platform

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

const (
	testAPIKey   = "sk-test"
	testCookie   = "cookie-test"
	testStatusID = "status-1"
	testCSV      = "example,labels\nhello,0\n"
)

func TestAssetsManager_UploadDataset_FormFieldsAndNoCredentialsToStorage(t *testing.T) {
	var putBody string
	storage := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			t.Errorf("storage got %s, want PUT", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("storage got Authorization %q, want none", got)
		}
		if got := r.Header.Get("Cookie"); got != "" {
			t.Errorf("storage got Cookie %q, want none", got)
		}
		if got := r.Header.Get("Content-Type"); got != "text/csv" {
			t.Errorf("storage got Content-Type %q, want text/csv", got)
		}
		b, _ := io.ReadAll(r.Body)
		putBody = string(b)
	}))
	defer storage.Close()

	var completeForm map[string][]string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer "+testAPIKey {
			t.Errorf("%s got Authorization %q", r.URL.Path, got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/x-www-form-urlencoded" {
			t.Errorf("%s got Content-Type %q, want a form", r.URL.Path, got)
		}
		if err := r.ParseForm(); err != nil {
			t.Fatal(err)
		}
		switch r.URL.Path {
		case "/create-dataset-upload-url":
			if r.PostForm.Get("client_id") != defaultClientID || r.PostForm.Get("filename") != "data.csv" {
				t.Errorf("create form = %v", r.PostForm)
			}
			json.NewEncoder(w).Encode(map[string]string{
				"upload_url":   storage.URL + "/pending/key?X-Amz-Signature=abc",
				"object_key":   "pending/key",
				"status_id":    testStatusID,
				"content_type": "text/csv",
			})
		case "/complete-dataset-upload":
			completeForm = r.PostForm
			json.NewEncoder(w).Encode(map[string]string{"status": "processing", "status_id": testStatusID})
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	defer backend.Close()

	file := filepath.Join(t.TempDir(), "data.csv")
	if err := os.WriteFile(file, []byte(testCSV), 0o600); err != nil {
		t.Fatal(err)
	}
	hc, _ := httpclient.New(backend.URL, httpclient.WithAPIKey(testAPIKey), httpclient.WithSessionCookie("session", testCookie))
	am := &AssetsManager{client: hc, clientID: defaultClientID}

	resp, err := am.UploadDataset(context.Background(), UploadDatasetParams{
		FilePath: file, TaskType: "ClassificationResponse",
		ExampleColumnName: "example", LabelColumnName: "labels",
		Config: &DatasetConfig{DynamicFieldsInPromptTemplate: []string{"a", "b"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusID != testStatusID {
		t.Errorf("status_id = %q", resp.StatusID)
	}
	if putBody != testCSV {
		t.Errorf("storage body = %q", putBody)
	}
	for field, want := range map[string]string{
		"object_key": "pending/key", "status_id": testStatusID, "upload_filename": "data.csv",
		"signed_content_type": "text/csv", "task_type": "ClassificationResponse", "is_benchmark": "false",
		"example_column_name": "example", "label_column_name": "labels", "label_mappings": "{}",
	} {
		if got := completeForm[field]; len(got) != 1 || got[0] != want {
			t.Errorf("complete %s = %v, want %q", field, got, want)
		}
	}
	if got := completeForm["dynamic_fields_in_prompt_template"]; len(got) != 2 {
		t.Errorf("dynamic_fields_in_prompt_template = %v, want both values", got)
	}
}
