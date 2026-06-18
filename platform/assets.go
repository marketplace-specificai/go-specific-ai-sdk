package platform

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

// AssetsManager manages dataset uploads on the SpecificAI platform.
type AssetsManager struct {
	client   *httpclient.Client
	clientID string
}

// UploadDatasetParams holds parameters for a dataset upload.
type UploadDatasetParams struct {
	FilePath                    string
	TaskType                    string
	IsBenchmark                 bool
	TaskID                      string
	DataCreationType            string
	ExampleColumnName           string
	LabelColumnName             string
	IsManualModelPredictions    bool
	Config                      *DatasetConfig
}

// UploadDataset uploads a local file as a dataset or benchmark.
func (m *AssetsManager) UploadDataset(ctx context.Context, params UploadDatasetParams) (*UploadDatasetResponse, error) {
	data, err := os.ReadFile(params.FilePath)
	if err != nil {
		return nil, fmt.Errorf("read file %s: %w", params.FilePath, err)
	}

	filename := filepath.Base(params.FilePath)
	contentType := inferContentType(filename)

	var createResp struct {
		UploadURL   string `json:"upload_url"`
		ObjectKey   string `json:"object_key"`
		StatusID    string `json:"status_id"`
		ContentType string `json:"content_type"`
	}
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/create-dataset-upload-url",
		JSONBody: map[string]any{
			"client_id":    m.clientID,
			"filename":     filename,
			"content_type": contentType,
		},
	}, &createResp)
	if err != nil {
		return nil, fmt.Errorf("create upload URL: %w", err)
	}

	putReq, err := http.NewRequestWithContext(ctx, http.MethodPut, createResp.UploadURL, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create PUT request: %w", err)
	}
	putReq.Header.Set("Content-Type", createResp.ContentType)

	putResp, err := http.DefaultClient.Do(putReq)
	if err != nil {
		return nil, fmt.Errorf("upload file: %w", err)
	}
	io.Copy(io.Discard, putResp.Body)
	putResp.Body.Close()
	if putResp.StatusCode >= 300 {
		return nil, fmt.Errorf("upload returned status %d", putResp.StatusCode)
	}

	cfg := params.Config
	if cfg == nil {
		cfg = &DatasetConfig{}
	}

	completePayload := map[string]any{
		"client_id":                         m.clientID,
		"is_benchmark":                      fmt.Sprintf("%t", params.IsBenchmark),
		"is_manual_model_predictions":       fmt.Sprintf("%t", params.IsManualModelPredictions),
		"task_type":                         params.TaskType,
		"usecase_id":                        params.TaskID,
		"data_creation_type":                params.DataCreationType,
		"example_column_name":               params.ExampleColumnName,
		"label_column_name":                 params.LabelColumnName,
		"label_mappings":                    mustJSON(cfg.LabelMappings),
		"relations_column_name":             cfg.RelationsColumnName,
		"relations_mappings":                mustJSON(cfg.RelationsMappings),
		"classification_labels_delimiter":   cfg.ClassificationLabelsDelimiter,
		"examples_are_with_prompt":          fmt.Sprintf("%t", cfg.ExamplesAreWithPrompt),
		"prompt_template":                   cfg.PromptTemplate,
		"example_dynamic_field_name":        cfg.ExampleDynamicFieldName,
		"object_key":                        createResp.ObjectKey,
		"status_id":                         createResp.StatusID,
		"upload_filename":                   filename,
		"signed_content_type":               createResp.ContentType,
	}

	var resp UploadDatasetResponse
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/complete-dataset-upload", JSONBody: completePayload,
	}, &resp)
	return &resp, err
}

// UploadHuggingFaceDatasetParams holds parameters for HuggingFace dataset upload.
type UploadHuggingFaceDatasetParams struct {
	DatasetID        string
	LabelColumn      string
	TextColumn       string
	TaskType         string
	TrainSplit       string
	TestSplit        string
	TaskID           string
	DataCreationType string
	LabelMappings    map[string]any
	IsMultilabel     bool
	Delimiter        string
	DatasetName      string
}

// UploadHuggingFaceDataset imports a HuggingFace dataset.
func (m *AssetsManager) UploadHuggingFaceDataset(ctx context.Context, params UploadHuggingFaceDatasetParams) (*UploadHuggingFaceDatasetResponse, error) {
	if params.DataCreationType == "" {
		params.DataCreationType = "huggingface"
	}
	if params.LabelMappings == nil {
		params.LabelMappings = map[string]any{}
	}

	payload := map[string]any{
		"dataset_id":         params.DatasetID,
		"label_column":       params.LabelColumn,
		"text_column":        params.TextColumn,
		"label_mappings":     params.LabelMappings,
		"client_id":          m.clientID,
		"is_benchmark":       false,
		"task_type":          params.TaskType,
		"usecase_id":         params.TaskID,
		"data_creation_type": params.DataCreationType,
		"train_split":        params.TrainSplit,
		"test_split":         params.TestSplit,
		"is_multilabel":      params.IsMultilabel,
		"delimiter":          params.Delimiter,
	}
	if params.DatasetName != "" {
		payload["dataset_name"] = params.DatasetName
	}

	var resp UploadHuggingFaceDatasetResponse
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/upload-huggingface-dataset", JSONBody: payload,
	}, &resp)
	return &resp, err
}

// GetHuggingFaceDatasetColumns returns column names from a HuggingFace dataset.
func (m *AssetsManager) GetHuggingFaceDatasetColumns(ctx context.Context, datasetID string) (*HuggingFaceDatasetColumnsResponse, error) {
	var resp HuggingFaceDatasetColumnsResponse
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-hf-dataset-columns",
		JSONBody: map[string]any{"dataset_id": datasetID, "client_id": m.clientID},
	}, &resp)
	return &resp, err
}

// GetHuggingFaceDatasetLabels returns unique labels from a HuggingFace dataset column.
func (m *AssetsManager) GetHuggingFaceDatasetLabels(ctx context.Context, datasetID, labelColumn string, isMultilabel bool, delimiter string) (*HuggingFaceDatasetLabelsResponse, error) {
	var resp HuggingFaceDatasetLabelsResponse
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-hf-dataset-labels",
		JSONBody: map[string]any{
			"dataset_id":    datasetID,
			"label_column":  labelColumn,
			"client_id":     m.clientID,
			"is_multilabel": isMultilabel,
			"delimiter":     delimiter,
		},
	}, &resp)
	return &resp, err
}

// GetHuggingFaceDatasetSplits returns available splits for a HuggingFace dataset.
func (m *AssetsManager) GetHuggingFaceDatasetSplits(ctx context.Context, datasetID string) (*HuggingFaceDatasetSplitsResponse, error) {
	var resp HuggingFaceDatasetSplitsResponse
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-hf-dataset-splits",
		JSONBody: map[string]any{"dataset_id": datasetID, "client_id": m.clientID},
	}, &resp)
	return &resp, err
}

// DeleteDataset removes a dataset or benchmark.
func (m *AssetsManager) DeleteDataset(ctx context.Context, dataset string, isBenchmark, isManualModelPredictions bool) (map[string]any, error) {
	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "DELETE", Path: "/datasets",
		JSONBody: map[string]any{
			"client_id":                   m.clientID,
			"dataset":                     dataset,
			"is_benchmark":                isBenchmark,
			"is_manual_model_predictions": isManualModelPredictions,
		},
	}, &resp)
	return resp, err
}

// GetUploadStatus checks dataset upload processing status.
func (m *AssetsManager) GetUploadStatus(ctx context.Context, statusID string) (*UploadStatus, error) {
	var resp UploadStatus
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "GET", Path: fmt.Sprintf("/upload-status/%s", statusID),
	}, &resp)
	return &resp, err
}

// GetFileColumns infers column names from a local file via the backend.
func (m *AssetsManager) GetFileColumns(ctx context.Context, filePath string) (*FileColumnsResponse, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	filename := filepath.Base(filePath)
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(filename)), ".")

	var resp FileColumnsResponse
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-file-columns",
		JSONBody: map[string]any{
			"filename":     filename,
			"client_id":    m.clientID,
			"file_content": base64.StdEncoding.EncodeToString(data),
			"file_type":    ext,
		},
	}, &resp)
	return &resp, err
}

// GetFileLabels infers unique labels from a file column via the backend.
func (m *AssetsManager) GetFileLabels(ctx context.Context, filePath, labelColumn, taskType, delimiter string) (*FileLabelsResponse, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	filename := filepath.Base(filePath)
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(filename)), ".")

	var resp FileLabelsResponse
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-file-labels",
		JSONBody: map[string]any{
			"filename":     filename,
			"client_id":    m.clientID,
			"label_column": labelColumn,
			"file_content": base64.StdEncoding.EncodeToString(data),
			"file_type":    ext,
			"task_type":    taskType,
			"delimiter":    delimiter,
		},
	}, &resp)
	return &resp, err
}

// GetLowConfidenceSamples fetches improvement bucket samples for re-annotation.
func (m *AssetsManager) GetLowConfidenceSamples(ctx context.Context, dataset, label, taskID string, isBenchmark bool, k int) (map[string]any, error) {
	if k == 0 {
		k = 30
	}
	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/get-top-ranked-examples",
		JSONBody: map[string]any{
			"dataset":          dataset,
			"label":            label,
			"client_id":        m.clientID,
			"is_benchmark":     isBenchmark,
			"k":                k,
			"llm_usecase_id":   taskID,
		},
	}, &resp)
	return resp, err
}

func inferContentType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".json":
		return "application/json"
	case ".jsonl", ".ndjson":
		return "application/x-ndjson"
	case ".csv":
		return "text/csv"
	default:
		return "application/octet-stream"
	}
}

func mustJSON(v any) string {
	if v == nil {
		return "{}"
	}
	b, _ := json.Marshal(v)
	return string(b)
}
