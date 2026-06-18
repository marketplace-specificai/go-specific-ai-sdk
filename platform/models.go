package platform

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

// ModelManager manages trained models on the SpecificAI platform.
type ModelManager struct {
	client   *httpclient.Client
	clientID string
}

// GetMetricsParams holds parameters for fetching model metrics.
type GetMetricsParams struct {
	TaskID             string
	Version            *float64
	ComparisonMode     bool
	CompareWithVersion *float64
	CompareWithFile    bool
	SubTaskType        string
	TaskType           string
}

// GetMetrics fetches evaluation metrics for a model.
func (m *ModelManager) GetMetrics(ctx context.Context, params GetMetricsParams) (*ModelMetrics, error) {
	payload := map[string]any{
		"client_id":            m.clientID,
		"llm_usecase_id":      params.TaskID,
		"version":             params.Version,
		"comparison_mode":     params.ComparisonMode,
		"compare_with_version": params.CompareWithVersion,
		"compare_with_file":   params.CompareWithFile,
		"sub_task_type":       params.SubTaskType,
		"task_type":           params.TaskType,
	}
	var resp ModelMetrics
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/evaluation", JSONBody: payload,
	}, &resp)
	return &resp, err
}

// ListVersions returns all model versions for a task with optional metrics.
func (m *ModelManager) ListVersions(ctx context.Context, taskID string) ([]ModelVersionInfo, error) {
	body, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/trainings",
		JSONBody: map[string]any{"value": m.clientID},
	})
	if err != nil {
		return nil, err
	}

	var wrapper struct {
		Trainings []map[string]any `json:"trainings"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse trainings: %w", err)
	}

	var out []ModelVersionInfo
	for _, t := range wrapper.Trainings {
		tid, _ := t["llm_usecase_id"].(string)
		if tid != taskID {
			continue
		}

		info := ModelVersionInfo{
			CreatedAt:           strVal(t, "created_datetime"),
			FinishedAt:          strVal(t, "finished_datetime"),
			Status:              strVal(t, "status"),
			DistillationEventID: strVal(t, "distillation_event_id"),
			TrainingID:          strVal(t, "id"),
		}
		if v, ok := t["version"].(float64); ok {
			info.Version = &v
		}
		if p, ok := t["progress"].(float64); ok {
			pi := int(p)
			info.Progress = &pi
		}
		if u, ok := t["created_by_user"].(map[string]any); ok {
			info.CreatedByUser = u
		}

		if info.Version != nil {
			metrics, err := m.GetMetrics(ctx, GetMetricsParams{TaskID: taskID, Version: info.Version})
			if err == nil {
				info.Metrics = metrics
			}
		}

		out = append(out, info)
	}

	// Sort by version descending.
	for i := 0; i < len(out); i++ {
		for j := i + 1; j < len(out); j++ {
			vi, vj := float64(-1), float64(-1)
			if out[i].Version != nil {
				vi = *out[i].Version
			}
			if out[j].Version != nil {
				vj = *out[j].Version
			}
			if vj > vi {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out, nil
}

// DeleteVersion deletes a specific model version.
func (m *ModelManager) DeleteVersion(ctx context.Context, taskID string, version float64) (map[string]any, error) {
	versions, err := m.ListVersions(ctx, taskID)
	if err != nil {
		return nil, err
	}

	var distID string
	for _, v := range versions {
		if v.Version != nil && *v.Version == version {
			distID = v.DistillationEventID
			break
		}
	}
	if distID == "" {
		return nil, fmt.Errorf("version %.0f not found for task %s", version, taskID)
	}

	var resp map[string]any
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/delete_training",
		JSONBody: map[string]any{
			"client_id":             m.clientID,
			"distillation_event_id": distID,
		},
	}, &resp)
	return resp, err
}

// Deploy deploys or undeploys a model version.
func (m *ModelManager) Deploy(ctx context.Context, taskID, taskType string, version float64, requestType string) (map[string]any, error) {
	if requestType == "" {
		requestType = "deploy"
	}
	payload := map[string]any{
		"client_id":         m.clientID,
		"llm_usecase_id":   taskID,
		"task_type":        taskType,
		"request_type":     requestType,
		"deployed_version": version,
	}
	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/model-deployment", JSONBody: payload,
	}, &resp)
	return resp, err
}

// Evaluate triggers re-evaluation for a model version.
func (m *ModelManager) Evaluate(ctx context.Context, taskID string, version *float64) (map[string]any, error) {
	ver := version
	if ver == nil {
		metrics, err := m.GetMetrics(ctx, GetMetricsParams{TaskID: taskID})
		if err != nil {
			return nil, err
		}
		if len(metrics.Versions) == 0 {
			return nil, fmt.Errorf("no versions found for task %s", taskID)
		}
		maxV := metrics.Versions[0]
		for _, v := range metrics.Versions[1:] {
			if v > maxV {
				maxV = v
			}
		}
		ver = &maxV
	}

	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/recalculate-evaluation",
		JSONBody: map[string]any{
			"client_id":       m.clientID,
			"llm_usecase_id":  taskID,
			"version":         *ver,
		},
	}, &resp)
	return resp, err
}

// GetBenchmarkPredictionsParams holds parameters for fetching benchmark predictions.
type GetBenchmarkPredictionsParams struct {
	TaskID             string
	Version            float64
	CompareWithVersion *float64
	CompareWithFile    bool
	SubTaskType        string
	PageSize           int
}

// GetBenchmarkPredictions fetches paginated benchmark model responses.
func (m *ModelManager) GetBenchmarkPredictions(ctx context.Context, params GetBenchmarkPredictionsParams) ([]ModelResponse, error) {
	if params.PageSize == 0 {
		params.PageSize = 200
	}

	var all []ModelResponse
	for page := 0; ; page++ {
		payload := map[string]any{
			"client_id":             m.clientID,
			"llm_usecase_id":       params.TaskID,
			"version":              params.Version,
			"page":                 page,
			"page_size":            params.PageSize,
			"filters":              map[string]any{},
			"comparison_version":   params.CompareWithVersion,
			"is_version_comparison": params.CompareWithVersion != nil,
			"is_file_comparison":   params.CompareWithFile,
			"sub_task_type":        params.SubTaskType,
			"primary_model":       "student",
		}

		body, _, err := m.client.Do(ctx, httpclient.RequestParams{
			Method: "POST", Path: "/model-responses", JSONBody: payload,
		})
		if err != nil {
			return all, err
		}

		var resp struct {
			Success bool            `json:"success"`
			Data    []ModelResponse `json:"data"`
		}
		if err := json.Unmarshal(body, &resp); err != nil || !resp.Success || len(resp.Data) == 0 {
			break
		}
		all = append(all, resp.Data...)
	}
	return all, nil
}

// DownloadModel downloads a model artifact to a local path (full flow).
func (m *ModelManager) DownloadModel(ctx context.Context, taskID string, version float64, outputPath string, timeoutSeconds float64) (string, error) {
	if timeoutSeconds == 0 {
		timeoutSeconds = 900
	}

	clientUUID := fmt.Sprintf("%d", time.Now().UnixNano())
	_, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/download-model-request",
		JSONBody: map[string]any{
			"client_id":         m.clientID,
			"llm_usecase_id":   taskID,
			"deployed_version": version,
			"client_uuid":      clientUUID,
		},
	})
	if err != nil {
		return "", fmt.Errorf("request download: %w", err)
	}

	deadline := time.Now().Add(time.Duration(timeoutSeconds) * time.Second)
	pollTimeout := 60.0

	var completion map[string]any
	for time.Now().Before(deadline) {
		remaining := time.Until(deadline).Seconds()
		pt := pollTimeout
		if remaining < pt {
			pt = remaining
		}
		if pt < 1 {
			pt = 1
		}

		body, _, err := m.client.Do(ctx, httpclient.RequestParams{
			Method: "POST", Path: "/wait-download-completion",
			JSONBody: map[string]any{"client_uuid": clientUUID, "timeout_s": pt},
			Timeout: time.Duration(pt+10) * time.Second,
		})
		if err != nil {
			continue
		}

		var resp map[string]any
		if err := json.Unmarshal(body, &resp); err != nil {
			continue
		}

		if resp["action"] != "downloadCompletionUpdate" {
			continue
		}
		if isErr, _ := resp["isError"].(bool); isErr {
			completion = resp
			break
		}
		if isCompleted, _ := resp["isCompleted"].(bool); isCompleted {
			completion = resp
			break
		}
	}

	if completion == nil {
		return "", fmt.Errorf("timeout waiting for model packaging (task=%s, version=%.0f)", taskID, version)
	}
	if isErr, _ := completion["isError"].(bool); isErr {
		msg, _ := completion["errorMessage"].(string)
		return "", fmt.Errorf("model packaging failed: %s", msg)
	}

	downloadPath, _ := completion["downloadPath"].(string)
	filename, _ := completion["filename"].(string)
	resolvedVersion, ok := completion["version"].(float64)
	if !ok {
		resolvedVersion = version
	}

	if downloadPath == "" || filename == "" {
		return "", fmt.Errorf("download completion missing downloadPath/filename")
	}

	out := outputPath
	info, err := os.Stat(out)
	if err == nil && info.IsDir() {
		out = filepath.Join(out, filename)
	}
	os.MkdirAll(filepath.Dir(out), 0o755)

	streamBody, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/download-model",
		JSONBody: map[string]any{
			"client_id":       m.clientID,
			"llm_usecase_id":  taskID,
			"version":         resolvedVersion,
			"download_path":   downloadPath,
			"filename":        filename,
		},
		Timeout: 300 * time.Second,
	})
	if err != nil {
		return "", fmt.Errorf("download model: %w", err)
	}

	f, err := os.Create(out)
	if err != nil {
		return "", fmt.Errorf("create output file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(f, io.NopCloser(io.NewSectionReader(bytesReaderAt(streamBody), 0, int64(len(streamBody))))); err != nil {
		return "", fmt.Errorf("write model file: %w", err)
	}

	return out, nil
}

type bytesReaderAt []byte

func (b bytesReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(b)) {
		return 0, io.EOF
	}
	n := copy(p, b[off:])
	return n, nil
}

func strVal(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

// StreamDownload downloads a model using a streaming HTTP response.
func (m *ModelManager) StreamDownload(ctx context.Context, taskID string, version float64, downloadPath, filename, outputPath string) (string, error) {
	out := outputPath
	os.MkdirAll(filepath.Dir(out), 0o755)

	payload := map[string]any{
		"client_id":       m.clientID,
		"llm_usecase_id":  taskID,
		"version":         version,
		"download_path":   downloadPath,
		"filename":        filename,
	}

	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal download request: %w", err)
	}

	baseURL := m.client.BaseURL()
	url := baseURL + "download-model"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, io.NopCloser(io.NewSectionReader(bytesReaderAt(bodyBytes), 0, int64(len(bodyBytes)))))
	if err != nil {
		return "", fmt.Errorf("create download request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download request: %w", err)
	}
	defer resp.Body.Close()

	f, err := os.Create(out)
	if err != nil {
		return "", fmt.Errorf("create output file: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(f, resp.Body); err != nil {
		return "", fmt.Errorf("write model file: %w", err)
	}
	return out, nil
}
