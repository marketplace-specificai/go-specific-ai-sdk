package platform

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

// TrainingManager manages training jobs on the SpecificAI platform.
type TrainingManager struct {
	client   *httpclient.Client
	clientID string
	tasks    *TaskManager
}

// StartParams holds parameters for starting a training run.
type StartParams struct {
	TaskID             string
	BaseModelName      string
	TrainRatio         int
	ValidationRatio    int
	MetricOptimization string
	BetaForFscore      float64
	HardwareDevice     string
	TrainingArguments  map[string]any
	CommitHash         string
}

// Start starts a training run for a task.
func (m *TrainingManager) Start(ctx context.Context, params StartParams) (*StartTrainingResponse, error) {
	if params.BaseModelName == "" {
		params.BaseModelName = "bert-base-uncased"
	}
	if params.TrainRatio == 0 {
		params.TrainRatio = 70
	}
	if params.ValidationRatio == 0 {
		params.ValidationRatio = 15
	}
	if params.MetricOptimization == "" {
		params.MetricOptimization = "f_beta_score"
	}
	if params.BetaForFscore == 0 {
		params.BetaForFscore = 1.0
	}
	if params.HardwareDevice == "" {
		params.HardwareDevice = "gpu"
	}

	task, err := m.tasks.Get(ctx, params.TaskID)
	if err != nil {
		return nil, fmt.Errorf("fetch task: %w", err)
	}
	if err := validateTaskReadyForTraining(task); err != nil {
		return nil, err
	}

	var currentVersion float64
	if task.Version != nil {
		currentVersion = *task.Version
	}
	newVersion := currentVersion + 1.0

	testRatio := 100 - params.TrainRatio - params.ValidationRatio
	if testRatio < 0 {
		return nil, fmt.Errorf("invalid split ratios: train(%d) + validation(%d) must be <= 100", params.TrainRatio, params.ValidationRatio)
	}

	modelEvalArgs := map[string]any{"optimization_metric": params.MetricOptimization}
	if params.MetricOptimization == "f_beta_score" {
		modelEvalArgs["f_score_metric_beta_coefficient"] = params.BetaForFscore
	}

	comparisonMode := false
	if task.ComparisonMode != nil {
		comparisonMode = *task.ComparisonMode
	}
	isMultilabel := false
	if task.IsMultilabel != nil {
		isMultilabel = *task.IsMultilabel
	}

	distillationEvent := map[string]any{
		"client_id":                  m.clientID,
		"llm_usecase_id":            params.TaskID,
		"datasets":                  task.Datasets,
		"benchmarks":                task.Benchmarks,
		"teacher_model":             task.TeacherModel,
		"task_type":                 task.TaskType,
		"base_distilled_model":      params.BaseModelName,
		"instruction":               task.Prompt,
		"version":                   newVersion,
		"parser_function":           task.ParserFunction,
		"hardware_device":           params.HardwareDevice,
		"comparison_mode":           comparisonMode,
		"comparison_mode_option":    task.ComparisonModeOption,
		"is_multilabel":             isMultilabel,
		"model_evaluation_arguments": modelEvalArgs,
		"trainer_arguments":         params.TrainingArguments,
	}
	if params.CommitHash != "" {
		distillationEvent["commit_hash"] = params.CommitHash
	}
	if params.TrainingArguments == nil {
		distillationEvent["trainer_arguments"] = map[string]any{}
	}

	training := map[string]any{
		"client_id":                  m.clientID,
		"version":                   newVersion,
		"llm_usecase_id":            params.TaskID,
		"base_model_name":           params.BaseModelName,
		"distillation_event_id":     "",
		"training_params": map[string]any{
			"train_ratio":      params.TrainRatio,
			"validation_ratio": params.ValidationRatio,
			"test_ratio":       testRatio,
			"args":             map[string]any{"metric_for_best_model": params.MetricOptimization},
		},
		"hardware_device":            params.HardwareDevice,
		"model_evaluation_arguments": modelEvalArgs,
	}

	return m.StartWithPayloads(ctx, distillationEvent, training)
}

// Stop stops the current training for a task.
func (m *TrainingManager) Stop(ctx context.Context, taskID string) (map[string]any, error) {
	latest, err := m.GetLatest(ctx, taskID)
	if err != nil {
		return nil, err
	}
	if latest == nil {
		return nil, fmt.Errorf("no training found for task %s", taskID)
	}

	var resp map[string]any
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/delete_training",
		JSONBody: map[string]any{
			"client_id":               m.clientID,
			"distillation_event_id":   latest.DistillationEventID,
		},
	}, &resp)
	return resp, err
}

// StartWithPayloads starts training with explicit distillation and training payloads.
func (m *TrainingManager) StartWithPayloads(ctx context.Context, distillationEvent, training map[string]any) (*StartTrainingResponse, error) {
	body, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/create_distillation_event", JSONBody: distillationEvent,
	})
	if err != nil {
		return nil, fmt.Errorf("create distillation event: %w", err)
	}

	var distResp map[string]any
	json.Unmarshal(body, &distResp)
	distID, _ := distResp["distillation_event_id"].(string)

	if distID != "" {
		training["distillation_event_id"] = distID
	}

	body, _, err = m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/create_training", JSONBody: training,
	})
	if err != nil {
		return nil, fmt.Errorf("create training: %w", err)
	}

	var trainResp map[string]any
	json.Unmarshal(body, &trainResp)
	success, _ := trainResp["success"].(bool)

	return &StartTrainingResponse{
		DistillationEventID: distID,
		TrainingCreated:     success,
	}, nil
}

// List returns all training jobs.
func (m *TrainingManager) List(ctx context.Context) ([]TrainingJob, error) {
	body, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/trainings",
		JSONBody: map[string]any{"value": m.clientID},
	})
	if err != nil {
		return nil, err
	}

	var wrapper struct {
		Trainings []TrainingJob `json:"trainings"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse trainings: %w", err)
	}
	return wrapper.Trainings, nil
}

// Get fetches a training by distillation event ID.
func (m *TrainingManager) Get(ctx context.Context, distillationEventID string) (*TrainingJob, error) {
	trainings, err := m.List(ctx)
	if err != nil {
		return nil, err
	}
	for i, t := range trainings {
		if t.DistillationEventID == distillationEventID {
			return &trainings[i], nil
		}
	}
	return nil, nil
}

// GetLatest returns the latest training for a task (by version).
func (m *TrainingManager) GetLatest(ctx context.Context, taskID string) (*TrainingJob, error) {
	trainings, err := m.List(ctx)
	if err != nil {
		return nil, err
	}

	var best *TrainingJob
	var bestVersion float64 = -1

	for i, t := range trainings {
		if t.TaskID != taskID {
			continue
		}
		if t.Version != nil && *t.Version > bestVersion {
			bestVersion = *t.Version
			best = &trainings[i]
		} else if best == nil {
			best = &trainings[i]
		}
	}
	return best, nil
}

func validateTaskReadyForTraining(task *Task) error {
	if len(task.Datasets) == 0 {
		return fmt.Errorf("task not ready: no datasets associated")
	}
	if task.Prompt == "" || task.TeacherModel == "" {
		return fmt.Errorf("task not ready: missing prompt and/or teacher_model")
	}
	comparisonMode := task.ComparisonMode != nil && *task.ComparisonMode
	if comparisonMode &&
		(task.ComparisonModeOption == "all" || task.ComparisonModeOption == "teacher") &&
		task.ParserFunction == "" {
		return fmt.Errorf("task not ready: comparison mode requires parser_function")
	}
	return nil
}
