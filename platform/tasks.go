package platform

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

// TaskManager manages tasks on the SpecificAI platform.
type TaskManager struct {
	client   *httpclient.Client
	clientID string
}

// Create creates one or more tasks under a project.
func (m *TaskManager) Create(ctx context.Context, projectName string, tasks []TaskCreate, projectGoal string) (*CreateTasksResponse, error) {
	taskPayloads := make([]map[string]any, len(tasks))
	for i, t := range tasks {
		b, _ := json.Marshal(t)
		var m map[string]any
		json.Unmarshal(b, &m)
		taskPayloads[i] = m
	}

	payload := map[string]any{
		"client_id":    m.clientID,
		"group_name":   projectName,
		"project_goal": projectGoal,
		"tasks":        taskPayloads,
	}

	var resp CreateTasksResponse
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/create_llm_usecase", JSONBody: payload,
	}, &resp)
	return &resp, err
}

// List returns all tasks grouped by project.
func (m *TaskManager) List(ctx context.Context) ([]TaskGroup, error) {
	body, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/llm_usecases", JSONBody: map[string]any{"value": m.clientID},
	})
	if err != nil {
		return nil, err
	}
	var groups []TaskGroup
	if err := json.Unmarshal(body, &groups); err != nil {
		return nil, fmt.Errorf("parse task groups: %w", err)
	}
	return groups, nil
}

// Get fetches a single task by ID.
func (m *TaskManager) Get(ctx context.Context, taskID string) (*Task, error) {
	body, _, err := m.client.Do(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/llm_usecase_info",
		JSONBody: map[string]any{"client_id": m.clientID, "llm_usecase_id": taskID},
	})
	if err != nil {
		return nil, err
	}

	var wrapper struct {
		Task Task `json:"llm_usecase"`
	}
	if err := json.Unmarshal(body, &wrapper); err == nil && wrapper.Task.TaskID != "" {
		return &wrapper.Task, nil
	}
	var task Task
	if err := json.Unmarshal(body, &task); err != nil {
		return nil, fmt.Errorf("parse task: %w", err)
	}
	return &task, nil
}

// EditParams holds the mutable fields for editing a task.
type EditParams struct {
	TaskName      *string
	TaskGoal      *string
	TaskType      *string
	ProjectName   *string
	IsMultilabel  *bool
	TaskLanguages []string
}

// Edit updates mutable task fields.
func (m *TaskManager) Edit(ctx context.Context, taskID string, params EditParams) (map[string]any, error) {
	current, err := m.Get(ctx, taskID)
	if err != nil {
		return nil, fmt.Errorf("fetch current task for edit: %w", err)
	}

	taskName := current.TaskName
	if params.TaskName != nil {
		taskName = *params.TaskName
	}
	taskType := current.TaskType
	if params.TaskType != nil {
		taskType = *params.TaskType
	}
	taskGoal := current.TaskGoal
	if params.TaskGoal != nil {
		taskGoal = *params.TaskGoal
	}
	projectName := current.ProjectName
	if params.ProjectName != nil {
		projectName = *params.ProjectName
	}
	isMultilabel := false
	if current.IsMultilabel != nil {
		isMultilabel = *current.IsMultilabel
	}
	if params.IsMultilabel != nil {
		isMultilabel = *params.IsMultilabel
	}
	taskLanguages := current.TaskLanguages
	if params.TaskLanguages != nil {
		taskLanguages = params.TaskLanguages
	}

	payload := map[string]any{
		"usecase_id":        taskID,
		"client_id":         m.clientID,
		"usecase_name":      taskName,
		"usecase_goal":      taskGoal,
		"task_type":         taskType,
		"group_name":        projectName,
		"is_multilabel":     isMultilabel,
		"classify_speakers": false,
		"task_languages":    taskLanguages,
	}

	var resp map[string]any
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/update_llm_usecase", JSONBody: payload,
	}, &resp)
	return resp, err
}

// SaveTeacherAndPrompt saves the teacher model and prompt for a task.
func (m *TaskManager) SaveTeacherAndPrompt(ctx context.Context, taskID, teacher, prompt string, provider ModelProvider, teacherDisplay string) (map[string]any, error) {
	if provider == "" {
		provider = ModelProviderOpenAI
	}
	payload := map[string]any{
		"client_id":       m.clientID,
		"llm_usecase_id":  taskID,
		"teacher":         teacher,
		"prompt":          prompt,
		"provider":        string(provider),
		"teacher_display": teacherDisplay,
	}

	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/save_teacher_and_prompt", JSONBody: payload,
	}, &resp)
	return resp, err
}

// SetComparisonConfig configures comparison mode for a task.
func (m *TaskManager) SetComparisonConfig(ctx context.Context, taskID string, compareWithTeacher, compareWithResultsFile bool, responseParser string) (map[string]any, error) {
	if compareWithTeacher && responseParser == "" {
		return nil, fmt.Errorf("response_parser is required when compareWithTeacher is true")
	}

	if compareWithTeacher && responseParser != "" {
		task, err := m.Get(ctx, taskID)
		if err != nil {
			return nil, err
		}
		err = m.client.DoJSON(ctx, httpclient.RequestParams{
			Method: "POST", Path: "/validate-code",
			JSONBody: map[string]any{
				"code":            responseParser,
				"raw_response":    "example",
				"client_id":       m.clientID,
				"llm_usecase_id":  taskID,
				"task_type":       task.TaskType,
			},
		}, nil)
		if err != nil {
			return nil, err
		}
	}

	var option string
	switch {
	case compareWithTeacher && compareWithResultsFile:
		option = "all"
	case compareWithTeacher:
		option = "teacher"
	case compareWithResultsFile:
		option = "manual_model_predictions"
	default:
		option = "none"
	}

	return m.UpdateComparisonModeOption(ctx, taskID, option)
}

// UpdateComparisonModeOption updates the comparison mode option.
func (m *TaskManager) UpdateComparisonModeOption(ctx context.Context, taskID, option string) (map[string]any, error) {
	_, err := m.UpdateComparisonMode(ctx, taskID, option != "none")
	if err != nil {
		return nil, err
	}

	payload := map[string]any{
		"client_id":              m.clientID,
		"llm_usecase_id":        taskID,
		"comparison_mode_option": option,
	}
	var resp map[string]any
	err = m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/update_comparison_mode_option", JSONBody: payload,
	}, &resp)
	return resp, err
}

// UpdateComparisonMode toggles the comparison mode flag.
func (m *TaskManager) UpdateComparisonMode(ctx context.Context, taskID string, comparisonMode bool) (map[string]any, error) {
	payload := map[string]any{
		"client_id":       m.clientID,
		"llm_usecase_id":  taskID,
		"comparison_mode": comparisonMode,
	}
	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/update_comparison_mode", JSONBody: payload,
	}, &resp)
	return resp, err
}

// Delete removes a task.
func (m *TaskManager) Delete(ctx context.Context, taskID string) (map[string]any, error) {
	var resp map[string]any
	err := m.client.DoJSON(ctx, httpclient.RequestParams{
		Method: "POST", Path: "/delete_llm_usecase",
		JSONBody: map[string]any{"client_id": m.clientID, "usecase_id": taskID},
	}, &resp)
	return resp, err
}
