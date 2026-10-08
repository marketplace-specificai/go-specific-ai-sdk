package platform

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
)

func startTrainingAndCapturePayloads(t *testing.T, params StartParams) (distillationEvent, training map[string]any) {
	t.Helper()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		switch r.URL.Path {
		case "/llm_usecase_info":
			json.NewEncoder(w).Encode(map[string]any{
				"llm_usecase": map[string]any{
					"_id":           params.TaskID,
					"task_type":     "Summarization",
					"datasets":      []string{"dataset-1"},
					"prompt":        "Summarize the text.",
					"teacher_model": "gpt-4o",
				},
			})
		case "/create_distillation_event":
			distillationEvent = body
			json.NewEncoder(w).Encode(map[string]any{"distillation_event_id": "dist-1"})
		case "/create_training":
			training = body
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	})

	hc := newTestClient(handler)
	tasks := &TaskManager{client: hc, clientID: defaultClientID}
	tm := &TrainingManager{client: hc, clientID: defaultClientID, tasks: tasks}
	if _, err := tm.Start(context.Background(), params); err != nil {
		t.Fatal(err)
	}
	return distillationEvent, training
}

func TestTrainingManager_Start_AutomaticSendsNoBaseModel(t *testing.T) {
	distillationEvent, training := startTrainingAndCapturePayloads(t, StartParams{TaskID: "task1"})

	if _, ok := distillationEvent["base_distilled_model"]; ok {
		t.Errorf("expected no base_distilled_model, got %v", distillationEvent["base_distilled_model"])
	}
	if _, ok := training["base_model_name"]; ok {
		t.Errorf("expected no base_model_name, got %v", training["base_model_name"])
	}
	if training["task_type"] != "Summarization" {
		t.Errorf("expected training task_type=Summarization, got %v", training["task_type"])
	}
}

func TestValidateTaskReadyForTraining_NoTeacherWithoutComparisonIsReady(t *testing.T) {
	comparisonMode := false
	task := &Task{
		Datasets:             []string{"dataset-1"},
		Prompt:               "Classify.",
		ComparisonMode:       &comparisonMode,
		ComparisonModeOption: "none",
	}

	if err := validateTaskReadyForTraining(task); err != nil {
		t.Errorf("expected task without teacher to be ready when comparison mode is off, got %v", err)
	}
}

func TestValidateTaskReadyForTraining_TeacherComparisonRequiresTeacherModel(t *testing.T) {
	comparisonMode := true
	task := &Task{
		Datasets:             []string{"dataset-1"},
		Prompt:               "Classify.",
		ComparisonMode:       &comparisonMode,
		ComparisonModeOption: "teacher",
		ParserFunction:       "def parse(x): return x",
	}

	if err := validateTaskReadyForTraining(task); err == nil {
		t.Error("expected teacher comparison without teacher_model to be rejected")
	}
}

func TestTrainingManager_Start_ChosenBaseModelIsSent(t *testing.T) {
	distillationEvent, training := startTrainingAndCapturePayloads(t, StartParams{
		TaskID:        "task1",
		BaseModelName: "Qwen/Qwen3-0.6B",
	})

	if distillationEvent["base_distilled_model"] != "Qwen/Qwen3-0.6B" {
		t.Errorf("expected base_distilled_model=Qwen/Qwen3-0.6B, got %v", distillationEvent["base_distilled_model"])
	}
	if training["base_model_name"] != "Qwen/Qwen3-0.6B" {
		t.Errorf("expected base_model_name=Qwen/Qwen3-0.6B, got %v", training["base_model_name"])
	}
}
