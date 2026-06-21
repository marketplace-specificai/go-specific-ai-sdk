package platform

// Task represents a task in the SpecificAI platform.
type Task struct {
	TaskID               string   `json:"_id,omitempty"`
	ClientID             string   `json:"client_id,omitempty"`
	ProjectName          string   `json:"group_name,omitempty"`
	SubGroupName         string   `json:"sub_group_name,omitempty"`
	TaskName             string   `json:"usecase_name,omitempty"`
	TaskGoal             string   `json:"usecase_goal,omitempty"`
	TaskType             string   `json:"task_type,omitempty"`
	Datasets             []string `json:"datasets"`
	Benchmarks           []string `json:"benchmarks"`
	Prompt               string   `json:"prompt,omitempty"`
	TeacherModel         string   `json:"teacher_model,omitempty"`
	ParserFunction       string   `json:"parser_function,omitempty"`
	ComparisonMode       *bool    `json:"comparison_mode,omitempty"`
	ComparisonModeOption string   `json:"comparison_mode_option,omitempty"`
	IsMultilabel         *bool    `json:"is_multilabel,omitempty"`
	TaskLanguages        []string `json:"task_languages"`
	Version              *float64 `json:"version,omitempty"`
	Status               string   `json:"status,omitempty"`
}

// TaskCreate is the payload for creating a task.
type TaskCreate struct {
	TaskName                 string   `json:"usecase_name"`
	TaskType                 string   `json:"task_type"`
	TaskGoal                 string   `json:"usecase_goal,omitempty"`
	ComparisonMode           bool     `json:"comparison_mode"`
	Datasets                 []string `json:"datasets"`
	Benchmarks               []string `json:"benchmarks"`
	Prompt                   string   `json:"prompt,omitempty"`
	TeacherModel             string   `json:"teacher_model,omitempty"`
	ParserFunction           string   `json:"parser_function,omitempty"`
	IsMultilabel             bool     `json:"is_multilabel"`
	TaskLanguages            []string `json:"task_languages"`
	ManualModelPredictions   []string `json:"manual_model_predictions"`
}

// TaskSubGroup is a subgroup of tasks within a project.
type TaskSubGroup struct {
	SubGroupName string `json:"sub_group_name"`
	Tasks        []Task `json:"usecases"`
}

// TaskGroup is a project grouping of tasks.
type TaskGroup struct {
	ProjectName string         `json:"group_name"`
	ProjectGoal string         `json:"project_goal,omitempty"`
	SubGroups   []TaskSubGroup `json:"sub_groups"`
}

// IterTasks flattens all tasks across sub-groups.
func (g *TaskGroup) IterTasks() []Task {
	var out []Task
	for _, sg := range g.SubGroups {
		out = append(out, sg.Tasks...)
	}
	return out
}

// CreateTasksResponse is returned by the create task endpoint.
type CreateTasksResponse struct {
	Success        bool     `json:"success"`
	CreatedTaskIDs []string `json:"created_usecases"`
}

// TrainingJob represents a training run.
type TrainingJob struct {
	ID                    string         `json:"id,omitempty"`
	TaskID                string         `json:"llm_usecase_id,omitempty"`
	DistillationEventID   string         `json:"distillation_event_id,omitempty"`
	Status                string         `json:"status,omitempty"`
	Progress              *int           `json:"progress,omitempty"`
	CreatedDatetime       string         `json:"created_datetime,omitempty"`
	FinishedDatetime      string         `json:"finished_datetime,omitempty"`
	Version               *float64       `json:"version,omitempty"`
	CreatedByUser         map[string]any `json:"created_by_user,omitempty"`
}

// ModelMetrics is evaluation data returned by POST /evaluation.
type ModelMetrics struct {
	Success            *bool          `json:"success,omitempty"`
	Versions           []float64      `json:"versions"`
	AllMetrics         map[string]any `json:"all_metrics"`
	ComparativeMetrics map[string]any `json:"comparativeMetrics"`
	ConfusionMatrix    map[string]any `json:"confusionMatrix"`
}

// StartTrainingResponse is returned after starting a training run.
type StartTrainingResponse struct {
	DistillationEventID string `json:"distillation_event_id,omitempty"`
	TrainingCreated     bool   `json:"training_created"`
}

// UploadDatasetResponse is returned by the dataset upload flow.
type UploadDatasetResponse struct {
	Status            string   `json:"status"`
	StatusID          string   `json:"status_id"`
	CurrentDatasets   []string `json:"current_datasets"`
	CurrentBenchmarks []string `json:"current_benchmarks"`
}

// DatasetConfig holds optional dataset upload configuration.
type DatasetConfig struct {
	LabelMappings                   map[string]any `json:"label_mappings,omitempty"`
	RelationsMappings               map[string]any `json:"relations_mappings,omitempty"`
	RelationsColumnName             string         `json:"relations_column_name,omitempty"`
	ClassificationLabelsDelimiter   string         `json:"classification_labels_delimiter,omitempty"`
	ExamplesAreWithPrompt           bool           `json:"examples_are_with_prompt"`
	PromptTemplate                  string         `json:"prompt_template,omitempty"`
	DynamicFieldsInPromptTemplate   []string       `json:"dynamic_fields_in_prompt_template,omitempty"`
	ExampleDynamicFieldName         string         `json:"example_dynamic_field_name,omitempty"`
}

// UploadHuggingFaceDatasetResponse is returned by HuggingFace dataset upload.
type UploadHuggingFaceDatasetResponse struct {
	Success  bool   `json:"success"`
	StatusID string `json:"status_id"`
	Message  string `json:"message,omitempty"`
}

// UploadStatus is returned by the upload status endpoint.
type UploadStatus struct {
	Filename      string `json:"filename"`
	TotalRows     int    `json:"total_rows"`
	ProcessedRows int    `json:"processed_rows"`
	Status        string `json:"status"`
	Error         string `json:"error,omitempty"`
	StartedAt     string `json:"started_at,omitempty"`
	CompletedAt   string `json:"completed_at,omitempty"`
}

// FileColumnsResponse is returned by the file columns endpoint.
type FileColumnsResponse struct {
	Success bool     `json:"success"`
	Columns []string `json:"columns"`
	Error   string   `json:"error,omitempty"`
}

// FileLabelsResponse is returned by the file labels endpoint.
type FileLabelsResponse struct {
	Success           bool           `json:"success"`
	Labels            []string       `json:"labels"`
	LabelExamples     map[string]any `json:"label_examples"`
	Error             string         `json:"error,omitempty"`
	PreviewTruncated  bool           `json:"preview_truncated"`
	PreviewSizeBytes  *int           `json:"preview_size_bytes,omitempty"`
	OriginalSizeBytes *int           `json:"original_size_bytes,omitempty"`
	TotalLabelCount   *int           `json:"total_label_count,omitempty"`
}

// HuggingFaceDatasetColumnsResponse is returned by the HF columns endpoint.
type HuggingFaceDatasetColumnsResponse struct {
	Success bool     `json:"success"`
	Columns []string `json:"columns"`
	Error   string   `json:"error,omitempty"`
}

// HuggingFaceDatasetLabelsResponse is returned by the HF labels endpoint.
type HuggingFaceDatasetLabelsResponse struct {
	Success bool     `json:"success"`
	Labels  []string `json:"labels"`
	Error   string   `json:"error,omitempty"`
}

// HuggingFaceDatasetSplitsResponse is returned by the HF splits endpoint.
type HuggingFaceDatasetSplitsResponse struct {
	Success bool     `json:"success"`
	Splits  []string `json:"splits"`
	Error   string   `json:"error,omitempty"`
}

// ModelResponse is a single prediction row from /model-responses.
type ModelResponse struct {
	RequestExample          string         `json:"request_example"`
	ExpectedResponse        map[string]any `json:"expected_response"`
	ModelResponseData       map[string]any `json:"model_response"`
	ComparisonModelResponse map[string]any `json:"comparison_model_response,omitempty"`
	TeacherResponse         map[string]any `json:"teacher_response,omitempty"`
}

// ModelProvider represents supported model providers.
type ModelProvider string

const (
	ModelProviderOpenAI  ModelProvider = "openai"
	ModelProviderBedrock ModelProvider = "bedrock"
	ModelProviderLocal   ModelProvider = "local"
)

// ModelVersionInfo is a summary of a model version from list_versions.
type ModelVersionInfo struct {
	Version              *float64       `json:"version"`
	CreatedAt            string         `json:"created_at,omitempty"`
	FinishedAt           string         `json:"finished_at,omitempty"`
	Status               string         `json:"status,omitempty"`
	Progress             *int           `json:"progress,omitempty"`
	DistillationEventID  string         `json:"distillation_event_id,omitempty"`
	TrainingID           string         `json:"training_id,omitempty"`
	CreatedByUser        map[string]any `json:"created_by_user,omitempty"`
	Metrics              *ModelMetrics  `json:"metrics,omitempty"`
}
