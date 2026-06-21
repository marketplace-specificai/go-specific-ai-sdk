package inference

// TaskType represents the type of ML task.
type TaskType string

const (
	TaskTypeClassification    TaskType = "ClassificationResponse"
	TaskTypeGeneration        TaskType = "GenerationResponse"
	TaskTypeEntityRecognition TaskType = "EntityRecognitionResponse"
	TaskTypeQA                TaskType = "QandA"
	TaskTypeChat              TaskType = "chat"
	TaskTypeDomainAdaptation  TaskType = "domainAdaptation"
)
