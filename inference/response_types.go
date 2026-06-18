package inference

// LMResponse is the common interface for all inference response types.
type LMResponse interface {
	IsValidated() bool
}

// ClassificationResponse contains classification results.
type ClassificationResponse struct {
	Labels      []string           `json:"labels"`
	Confidences map[string]float64 `json:"confidences"`
	Thresholds  map[string]float64 `json:"thresholds,omitempty"`
	ExtraParams map[string]any     `json:"extra_params,omitempty"`
	Validated   bool               `json:"is_validated"`
}

func (r *ClassificationResponse) IsValidated() bool { return r.Validated }

// Entity represents a named entity in a NER response.
type Entity struct {
	Label      *string `json:"label,omitempty"`
	Content    *string `json:"content,omitempty"`
	StartIndex *int    `json:"start_index,omitempty"`
	Validated  bool    `json:"is_validated"`
}

// Relation represents a relation between entities.
type Relation struct {
	Label            string `json:"label"`
	SourceStartIndex int    `json:"source_start_index"`
	TargetStartIndex int    `json:"target_start_index"`
	Validated        bool   `json:"is_validated"`
}

// EntityRecognitionResponse contains NER results.
type EntityRecognitionResponse struct {
	Entities    []Entity       `json:"entities"`
	Relations   []Relation     `json:"relations"`
	ExtraParams map[string]any `json:"extra_params,omitempty"`
	Validated   bool           `json:"is_validated"`
}

func (r *EntityRecognitionResponse) IsValidated() bool { return r.Validated }

// GenerationResponse contains text generation results.
type GenerationResponse struct {
	Response  string `json:"response"`
	Validated bool   `json:"is_validated"`
}

func (r *GenerationResponse) IsValidated() bool { return r.Validated }
