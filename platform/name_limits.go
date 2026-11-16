package platform

import (
	"context"
	"fmt"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

const namePreviewChars = 40

// NameLimits holds the platform's longest accepted task name and label name, in
// characters (GET /name-limits; NAME_MAX_LENGTH / LABEL_NAME_MAX_LENGTH, 32 and 100 by default).
type NameLimits struct {
	TaskNameMaxLength  int `json:"task_name_max_length"`
	LabelNameMaxLength int `json:"label_name_max_length"`
}

// nameLimits fetches the limits once and keeps them. A platform that cannot
// answer (an older backend, a blip) yields nil and is asked again next time;
// the names are then left to the platform to check.
func (m *TaskManager) nameLimits(ctx context.Context) *NameLimits {
	m.limitsMu.Lock()
	defer m.limitsMu.Unlock()
	if m.limits != nil {
		return m.limits
	}
	var limits NameLimits
	err := m.client.DoJSON(ctx, httpclient.RequestParams{Method: "GET", Path: "/name-limits"}, &limits)
	if err != nil || limits.TaskNameMaxLength < 1 || limits.LabelNameMaxLength < 1 {
		return nil
	}
	m.limits = &limits
	return m.limits
}

// checkTaskNames returns an error for the first task name longer than the platform accepts.
func (m *TaskManager) checkTaskNames(ctx context.Context, names ...string) error {
	var longest string
	for _, name := range names {
		if len([]rune(name)) > len([]rune(longest)) {
			longest = name
		}
	}
	if longest == "" {
		return nil
	}
	limits := m.nameLimits(ctx)
	if limits == nil {
		return nil
	}
	length := len([]rune(longest))
	if length <= limits.TaskNameMaxLength {
		return nil
	}
	preview := []rune(longest)
	if len(preview) > namePreviewChars {
		preview = append(preview[:namePreviewChars], []rune("...")...)
	}
	return fmt.Errorf(
		"task name must be at most %d characters; %q is %d characters long (NAME_MAX_LENGTH on the platform)",
		limits.TaskNameMaxLength, string(preview), length,
	)
}
