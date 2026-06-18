package specificai

import (
	"fmt"
)

// SpecificAIError is the base error type for all SDK errors.
type SpecificAIError struct {
	Message string
	Cause   error
}

func (e *SpecificAIError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

func (e *SpecificAIError) Unwrap() error { return e.Cause }

// APIErrorDetails contains structured information from an HTTP error response.
type APIErrorDetails struct {
	StatusCode   int
	Message      string
	ResponseJSON map[string]any
	ResponseText string
	RequestID    string
}

// APIError is raised when the SpecificAI API returns an error response.
type APIError struct {
	Details APIErrorDetails
}

func (e *APIError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.Details.StatusCode, e.Details.Message)
}

// AuthenticationError represents a 401 Unauthorized response.
type AuthenticationError struct{ APIError }

// PermissionDeniedError represents a 403 Forbidden response.
type PermissionDeniedError struct{ APIError }

// NotFoundError represents a 404 Not Found response.
type NotFoundError struct{ APIError }

// RateLimitError represents a 429 Too Many Requests response.
type RateLimitError struct{ APIError }

// ValidationError represents a 400/422 validation error response.
type ValidationError struct{ APIError }

// ServerError represents a 5xx server error response.
type ServerError struct{ APIError }

// NetworkError is raised on transport-level failures before receiving an HTTP response.
type NetworkError struct {
	SpecificAIError
}

// newAPIError creates the appropriate typed error based on HTTP status code.
func newAPIError(details APIErrorDetails) error {
	base := APIError{Details: details}
	switch {
	case details.StatusCode == 401:
		return &AuthenticationError{base}
	case details.StatusCode == 403:
		return &PermissionDeniedError{base}
	case details.StatusCode == 404:
		return &NotFoundError{base}
	case details.StatusCode == 429:
		return &RateLimitError{base}
	case details.StatusCode == 400 || details.StatusCode == 422:
		return &ValidationError{base}
	case details.StatusCode >= 500 && details.StatusCode <= 599:
		return &ServerError{base}
	default:
		return &base
	}
}
