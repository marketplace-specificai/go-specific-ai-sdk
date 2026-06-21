package specificai

import "time"

// Option configures the SpecificAI client.
type Option func(*clientConfig)

type clientConfig struct {
	baseURL           string
	apiKey            string
	apiKeyHeader      string
	apiKeyPrefix      string
	timeout           time.Duration
	inferenceURL      string
	trace             bool
	sessionCookie     string
	sessionCookieName string
}

// WithBaseURL sets the SpecificAI backend base URL.
// Can also be set via SPECIFIC_AI_BASE_URL environment variable.
func WithBaseURL(url string) Option {
	return func(c *clientConfig) { c.baseURL = url }
}

// WithAPIKey sets the API key for authentication.
// Can also be set via SPECIFIC_AI_API_KEY environment variable.
func WithAPIKey(key string) Option {
	return func(c *clientConfig) { c.apiKey = key }
}

// WithAPIKeyHeader overrides the authentication header name (default: "Authorization").
func WithAPIKeyHeader(header string) Option {
	return func(c *clientConfig) { c.apiKeyHeader = header }
}

// WithAPIKeyPrefix overrides the authentication prefix (default: "Bearer"). Pass "" for no prefix.
func WithAPIKeyPrefix(prefix string) Option {
	return func(c *clientConfig) { c.apiKeyPrefix = prefix }
}

// WithTimeout sets the default HTTP request timeout (default: 30s).
func WithTimeout(d time.Duration) Option {
	return func(c *clientConfig) { c.timeout = d }
}

// WithInferenceURL sets a direct inference (Triton) URL (e.g. "http://triton:8000").
// Only use this when connecting directly to an inference server, not when going
// through the SpecificAI gateway (WithBaseURL handles that automatically).
func WithInferenceURL(url string) Option {
	return func(c *clientConfig) { c.inferenceURL = url }
}

// WithTrace enables trace collection to the SpecificAI platform.
// Requires WithBaseURL to be set.
func WithTrace(enabled bool) Option {
	return func(c *clientConfig) { c.trace = enabled }
}

// WithSessionCookie sets a session cookie for cookie-based authentication.
func WithSessionCookie(name, value string) Option {
	return func(c *clientConfig) {
		c.sessionCookieName = name
		c.sessionCookie = value
	}
}
