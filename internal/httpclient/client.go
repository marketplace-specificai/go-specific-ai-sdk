package httpclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Option configures the HTTP client.
type Option func(*Client)

// WithAPIKey sets the Bearer token for authentication.
func WithAPIKey(key string) Option {
	return func(c *Client) { c.apiKey = key }
}

// WithAPIKeyHeader overrides the header name (default "Authorization").
func WithAPIKeyHeader(header string) Option {
	return func(c *Client) { c.apiKeyHeader = header }
}

// WithAPIKeyPrefix overrides the prefix (default "Bearer"). Pass "" for no prefix.
func WithAPIKeyPrefix(prefix string) Option {
	return func(c *Client) { c.apiKeyPrefix = prefix }
}

// WithTimeout sets the default request timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.timeout = d }
}

// WithMaxRetries sets the maximum number of retry attempts.
func WithMaxRetries(n int) Option {
	return func(c *Client) { c.maxRetries = n }
}

// WithBackoffFactor sets the exponential backoff factor.
func WithBackoffFactor(f float64) Option {
	return func(c *Client) { c.backoffFactor = f }
}

// WithSessionCookie sets a session cookie for cookie-based auth.
func WithSessionCookie(name, value string) Option {
	return func(c *Client) {
		c.sessionCookieName = name
		c.sessionCookieValue = value
	}
}

// WithHTTPClient overrides the underlying http.Client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// Client is the core HTTP client for SpecificAI API calls.
type Client struct {
	baseURL            string
	apiKey             string
	apiKeyHeader       string
	apiKeyPrefix       string
	timeout            time.Duration
	maxRetries         int
	backoffFactor      float64
	sessionCookieName  string
	sessionCookieValue string
	httpClient         *http.Client
}

// New creates a new HTTP client.
func New(baseURL string, opts ...Option) (*Client, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return nil, fmt.Errorf("base_url is required")
	}
	baseURL = strings.TrimRight(baseURL, "/") + "/"

	c := &Client{
		baseURL:       baseURL,
		apiKeyHeader:  "Authorization",
		apiKeyPrefix:  "Bearer",
		timeout:       30 * time.Second,
		maxRetries:    3,
		backoffFactor: 0.5,
		httpClient: &http.Client{
			// Prevent Go's default redirect behavior from silently changing POST
			// to GET on 301/302 redirects, which causes 405 errors on POST-only
			// backend routes.
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c, nil
}

// BaseURL returns the configured base URL.
func (c *Client) BaseURL() string { return c.baseURL }

// RequestParams holds parameters for an HTTP request.
type RequestParams struct {
	Method   string
	Path     string
	JSONBody any
	Query    url.Values
	Headers  map[string]string
	Timeout  time.Duration
}

// Do executes an HTTP request with retries and error handling.
// It returns the raw response body and status code.
func (c *Client) Do(ctx context.Context, p RequestParams) ([]byte, int, error) {
	reqURL := c.resolveURL(p.Path)

	var bodyBytes []byte
	if p.JSONBody != nil {
		var err error
		bodyBytes, err = json.Marshal(p.JSONBody)
		if err != nil {
			return nil, 0, fmt.Errorf("marshal request body: %w", err)
		}
	}

	timeout := c.timeout
	if p.Timeout > 0 {
		timeout = p.Timeout
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(float64(time.Second) * c.backoffFactor * math.Pow(2, float64(attempt-1)))
			select {
			case <-ctx.Done():
				return nil, 0, &networkError{msg: "request cancelled", cause: ctx.Err()}
			case <-time.After(backoff):
			}
		}

		reqCtx, cancel := context.WithTimeout(ctx, timeout)
		var body io.Reader
		if bodyBytes != nil {
			body = bytes.NewReader(bodyBytes)
		}
		req, err := http.NewRequestWithContext(reqCtx, p.Method, reqURL, body)
		if err != nil {
			cancel()
			return nil, 0, &networkError{msg: "create request", cause: err}
		}

		c.setHeaders(req, p.Headers, bodyBytes != nil)

		if p.Query != nil {
			req.URL.RawQuery = p.Query.Encode()
		}

		resp, respBody, doErr := c.doWithRedirects(req, p.Method, bodyBytes)
		cancel()
		if doErr != nil {
			lastErr = &networkError{msg: "execute request", cause: doErr}
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return respBody, resp.StatusCode, nil
		}

		apiErr := toAPIError(resp.StatusCode, respBody, resp.Header.Get("x-request-id"))
		if isRetryable(resp.StatusCode) && attempt < c.maxRetries {
			lastErr = apiErr
			continue
		}
		return nil, resp.StatusCode, apiErr
	}
	return nil, 0, lastErr
}

// DoJSON executes a request and unmarshals the response into target.
func (c *Client) DoJSON(ctx context.Context, p RequestParams, target any) error {
	body, statusCode, err := c.Do(ctx, p)
	if err != nil {
		return err
	}
	if target == nil {
		return nil
	}
	if err := json.Unmarshal(body, target); err != nil {
		return toAPIError(statusCode, body, "")
	}
	return nil
}

// doWithRedirects executes a request and follows up to 10 redirects while
// preserving the original HTTP method (Go's default client downgrades POST to
// GET on 301/302, which causes 405 errors on POST-only API routes).
func (c *Client) doWithRedirects(req *http.Request, method string, bodyBytes []byte) (*http.Response, []byte, error) {
	const maxRedirects = 10
	for i := 0; i <= maxRedirects; i++ {
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return nil, nil, err
		}
		respBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, nil, readErr
		}

		if resp.StatusCode < 300 || resp.StatusCode >= 400 {
			return resp, respBody, nil
		}

		loc := resp.Header.Get("Location")
		if loc == "" {
			return resp, respBody, nil
		}
		redirectURL, err := req.URL.Parse(loc)
		if err != nil {
			return resp, respBody, nil
		}

		var body io.Reader
		if bodyBytes != nil {
			body = bytes.NewReader(bodyBytes)
		}
		req, err = http.NewRequestWithContext(req.Context(), method, redirectURL.String(), body)
		if err != nil {
			return nil, nil, err
		}
		c.setHeaders(req, nil, bodyBytes != nil)
	}
	return nil, nil, fmt.Errorf("too many redirects")
}

func (c *Client) resolveURL(path string) string {
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return path
	}
	return c.baseURL + strings.TrimLeft(path, "/")
}

func (c *Client) setHeaders(req *http.Request, extra map[string]string, hasBody bool) {
	req.Header.Set("Accept", "application/json")
	if hasBody {
		req.Header.Set("Content-Type", "application/json")
	}

	if c.apiKey != "" {
		token := c.apiKey
		if c.apiKeyPrefix != "" {
			token = c.apiKeyPrefix + " " + c.apiKey
		}
		req.Header.Set(c.apiKeyHeader, token)
	}

	if c.sessionCookieValue != "" {
		name := c.sessionCookieName
		if name == "" {
			name = "session"
		}
		req.AddCookie(&http.Cookie{Name: name, Value: c.sessionCookieValue})
	}

	for k, v := range extra {
		req.Header.Set(k, v)
	}
}

func isRetryable(status int) bool {
	return status == 429 || (status >= 500 && status <= 504)
}

// networkError is an internal transport error.
type networkError struct {
	msg   string
	cause error
}

func (e *networkError) Error() string {
	if e.cause != nil {
		return fmt.Sprintf("%s: %v", e.msg, e.cause)
	}
	return e.msg
}

func (e *networkError) Unwrap() error { return e.cause }

func toAPIError(statusCode int, body []byte, requestID string) error {
	message := http.StatusText(statusCode)
	var parsed map[string]any
	var responseText string

	if err := json.Unmarshal(body, &parsed); err == nil {
		if detail, ok := parsed["detail"]; ok {
			message = fmt.Sprintf("%v", detail)
		} else if msg, ok := parsed["message"]; ok {
			message = fmt.Sprintf("%v", msg)
		}
	} else {
		responseText = string(body)
	}

	details := apiErrorDetails{
		StatusCode:   statusCode,
		Message:      message,
		ResponseJSON: parsed,
		ResponseText: responseText,
		RequestID:    requestID,
	}

	return newTypedError(details)
}

type apiErrorDetails struct {
	StatusCode   int
	Message      string
	ResponseJSON map[string]any
	ResponseText string
	RequestID    string
}

type apiError struct {
	details apiErrorDetails
}

func (e *apiError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.details.StatusCode, e.details.Message)
}

func newTypedError(d apiErrorDetails) error {
	base := &apiError{details: d}
	switch {
	case d.StatusCode == 401:
		return &authenticationError{apiError: base}
	case d.StatusCode == 403:
		return &permissionDeniedError{apiError: base}
	case d.StatusCode == 404:
		return &notFoundError{apiError: base}
	case d.StatusCode == 429:
		return &rateLimitError{apiError: base}
	case d.StatusCode == 400 || d.StatusCode == 422:
		return &validationError{apiError: base}
	case d.StatusCode >= 500:
		return &serverError{apiError: base}
	default:
		return base
	}
}

type authenticationError struct{ *apiError }
type permissionDeniedError struct{ *apiError }
type notFoundError struct{ *apiError }
type rateLimitError struct{ *apiError }
type validationError struct{ *apiError }
type serverError struct{ *apiError }
