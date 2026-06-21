package httpclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNew_RequiresBaseURL(t *testing.T) {
	_, err := New("")
	if err == nil {
		t.Fatal("expected error for empty base URL")
	}
}

func TestNew_TrailingSlash(t *testing.T) {
	c, err := New("https://example.com")
	if err != nil {
		t.Fatal(err)
	}
	if c.BaseURL() != "https://example.com/" {
		t.Fatalf("expected trailing slash, got %s", c.BaseURL())
	}

	c2, err := New("https://example.com/")
	if err != nil {
		t.Fatal(err)
	}
	if c2.BaseURL() != "https://example.com/" {
		t.Fatalf("expected single trailing slash, got %s", c2.BaseURL())
	}
}

func TestDo_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %s", r.Header.Get("Authorization"))
		}
		w.WriteHeader(200)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithAPIKey("test-key"))
	body, status, err := c.Do(context.Background(), RequestParams{
		Method: "POST", Path: "/test",
		JSONBody: map[string]string{"hello": "world"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if status != 200 {
		t.Fatalf("expected 200, got %d", status)
	}
	var resp map[string]string
	json.Unmarshal(body, &resp)
	if resp["status"] != "ok" {
		t.Fatalf("unexpected response: %v", resp)
	}
}

func TestDo_APIError401(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
		json.NewEncoder(w).Encode(map[string]string{"detail": "Unauthorized"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithAPIKey("bad-key"), WithMaxRetries(0))
	_, _, err := c.Do(context.Background(), RequestParams{Method: "GET", Path: "/test"})
	if err == nil {
		t.Fatal("expected error")
	}

	var authErr *authenticationError
	if _, ok := err.(*authenticationError); !ok {
		t.Fatalf("expected authenticationError, got %T: %v", err, authErr)
	}
}

func TestDo_APIError404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
		json.NewEncoder(w).Encode(map[string]string{"detail": "Not found"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithMaxRetries(0))
	_, _, err := c.Do(context.Background(), RequestParams{Method: "GET", Path: "/test"})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, ok := err.(*notFoundError); !ok {
		t.Fatalf("expected notFoundError, got %T", err)
	}
}

func TestDo_Retries429(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts <= 2 {
			w.WriteHeader(429)
			json.NewEncoder(w).Encode(map[string]string{"detail": "rate limited"})
			return
		}
		w.WriteHeader(200)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithMaxRetries(3), WithBackoffFactor(0.01))
	_, status, err := c.Do(context.Background(), RequestParams{Method: "GET", Path: "/test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if status != 200 {
		t.Fatalf("expected 200, got %d", status)
	}
	if attempts != 3 {
		t.Fatalf("expected 3 attempts, got %d", attempts)
	}
}

func TestDo_Retries500(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts <= 1 {
			w.WriteHeader(500)
			json.NewEncoder(w).Encode(map[string]string{"message": "server error"})
			return
		}
		w.WriteHeader(200)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithMaxRetries(2), WithBackoffFactor(0.01))
	_, _, err := c.Do(context.Background(), RequestParams{Method: "GET", Path: "/test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDo_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithTimeout(50*time.Millisecond), WithMaxRetries(0))
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, _, err := c.Do(ctx, RequestParams{Method: "GET", Path: "/test"})
	if err == nil {
		t.Fatal("expected error from timeout")
	}
}

func TestDoJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"success": true, "count": 42})
	}))
	defer srv.Close()

	c, _ := New(srv.URL)
	var resp map[string]any
	err := c.DoJSON(context.Background(), RequestParams{Method: "GET", Path: "/test"}, &resp)
	if err != nil {
		t.Fatal(err)
	}
	if resp["success"] != true {
		t.Fatal("expected success=true")
	}
}

func TestDo_RedirectPreservesMethod(t *testing.T) {
	// Simulates a 301 redirect (Go's default client would change POST to GET,
	// causing a 405 on POST-only backend routes).
	var finalMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/old" {
			w.Header().Set("Location", "/new")
			w.WriteHeader(301)
			return
		}
		finalMethod = r.Method
		w.WriteHeader(200)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithMaxRetries(0))
	_, status, err := c.Do(context.Background(), RequestParams{
		Method: "POST", Path: "/old", JSONBody: map[string]string{"key": "val"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if status != 200 {
		t.Fatalf("expected 200, got %d", status)
	}
	if finalMethod != "POST" {
		t.Fatalf("redirect changed method to %s, expected POST to be preserved", finalMethod)
	}
}

func TestSessionCookie(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cookie, err := r.Cookie("my_session")
		if err != nil || cookie.Value != "abc123" {
			t.Errorf("expected session cookie, got err=%v cookie=%v", err, cookie)
		}
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c, _ := New(srv.URL, WithSessionCookie("my_session", "abc123"))
	_, _, err := c.Do(context.Background(), RequestParams{Method: "GET", Path: "/test"})
	if err != nil {
		t.Fatal(err)
	}
}
