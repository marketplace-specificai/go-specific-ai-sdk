package tracing

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestCollector_SendsTrace(t *testing.T) {
	var received atomic.Int32
	var lastBody Record

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/public/api/collect_raw_data" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected json content type, got %s", r.Header.Get("Content-Type"))
		}
		json.NewDecoder(r.Body).Decode(&lastBody)
		received.Add(1)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c := New(srv.URL)
	c.Collect(Record{
		ModelName:    "gpt-4",
		Prompt:       "hello",
		Response:     "world",
		UsecaseName:  "test_task",
		UsecaseGroup: "test_project",
		ResponseTime: 0.5,
	})

	// Give the worker time to process.
	time.Sleep(200 * time.Millisecond)
	c.Close()

	if received.Load() != 1 {
		t.Fatalf("expected 1 request, got %d", received.Load())
	}
	if lastBody.ModelName != "gpt-4" {
		t.Fatalf("expected modelname=gpt-4, got %s", lastBody.ModelName)
	}
	if lastBody.Prompt != "hello" {
		t.Fatalf("expected prompt=hello, got %s", lastBody.Prompt)
	}
}

func TestCollector_DefaultsDatasets(t *testing.T) {
	var lastBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&lastBody)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c := New(srv.URL)
	c.Collect(Record{
		ModelName:    "test",
		UsecaseName:  "t",
		UsecaseGroup: "g",
	})
	time.Sleep(200 * time.Millisecond)
	c.Close()

	datasets, ok := lastBody["datasets"].([]any)
	if !ok {
		t.Fatalf("expected datasets array, got %T", lastBody["datasets"])
	}
	if len(datasets) != 0 {
		t.Fatalf("expected empty datasets, got %v", datasets)
	}
}

func TestCollector_GracefulShutdown(t *testing.T) {
	var count atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count.Add(1)
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c := New(srv.URL)
	for i := 0; i < 5; i++ {
		c.Collect(Record{ModelName: "test", UsecaseName: "t", UsecaseGroup: "g"})
	}
	c.Close()

	if count.Load() != 5 {
		t.Fatalf("expected 5 traces flushed, got %d", count.Load())
	}
}
