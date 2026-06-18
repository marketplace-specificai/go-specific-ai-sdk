package tracing

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
)

// Record is the payload sent to the tracing endpoint.
type Record struct {
	ModelName         string    `json:"modelname"`
	Prompt            string    `json:"prompt"`
	Response          any       `json:"response"`
	UsecaseName       string    `json:"usecase_name"`
	UsecaseGroup      string    `json:"usecase_group"`
	Datasets          []string  `json:"datasets"`
	ResponseTime      float64   `json:"response_time"`
	IsFromOptuneModel bool      `json:"is_from_optune_model"`
	InferenceError    *string   `json:"inference_error,omitempty"`
	Logprobs          []float64 `json:"logprobs,omitempty"`
	Probs             []float64 `json:"probs,omitempty"`
	RawLogits         []float64 `json:"raw_logits,omitempty"`
}

// Collector sends trace records to the SpecificAI platform in the background.
type Collector struct {
	endpoint   string
	httpClient *http.Client
	ch         chan Record
	wg         sync.WaitGroup
	done       chan struct{}
}

// New creates a new trace collector that posts to {baseURL}/public/api/collect_raw_data.
func New(baseURL string) *Collector {
	endpoint := baseURL + "/public/api/collect_raw_data"
	c := &Collector{
		endpoint:   endpoint,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		ch:         make(chan Record, 256),
		done:       make(chan struct{}),
	}
	c.wg.Add(1)
	go c.worker()
	return c
}

// Collect queues a trace record for background dispatch.
func (c *Collector) Collect(r Record) {
	if r.Datasets == nil {
		r.Datasets = []string{}
	}
	select {
	case c.ch <- r:
	default:
		log.Println("specificai: tracing channel full, dropping record")
	}
}

// Close flushes pending traces and shuts down the worker.
func (c *Collector) Close() {
	close(c.done)
	c.wg.Wait()
}

func (c *Collector) worker() {
	defer c.wg.Done()
	for {
		select {
		case r := <-c.ch:
			c.send(r)
		case <-c.done:
			// Drain remaining records.
			for {
				select {
				case r := <-c.ch:
					c.send(r)
				default:
					return
				}
			}
		}
	}
}

func (c *Collector) send(r Record) {
	body, err := json.Marshal(r)
	if err != nil {
		log.Printf("specificai: tracing marshal error: %v", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		log.Printf("specificai: tracing request error: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		log.Printf("specificai: tracing send error: %v", err)
		return
	}
	resp.Body.Close()
}
