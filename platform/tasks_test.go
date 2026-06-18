package platform

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

func newTestClient(handler http.Handler) *httpclient.Client {
	srv := httptest.NewServer(handler)
	c, _ := httpclient.New(srv.URL)
	return c
}

func TestTaskManager_List(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/llm_usecases" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["value"] != defaultClientID {
			t.Errorf("expected client_id=%s, got %v", defaultClientID, body["value"])
		}
		json.NewEncoder(w).Encode([]map[string]any{
			{
				"group_name": "my_project",
				"sub_groups": []map[string]any{
					{
						"sub_group_name": "",
						"usecases": []map[string]any{
							{"_id": "task1", "usecase_name": "Task One", "task_type": "ClassificationResponse"},
						},
					},
				},
			},
		})
	})

	hc := newTestClient(handler)
	tm := &TaskManager{client: hc, clientID: defaultClientID}

	groups, err := tm.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 {
		t.Fatalf("expected 1 group, got %d", len(groups))
	}
	if groups[0].ProjectName != "my_project" {
		t.Fatalf("expected project=my_project, got %s", groups[0].ProjectName)
	}
	tasks := groups[0].IterTasks()
	if len(tasks) != 1 {
		t.Fatalf("expected 1 task, got %d", len(tasks))
	}
	if tasks[0].TaskName != "Task One" {
		t.Fatalf("expected task name='Task One', got %s", tasks[0].TaskName)
	}
}

func TestTaskManager_Get(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"llm_usecase": map[string]any{
				"_id":          "task123",
				"usecase_name": "Test Task",
				"task_type":    "ClassificationResponse",
				"datasets":     []string{"ds1"},
			},
		})
	})

	hc := newTestClient(handler)
	tm := &TaskManager{client: hc, clientID: defaultClientID}

	task, err := tm.Get(context.Background(), "task123")
	if err != nil {
		t.Fatal(err)
	}
	if task.TaskID != "task123" {
		t.Fatalf("expected task ID=task123, got %s", task.TaskID)
	}
	if task.TaskName != "Test Task" {
		t.Fatalf("expected name='Test Task', got %s", task.TaskName)
	}
}

func TestTaskManager_Create(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["group_name"] != "my_project" {
			t.Errorf("expected group_name=my_project, got %v", body["group_name"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"success":          true,
			"created_usecases": []string{"new_id_1"},
		})
	})

	hc := newTestClient(handler)
	tm := &TaskManager{client: hc, clientID: defaultClientID}

	resp, err := tm.Create(context.Background(), "my_project", []TaskCreate{
		{TaskName: "New Task", TaskType: "ClassificationResponse"},
	}, "classify things")
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Success {
		t.Fatal("expected success=true")
	}
	if len(resp.CreatedTaskIDs) != 1 || resp.CreatedTaskIDs[0] != "new_id_1" {
		t.Fatalf("unexpected created IDs: %v", resp.CreatedTaskIDs)
	}
}

func TestTaskManager_Delete(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/delete_llm_usecase" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	})

	hc := newTestClient(handler)
	tm := &TaskManager{client: hc, clientID: defaultClientID}

	resp, err := tm.Delete(context.Background(), "task_to_delete")
	if err != nil {
		t.Fatal(err)
	}
	if resp["success"] != true {
		t.Fatal("expected success=true")
	}
}
