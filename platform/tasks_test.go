package platform

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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
		if r.URL.Path == "/name-limits" {
			http.NotFound(w, r)
			return
		}
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

func TestTaskManager_Create_RefusesTaskNameOverPlatformLimit(t *testing.T) {
	created := false
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/name-limits":
			json.NewEncoder(w).Encode(map[string]any{"task_name_max_length": 10, "label_name_max_length": 100})
		case "/create_llm_usecase":
			created = true
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		}
	})
	tm := &TaskManager{client: newTestClient(handler), clientID: defaultClientID}

	_, err := tm.Create(context.Background(), "proj", []TaskCreate{{TaskName: "eleven-char"}}, "")
	if err == nil || created {
		t.Fatalf("expected the name to be refused before creating, err=%v created=%v", err, created)
	}
	if want := "at most 10 characters"; !strings.Contains(err.Error(), want) {
		t.Fatalf("expected %q in %q", want, err.Error())
	}

	if _, err := tm.Create(context.Background(), "proj", []TaskCreate{{TaskName: "ten-chars!"}}, ""); err != nil || !created {
		t.Fatalf("expected a name at the limit to be created, err=%v created=%v", err, created)
	}
}

func TestTaskManager_Create_LeavesNamesToPlatformWithoutLimitsEndpoint(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/name-limits" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	})
	tm := &TaskManager{client: newTestClient(handler), clientID: defaultClientID}

	long := strings.Repeat("x", 500)
	if _, err := tm.Create(context.Background(), "proj", []TaskCreate{{TaskName: long}}, ""); err != nil {
		t.Fatalf("expected an older platform to decide, got %v", err)
	}
}

func TestTaskManager_Edit_KeepsStoredNameButRefusesNewLongName(t *testing.T) {
	stored := strings.Repeat("s", 20)
	updated := false
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/name-limits":
			json.NewEncoder(w).Encode(map[string]any{"task_name_max_length": 10, "label_name_max_length": 100})
		case "/llm_usecase_info":
			json.NewEncoder(w).Encode(map[string]any{"llm_usecase": map[string]any{
				"_id": "t1", "usecase_name": stored, "task_type": "ClassificationResponse",
			}})
		case "/update_llm_usecase":
			updated = true
			json.NewEncoder(w).Encode(map[string]any{"success": true})
		}
	})
	tm := &TaskManager{client: newTestClient(handler), clientID: defaultClientID}

	tooLong := strings.Repeat("n", 11)
	if _, err := tm.Edit(context.Background(), "t1", EditParams{TaskName: &tooLong}); err == nil || updated {
		t.Fatalf("expected a new long name to be refused, err=%v updated=%v", err, updated)
	}
	if _, err := tm.Edit(context.Background(), "t1", EditParams{TaskName: &stored}); err != nil || !updated {
		t.Fatalf("expected the stored name to be kept, err=%v updated=%v", err, updated)
	}
}
