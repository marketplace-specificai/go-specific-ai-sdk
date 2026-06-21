package platform

import (
	"github.com/marketplace-specificai/go-specific-ai-sdk/internal/httpclient"
)

const defaultClientID = "40b3a00e282ee2db1338c3bba881b99125fc3ba1d7f5c29232ce1ad63fab1e75"

// Client is the high-level platform client for managing tasks, datasets, trainings, and models.
type Client struct {
	http     *httpclient.Client
	clientID string

	Tasks     *TaskManager
	Assets    *AssetsManager
	Trainings *TrainingManager
	Models    *ModelManager
}

// NewClient creates a new platform client backed by the given HTTP client.
func NewClient(httpClient *httpclient.Client) *Client {
	c := &Client{
		http:     httpClient,
		clientID: defaultClientID,
	}
	c.Tasks = &TaskManager{client: httpClient, clientID: c.clientID}
	c.Assets = &AssetsManager{client: httpClient, clientID: c.clientID}
	c.Trainings = &TrainingManager{client: httpClient, clientID: c.clientID, tasks: c.Tasks}
	c.Models = &ModelManager{client: httpClient, clientID: c.clientID}
	return c
}

// ClientID returns the internal tenant identifier.
func (c *Client) ClientID() string { return c.clientID }
