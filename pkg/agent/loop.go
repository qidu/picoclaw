// PicoClaw - Ultra-lightweight personal AI agent
// Inspired by and based on nanobot: https://github.com/HKUDS/nanobot
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors
//
// Refactored based on OpenClaw patterns:
// - Run management (queue, abort, status)
// - Session lanes
// - History limiting with context window guard
// - Session compaction
// - Modular architecture

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/constants"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/session"
	"github.com/sipeed/picoclaw/pkg/state"
	"github.com/sipeed/picoclaw/pkg/tools"
	"github.com/sipeed/picoclaw/pkg/utils"
)

// AgentLoop manages the core agent execution loop with OpenClaw-inspired patterns.
type AgentLoop struct {
	bus            *bus.MessageBus
	provider       providers.LLMProvider
	workspace      string
	model          string
	contextWindow  int // Maximum context window size in tokens
	maxIterations  int
	sessions       *session.SessionManager
	state          *state.Manager
	contextBuilder *ContextBuilder
	tools          *tools.ToolRegistry
	running        atomic.Bool
	summarizing    sync.Map // Tracks which sessions are currently being summarized

	// Run management (OpenClaw pattern)
	runManager *RunManager

	// Session lanes (OpenClaw pattern)
	lanes *LaneManager
}

// RunManager manages active runs with queue and abort support.
type RunManager struct {
	mu       sync.RWMutex
	active   map[string]*ActiveRun
	queue    chan *RunRequest
	aborted  map[string]bool
	stats    RunStats
}

type ActiveRun struct {
	ID          string
	SessionKey  string
	StartTime   time.Time
	Status      RunStatus
	Chan        chan RunEvent
}

type RunRequest struct {
	SessionKey string
	Channel    string
	ChatID     string
	Message    string
	Opts       processOptions
	ResultChan chan RunResult
}

type RunEvent struct {
	Type    string
	Message string
}

type RunResult struct {
	Response string
	Error    error
}

type RunStatus string

const (
	RunStatusPending   RunStatus = "pending"
	RunStatusRunning   RunStatus = "running"
	RunStatusComplete  RunStatus = "complete"
	RunStatusAborted  RunStatus = "aborted"
	RunStatusFailed   RunStatus = "failed"
)

type RunStats struct {
	TotalRuns   int64
	ActiveRuns  int64
	QueuedRuns  int64
	AbortedRuns int64
}

// LaneManager manages session lanes for isolation (OpenClaw pattern).
type LaneManager struct {
	mu      sync.RWMutex
	lanes   map[string]*Lane
	defaultLane *Lane
}

type Lane struct {
	ID            string
	Name          string
	Sessions      map[string]*LaneSession
	Priority      int
	MaxConcurrent int
	mu            sync.RWMutex
}

type LaneSession struct {
	SessionKey string
	Active     bool
	LastActive time.Time
}

// processOptions configures how a message is processed
type processOptions struct {
	SessionKey      string // Session identifier for history/context
	Lane            string // Session lane for isolation
	Channel         string // Target channel for tool execution
	ChatID          string // Target chat ID for tool execution
	UserMessage     string // User message content (may include prefix)
	DefaultResponse string // Response when LLM returns empty
	EnableSummary   bool   // Whether to trigger summarization
	SendResponse    bool   // Whether to send response via bus
	NoHistory       bool   // If true, don't load session history (for heartbeat)
	RunID           string // Optional run ID for tracking
	TimeoutSeconds  int    // Optional timeout override
}

// NewAgentLoop creates a new agent loop with OpenClaw-inspired features.
func NewAgentLoop(cfg *config.Config, msgBus *bus.MessageBus, provider providers.LLMProvider) *AgentLoop {
	workspace := cfg.WorkspacePath()
	os.MkdirAll(workspace, 0755)

	restrict := cfg.Agents.Defaults.RestrictToWorkspace

	// Create tool registry for main agent
	toolsRegistry := createToolRegistry(workspace, restrict, cfg, msgBus)

	// Create subagent manager with its own tool registry
	subagentManager := tools.NewSubagentManager(provider, cfg.Agents.Defaults.Model, workspace, msgBus)
	subagentTools := createToolRegistry(workspace, restrict, cfg, msgBus)
	subagentManager.SetTools(subagentTools)

	// Register spawn tool (for main agent)
	spawnTool := tools.NewSpawnTool(subagentManager)
	toolsRegistry.Register(spawnTool)

	// Register subagent tool (synchronous execution)
	subagentTool := tools.NewSubagentTool(subagentManager)
	toolsRegistry.Register(subagentTool)

	sessionsManager := session.NewSessionManager(filepath.Join(workspace, "sessions"))

	// Create state manager for atomic state persistence
	stateManager := state.NewManager(workspace)

	// Create context builder and set tools registry
	contextBuilder := NewContextBuilder(workspace)
	contextBuilder.SetToolsRegistry(toolsRegistry)

	agent := &AgentLoop{
		bus:            msgBus,
		provider:       provider,
		workspace:      workspace,
		model:          cfg.Agents.Defaults.Model,
		contextWindow:  cfg.Agents.Defaults.MaxTokens,
		maxIterations:  cfg.Agents.Defaults.MaxToolIterations,
		sessions:       sessionsManager,
		state:          stateManager,
		contextBuilder: contextBuilder,
		tools:          toolsRegistry,
		summarizing:    sync.Map{},
		runManager:     NewRunManager(),
		lanes:          NewLaneManager(),
	}

	return agent
}

// NewRunManager creates a new run manager.
func NewRunManager() *RunManager {
	rm := &RunManager{
		active:  make(map[string]*ActiveRun),
		queue:   make(chan *RunRequest, 100),
		aborted: make(map[string]bool),
		stats:   RunStats{},
	}
	go rm.processQueue()
	return rm
}

// processQueue processes run requests from the queue.
func (rm *RunManager) processQueue() {
	for req := range rm.queue {
		rm.mu.Lock()
		rm.stats.QueuedRuns--
		rm.mu.Unlock()

		// Check if run was aborted while queued
		rm.mu.RLock()
		aborted := rm.aborted[req.SessionKey]
		rm.mu.RUnlock()

		if aborted {
			req.ResultChan <- RunResult{Error: fmt.Errorf("run was aborted")}
			continue
		}

		// Execute run
		// Note: Actual execution happens in Run()
		_ = req
	}
}

// QueueRun queues a run for execution.
func (rm *RunManager) QueueRun(req *RunRequest) {
	rm.mu.Lock()
	rm.stats.QueuedRuns++
	rm.mu.Unlock()
	rm.queue <- req
}

// StartRun starts tracking a new run.
func (rm *RunManager) StartRun(runID, sessionKey string) *ActiveRun {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.stats.TotalRuns++
	rm.stats.ActiveRuns++

	run := &ActiveRun{
		ID:         runID,
		SessionKey: sessionKey,
		StartTime:  time.Now(),
		Status:     RunStatusRunning,
		Chan:       make(chan RunEvent, 10),
	}
	rm.active[runID] = run
	return run
}

// CompleteRun marks a run as complete.
func (rm *RunManager) CompleteRun(runID string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if run, ok := rm.active[runID]; ok {
		run.Status = RunStatusComplete
		close(run.Chan)
	}
	delete(rm.active, runID)
	rm.stats.ActiveRuns--
}

// AbortRun marks a run as aborted.
func (rm *RunManager) AbortRun(runID string) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.stats.AbortedRuns++
	rm.aborted[runID] = true

	if run, ok := rm.active[runID]; ok {
		run.Status = RunStatusAborted
		run.Chan <- RunEvent{Type: "aborted", Message: "Run was aborted"}
		close(run.Chan)
	}
	delete(rm.active, runID)
	rm.stats.ActiveRuns--
}

// GetActiveRun returns the currently active run for a session.
func (rm *RunManager) GetActiveRun(sessionKey string) *ActiveRun {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	for _, run := range rm.active {
		if run.SessionKey == sessionKey && run.Status == RunStatusRunning {
			return run
		}
	}
	return nil
}

// GetStats returns current run statistics.
func (rm *RunManager) GetStats() RunStats {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.stats
}

// NewLaneManager creates a new lane manager.
func NewLaneManager() *LaneManager {
	defaultLane := &Lane{
		ID:            "default",
		Name:          "Default",
		Sessions:      make(map[string]*LaneSession),
		Priority:      0,
		MaxConcurrent: 5,
	}

	return &LaneManager{
		lanes: map[string]*Lane{
			"default": defaultLane,
		},
		defaultLane: defaultLane,
	}
}

// GetLane returns a lane by ID.
func (lm *LaneManager) GetLane(laneID string) *Lane {
	lm.mu.RLock()
	defer lm.mu.RUnlock()

	if lane, ok := lm.lanes[laneID]; ok {
		return lane
	}
	return lm.defaultLane
}

// RegisterSession registers a session in a lane.
func (lm *LaneManager) RegisterSession(laneID, sessionKey string) {
	lane := lm.GetLane(laneID)
	lane.mu.Lock()
	defer lane.mu.Unlock()

	lane.Sessions[sessionKey] = &LaneSession{
		SessionKey: sessionKey,
		Active:     true,
		LastActive: time.Now(),
	}
}

// UnregisterSession removes a session from a lane.
func (lm *LaneManager) UnregisterSession(laneID, sessionKey string) {
	lane := lm.GetLane(laneID)
	lane.mu.Lock()
	defer lane.mu.Unlock()

	delete(lane.Sessions, sessionKey)
}

// GetLaneStats returns statistics for a lane.
func (lm *LaneManager) GetLaneStats(laneID string) (active, total int) {
	lane := lm.GetLane(laneID)
	lane.mu.RLock()
	defer lane.mu.RUnlock()

	active = 0
	for _, s := range lane.Sessions {
		if s.Active {
			active++
		}
	}
	total = len(lane.Sessions)
	return
}

// createToolRegistry creates a tool registry with common tools.
// This is shared between main agent and subagents.
func createToolRegistry(workspace string, restrict bool, cfg *config.Config, msgBus *bus.MessageBus) *tools.ToolRegistry {
	registry := tools.NewToolRegistry()

	// File system tools
	registry.Register(tools.NewReadFileTool(workspace, restrict))
	registry.Register(tools.NewWriteFileTool(workspace, restrict))
	registry.Register(tools.NewListDirTool(workspace, restrict))
	registry.Register(tools.NewEditFileTool(workspace, restrict))
	registry.Register(tools.NewAppendFileTool(workspace, restrict))

	// Shell execution
	registry.Register(tools.NewExecTool(workspace, restrict))

	if searchTool := tools.NewWebSearchTool(tools.WebSearchToolOptions{
		BraveAPIKey:          cfg.Tools.Web.Brave.APIKey,
		BraveMaxResults:      cfg.Tools.Web.Brave.MaxResults,
		BraveEnabled:         cfg.Tools.Web.Brave.Enabled,
		DuckDuckGoMaxResults: cfg.Tools.Web.DuckDuckGo.MaxResults,
		DuckDuckGoEnabled:    cfg.Tools.Web.DuckDuckGo.Enabled,
	}); searchTool != nil {
		registry.Register(searchTool)
	}
	registry.Register(tools.NewWebFetchTool(50000))

	// Hardware tools (I2C, SPI) - Linux only, returns error on other platforms
	registry.Register(tools.NewI2CTool())
	registry.Register(tools.NewSPITool())

	// Message tool - available to both agent and subagent
	messageTool := tools.NewMessageTool()
	messageTool.SetSendCallback(func(channel, chatID, content string) error {
		msgBus.PublishOutbound(bus.OutboundMessage{
			Channel: channel,
			ChatID:  chatID,
			Content: content,
		})
		return nil
	})
	registry.Register(messageTool)

	return registry
}

func NewAgentLoopSimple(cfg *config.Config, msgBus *bus.MessageBus, provider providers.LLMProvider) *AgentLoop {
	return NewAgentLoop(cfg, msgBus, provider)
}

func (al *AgentLoop) Run(ctx context.Context) error {
	al.running.Store(true)

	// Start run manager processing
	go al.runManager.processQueue()

	for al.running.Load() {
		select {
		case <-ctx.Done():
			return nil
		default:
			msg, ok := al.bus.ConsumeInbound(ctx)
			if !ok {
				continue
			}

			// Start tracking this run
			runID := utils.GenerateRunID()
			run := al.runManager.StartRun(runID, msg.SessionKey)
			defer al.runManager.CompleteRun(runID)

			// Send run start event
			run.Chan <- RunEvent{Type: "started", Message: fmt.Sprintf("Run %s started", runID)}

			response, err := al.processMessage(ctx, msg)
			if err != nil {
				response = fmt.Sprintf("Error processing message: %v", err)
				run.Chan <- RunEvent{Type: "error", Message: err.Error()}
			} else {
				run.Chan <- RunEvent{Type: "complete", Message: "Run completed successfully"}
			}

			if response != "" {
				// Check if the message tool already sent a response
				alreadySent := false
				if tool, ok := al.tools.Get("message"); ok {
					if mt, ok := tool.(*tools.MessageTool); ok {
						alreadySent = mt.HasSentInRound()
					}
				}

				if !alreadySent {
					al.bus.PublishOutbound(bus.OutboundMessage{
						Channel: msg.Channel,
						ChatID:  msg.ChatID,
						Content: response,
					})
				}
			}
		}
	}

	return nil
}

func (al *AgentLoop) Stop() {
	al.running.Store(false)
}

func (al *AgentLoop) RegisterTool(tool tools.Tool) {
	al.tools.Register(tool)
}

// RecordLastChannel records the last active channel for this workspace.
func (al *AgentLoop) RecordLastChannel(channel string) error {
	return al.state.SetLastChannel(channel)
}

// RecordLastChatID records the last active chat ID for this workspace.
func (al *AgentLoop) RecordLastChatID(chatID string) error {
	return al.state.SetLastChatID(chatID)
}

func (al *AgentLoop) ProcessDirect(ctx context.Context, content, sessionKey string) (string, error) {
	return al.ProcessDirectWithChannel(ctx, content, sessionKey, "cli", "direct")
}

func (al *AgentLoop) ProcessDirectWithChannel(ctx context.Context, content, sessionKey, channel, chatID string) (string, error) {
	msg := bus.InboundMessage{
		Channel:    channel,
		SenderID:   "cron",
		ChatID:     chatID,
		Content:    content,
		SessionKey: sessionKey,
	}

	return al.processMessage(ctx, msg)
}

// ProcessHeartbeat processes a heartbeat request without session history.
func (al *AgentLoop) ProcessHeartbeat(ctx context.Context, content, channel, chatID string) (string, error) {
	return al.runAgentLoop(ctx, processOptions{
		SessionKey:      "heartbeat",
		Channel:         channel,
		ChatID:          chatID,
		UserMessage:     content,
		DefaultResponse: "I've completed processing but have no response to give.",
		EnableSummary:   false,
		SendResponse:    false,
		NoHistory:       true,
	})
}

func (al *AgentLoop) processMessage(ctx context.Context, msg bus.InboundMessage) (string, error) {
	// Add message preview to log
	var logContent string
	if strings.Contains(msg.Content, "Error:") || strings.Contains(msg.Content, "error") {
		logContent = msg.Content
	} else {
		logContent = utils.Truncate(msg.Content, 80)
	}
	logger.InfoCF("agent", fmt.Sprintf("Processing message from %s:%s: %s", msg.Channel, msg.SenderID, logContent),
		map[string]interface{}{
			"channel":     msg.Channel,
			"chat_id":     msg.ChatID,
			"sender_id":   msg.SenderID,
			"session_key": msg.SessionKey,
		})

	// Route system messages to processSystemMessage
	if msg.Channel == "system" {
		return al.processSystemMessage(ctx, msg)
	}

	// Process as user message
	return al.runAgentLoop(ctx, processOptions{
		SessionKey:      msg.SessionKey,
		Channel:         msg.Channel,
		ChatID:          msg.ChatID,
		UserMessage:     msg.Content,
		DefaultResponse: "I've completed processing but have no response to give.",
		EnableSummary:   true,
		SendResponse:    false,
	})
}

func (al *AgentLoop) processSystemMessage(ctx context.Context, msg bus.InboundMessage) (string, error) {
	if msg.Channel != "system" {
		return "", fmt.Errorf("processSystemMessage called with non-system message channel: %s", msg.Channel)
	}

	logger.InfoCF("agent", "Processing system message",
		map[string]interface{}{
			"sender_id": msg.SenderID,
			"chat_id":   msg.ChatID,
		})

	// Parse origin channel from chat_id (format: "channel:chat_id")
	var originChannel string
	if idx := strings.Index(msg.ChatID, ":"); idx > 0 {
		originChannel = msg.ChatID[:idx]
	} else {
		originChannel = "cli"
	}

	// Extract subagent result from message content
	content := msg.Content
	if idx := strings.Index(content, "Result:\n"); idx >= 0 {
		content = content[idx+8:]
	}

	// Skip internal channels - only log, don't send to user
	if constants.IsInternalChannel(originChannel) {
		logger.InfoCF("agent", "Subagent completed (internal channel)",
			map[string]interface{}{
				"sender_id":   msg.SenderID,
				"content_len": len(content),
				"channel":     originChannel,
			})
		return "", nil
	}

	logger.InfoCF("agent", "Subagent completed",
		map[string]interface{}{
			"sender_id":   msg.SenderID,
			"channel":     originChannel,
			"content_len": len(content),
		})

	return "", nil
}

// runAgentLoop is the core message processing logic.
// Handles context building, LLM calls, tool execution, and response handling.
func (al *AgentLoop) runAgentLoop(ctx context.Context, opts processOptions) (string, error) {
	// 0. Record last channel for heartbeat notifications (skip internal channels)
	if opts.Channel != "" && opts.ChatID != "" {
		if !constants.IsInternalChannel(opts.Channel) {
			channelKey := fmt.Sprintf("%s:%s", opts.Channel, opts.ChatID)
			if err := al.RecordLastChannel(channelKey); err != nil {
				logger.WarnCF("agent", "Failed to record last channel: %v", map[string]interface{}{"error": err.Error()})
			}
		}
	}

	// 1. Update tool contexts
	al.updateToolContexts(opts.Channel, opts.ChatID)

	// 2. Build messages with history limiting (OpenClaw pattern: limitHistoryTurns)
	var history []providers.Message
	var summary string
	if !opts.NoHistory {
		history = al.sessions.GetHistory(opts.SessionKey)
		summary = al.sessions.GetSummary(opts.SessionKey)

		// Apply history limiting based on context window
		history = al.limitHistoryTurns(history, al.contextWindow)
	}

	messages := al.contextBuilder.BuildMessages(
		history,
		summary,
		opts.UserMessage,
		nil,
		opts.Channel,
		opts.ChatID,
	)

	// 3. Save user message to session
	al.sessions.AddMessage(opts.SessionKey, "user", opts.UserMessage)

	// 4. Run LLM iteration loop
	finalContent, iteration, err := al.runLLMIteration(ctx, messages, opts)
	if err != nil {
		return "", err
	}

	// 5. Handle empty response
	if finalContent == "" {
		finalContent = opts.DefaultResponse
	}

	// 6. Save final assistant message to session
	al.sessions.AddMessage(opts.SessionKey, "assistant", finalContent)
	al.sessions.Save(opts.SessionKey)

	// 7. Optional: summarization (OpenClaw pattern: compaction)
	if opts.EnableSummary {
		al.maybeSummarize(opts.SessionKey)
	}

	// 8. Optional: send response via bus
	if opts.SendResponse {
		al.bus.PublishOutbound(bus.OutboundMessage{
			Channel: opts.Channel,
			ChatID:  opts.ChatID,
			Content: finalContent,
		})
	}

	// 9. Log response
	responsePreview := utils.Truncate(finalContent, 120)
	logger.InfoCF("agent", fmt.Sprintf("Response: %s", responsePreview),
		map[string]interface{}{
			"session_key":  opts.SessionKey,
			"iterations":   iteration,
			"final_length": len(finalContent),
		})

	return finalContent, nil
}

// limitHistoryTurns limits the history turns based on context window (OpenClaw pattern).
func (al *AgentLoop) limitHistoryTurns(messages []providers.Message, maxTokens int) []providers.Message {
	if len(messages) == 0 {
		return messages
	}

	// Estimate tokens used by messages
	tokenCount := 0
	for _, msg := range messages {
		tokenCount += len(msg.Content) / 4
	}

	// If we're under the limit, return as-is
	if tokenCount < maxTokens*75/100 {
		return messages
	}

	// Count message pairs (user + assistant)
	// Keep pairs until we hit the limit
	var limited []providers.Message
	pairCount := 0

	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		msgTokens := len(msg.Content) / 4

		// If this is an assistant message, start counting a new pair
		if msg.Role == "assistant" {
			pairCount++
		}

		// Add messages if we're within limits
		if tokenCount+msgTokens < maxTokens*70/100 {
			tokenCount += msgTokens
			limited = append([]providers.Message{msg}, limited...)
		} else {
			break
		}

		// Stop after keeping enough pairs (leave room for new message)
		if pairCount >= 20 {
			break
		}
	}

	return limited
}

// runLLMIteration executes the LLM call loop with tool handling.
func (al *AgentLoop) runLLMIteration(ctx context.Context, messages []providers.Message, opts processOptions) (string, int, error) {
	iteration := 0
	var finalContent string

	for iteration < al.maxIterations {
		iteration++

		logger.DebugCF("agent", "LLM iteration",
			map[string]interface{}{
				"iteration": iteration,
				"max":       al.maxIterations,
			})

		// Build tool definitions
		providerToolDefs := al.tools.ToProviderDefs()

		// Log LLM request details
		logger.DebugCF("agent", "LLM request",
			map[string]interface{}{
				"iteration":         iteration,
				"model":             al.model,
				"messages_count":    len(messages),
				"tools_count":       len(providerToolDefs),
				"max_tokens":        8192,
				"temperature":       0.7,
				"system_prompt_len": len(messages[0].Content),
			})

		// Call LLM
		response, err := al.provider.Chat(ctx, messages, providerToolDefs, al.model, map[string]interface{}{
			"max_tokens":  8192,
			"temperature": 0.7,
		})

		if err != nil {
			logger.ErrorCF("agent", "LLM call failed",
				map[string]interface{}{
					"iteration": iteration,
					"error":     err.Error(),
				})
			return "", iteration, fmt.Errorf("LLM call failed: %w", err)
		}

		// Check if no tool calls - we're done
		if len(response.ToolCalls) == 0 {
			finalContent = response.Content
			logger.InfoCF("agent", "LLM response without tool calls (direct answer)",
				map[string]interface{}{
					"iteration":     iteration,
					"content_chars": len(finalContent),
				})
			break
		}

		// Log tool calls
		toolNames := make([]string, 0, len(response.ToolCalls))
		for _, tc := range response.ToolCalls {
			toolNames = append(toolNames, tc.Name)
		}
		logger.InfoCF("agent", "LLM requested tool calls",
			map[string]interface{}{
				"tools":     toolNames,
				"count":     len(response.ToolCalls),
				"iteration": iteration,
			})

		// Build assistant message with tool calls
		assistantMsg := providers.Message{
			Role:    "assistant",
			Content: response.Content,
		}
		for _, tc := range response.ToolCalls {
			argumentsJSON, _ := json.Marshal(tc.Arguments)
			assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, providers.ToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: &providers.FunctionCall{
					Name:      tc.Name,
					Arguments: string(argumentsJSON),
				},
			})
		}
		messages = append(messages, assistantMsg)

		// Save assistant message with tool calls to session
		al.sessions.AddFullMessage(opts.SessionKey, assistantMsg)

		// Execute tool calls
		for _, tc := range response.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Arguments)
			argsPreview := utils.Truncate(string(argsJSON), 200)
			logger.InfoCF("agent", fmt.Sprintf("Tool call: %s(%s)", tc.Name, argsPreview),
				map[string]interface{}{
					"tool":      tc.Name,
					"iteration": iteration,
				})

			asyncCallback := func(callbackCtx context.Context, result *tools.ToolResult) {
				if !result.Silent && result.ForUser != "" {
					logger.InfoCF("agent", "Async tool completed, agent will handle notification",
						map[string]interface{}{
							"tool":        tc.Name,
							"content_len": len(result.ForUser),
						})
				}
			}

			toolResult := al.tools.ExecuteWithContext(ctx, tc.Name, tc.Arguments, opts.Channel, opts.ChatID, asyncCallback)

			// Send ForUser content to user immediately if not Silent
			if !toolResult.Silent && toolResult.ForUser != "" && opts.SendResponse {
				al.bus.PublishOutbound(bus.OutboundMessage{
					Channel: opts.Channel,
					ChatID:  opts.ChatID,
					Content: toolResult.ForUser,
				})
				logger.DebugCF("agent", "Sent tool result to user",
					map[string]interface{}{
						"tool":        tc.Name,
						"content_len": len(toolResult.ForUser),
					})
			}

			// Determine content for LLM
			contentForLLM := toolResult.ForLLM
			if contentForLLM == "" && toolResult.Err != nil {
				contentForLLM = toolResult.Err.Error()
			}

			toolResultMsg := providers.Message{
				Role:       "tool",
				Content:    contentForLLM,
				ToolCallID: tc.ID,
			}
			messages = append(messages, toolResultMsg)

			// Save tool result message to session
			al.sessions.AddFullMessage(opts.SessionKey, toolResultMsg)
		}
	}

	return finalContent, iteration, nil
}

// updateToolContexts updates the context for tools that need channel/chatID info.
func (al *AgentLoop) updateToolContexts(channel, chatID string) {
	if tool, ok := al.tools.Get("message"); ok {
		if mt, ok := tool.(tools.ContextualTool); ok {
			mt.SetContext(channel, chatID)
		}
	}
	if tool, ok := al.tools.Get("spawn"); ok {
		if st, ok := tool.(tools.ContextualTool); ok {
			st.SetContext(channel, chatID)
		}
	}
	if tool, ok := al.tools.Get("subagent"); ok {
		if st, ok := tool.(tools.ContextualTool); ok {
			st.SetContext(channel, chatID)
		}
	}
}

// maybeSummarize triggers summarization if the session history exceeds thresholds.
func (al *AgentLoop) maybeSummarize(sessionKey string) {
	newHistory := al.sessions.GetHistory(sessionKey)
	tokenEstimate := al.estimateTokens(newHistory)
	threshold := al.contextWindow * 75 / 100

	if len(newHistory) > 20 || tokenEstimate > threshold {
		if _, loading := al.summarizing.LoadOrStore(sessionKey, true); !loading {
			go func() {
				defer al.summarizing.Delete(sessionKey)
				al.summarizeSession(sessionKey)
			}()
		}
	}
}

// GetStartupInfo returns information about loaded tools and skills for logging.
func (al *AgentLoop) GetStartupInfo() map[string]interface{} {
	info := make(map[string]interface{})

	tools := al.tools.List()
	info["tools"] = map[string]interface{}{
		"count": len(tools),
		"names": tools,
	}

	info["skills"] = al.contextBuilder.GetSkillsInfo()

	return info
}

// GetRunStats returns current run statistics.
func (al *AgentLoop) GetRunStats() RunStats {
	return al.runManager.GetStats()
}

// GetLaneStats returns statistics for a lane.
func (al *AgentLoop) GetLaneStats(laneID string) (active, total int) {
	return al.lanes.GetLaneStats(laneID)
}

// formatMessagesForLog formats messages for logging
func formatMessagesForLog(messages []providers.Message) string {
	if len(messages) == 0 {
		return "[]"
	}

	var result string
	result += "[\n"
	for i, msg := range messages {
		result += fmt.Sprintf("  [%d] Role: %s\n", i, msg.Role)
		if msg.ToolCalls != nil && len(msg.ToolCalls) > 0 {
			result += "  ToolCalls:\n"
			for _, tc := range msg.ToolCalls {
				result += fmt.Sprintf("    - ID: %s, Type: %s, Name: %s\n", tc.ID, tc.Type, tc.Name)
				if tc.Function != nil {
					result += fmt.Sprintf("      Arguments: %s\n", utils.Truncate(tc.Function.Arguments, 200))
				}
			}
		}
		if msg.Content != "" {
			content := utils.Truncate(msg.Content, 200)
			result += fmt.Sprintf("  Content: %s\n", content)
		}
		if msg.ToolCallID != "" {
			result += fmt.Sprintf("  ToolCallID: %s\n", msg.ToolCallID)
		}
		result += "\n"
	}
	result += "]"
	return result
}

// formatToolsForLog formats tool definitions for logging
func formatToolsForLog(tools []providers.ToolDefinition) string {
	if len(tools) == 0 {
		return "[]"
	}

	var result string
	result += "[\n"
	for i, tool := range tools {
		result += fmt.Sprintf("  [%d] Type: %s, Name: %s\n", i, tool.Type, tool.Function.Name)
		result += fmt.Sprintf("      Description: %s\n", tool.Function.Description)
		if len(tool.Function.Parameters) > 0 {
			result += fmt.Sprintf("      Parameters: %s\n", utils.Truncate(fmt.Sprintf("%v", tool.Function.Parameters), 200))
		}
	}
	result += "]"
	return result
}

// summarizeSession summarizes the conversation history for a session.
func (al *AgentLoop) summarizeSession(sessionKey string) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	history := al.sessions.GetHistory(sessionKey)
	summary := al.sessions.GetSummary(sessionKey)

	// Keep last 4 messages for continuity
	if len(history) <= 4 {
		return
	}

	toSummarize := history[:len(history)-4]

	// Oversized Message Guard
	maxMessageTokens := al.contextWindow / 2
	validMessages := make([]providers.Message, 0)
	omitted := false

	for _, m := range toSummarize {
		if m.Role != "user" && m.Role != "assistant" {
			continue
		}
		msgTokens := len(m.Content) / 4
		if msgTokens > maxMessageTokens {
			omitted = true
			continue
		}
		validMessages = append(validMessages, m)
	}

	if len(validMessages) == 0 {
		return
	}

	// Multi-Part Summarization
	var finalSummary string
	if len(validMessages) > 10 {
		mid := len(validMessages) / 2
		part1 := validMessages[:mid]
		part2 := validMessages[mid:]

		s1, _ := al.summarizeBatch(ctx, part1, "")
		s2, _ := al.summarizeBatch(ctx, part2, "")

		mergePrompt := fmt.Sprintf("Merge these two conversation summaries into one cohesive summary:\n\n1: %s\n\n2: %s", s1, s2)
		resp, err := al.provider.Chat(ctx, []providers.Message{{Role: "user", Content: mergePrompt}}, nil, al.model, map[string]interface{}{
			"max_tokens":  1024,
			"temperature": 0.3,
		})
		if err == nil {
			finalSummary = resp.Content
		} else {
			finalSummary = s1 + " " + s2
		}
	} else {
		finalSummary, _ = al.summarizeBatch(ctx, validMessages, summary)
	}

	if omitted && finalSummary != "" {
		finalSummary += "\n[Note: Some oversized messages were omitted from this summary for efficiency.]"
	}

	if finalSummary != "" {
		al.sessions.SetSummary(sessionKey, finalSummary)
		al.sessions.TruncateHistory(sessionKey, 4)
		al.sessions.Save(sessionKey)
	}
}

// summarizeBatch summarizes a batch of messages.
func (al *AgentLoop) summarizeBatch(ctx context.Context, batch []providers.Message, existingSummary string) (string, error) {
	prompt := "Provide a concise summary of this conversation segment, preserving core context and key points.\n"
	if existingSummary != "" {
		prompt += "Existing context: " + existingSummary + "\n"
	}
	prompt += "\nCONVERSATION:\n"
	for _, m := range batch {
		prompt += fmt.Sprintf("%s: %s\n", m.Role, m.Content)
	}

	response, err := al.provider.Chat(ctx, []providers.Message{{Role: "user", Content: prompt}}, nil, al.model, map[string]interface{}{
		"max_tokens":  1024,
		"temperature": 0.3,
	})
	if err != nil {
		return "", err
	}
	return response.Content, nil
}

// estimateTokens estimates the number of tokens in a message list.
func (al *AgentLoop) estimateTokens(messages []providers.Message) int {
	total := 0
	for _, m := range messages {
		total += len(m.Content) / 4
	}
	return total
}