# PicoClaw vs OpenClaw - Agent Architecture Comparison

> Generated: 2026-02-16
> Author: Clawra

## Executive Summary

PicoClaw is essentially **OpenClaw's core agent loop functionality extracted and rewritten in Go** for ultra-lightweight deployment. It strips away the enterprise features (gateway, sandbox, auth profiles, compaction) to create a ~10MB RAM agent that starts in <1 second.

---

## 1. Codebase Overview

| | **PicoClaw** | **OpenClaw** |
|---|---|---|
| **Language** | Go | TypeScript |
| **Agent Loop File** | `pkg/agent/loop.go` (~700 lines) | `pi-embedded-runner.ts` + 500+ related files |
| **Agent Package** | Single `agent` package | Massive `agents/` directory (100+ files) |
| **Memory Footprint** | ~10MB RAM | >1GB RAM |
| **Startup Time** | <1 second | ~500 seconds |
| **Dependencies** | Minimal (~20 imports) | Enterprise-grade (auth, sandbox, cron, etc.) |
| **Repository Size** | ~2MB | ~50MB+ |

---

## 2. Agent Flow Comparison

### PicoClaw Flow (Simple)

```
CLI Input → AgentLoop.Run() → ContextBuilder.BuildMessages() → LLM.Chat() → Tools.Execute() → Save Session
```

**Key characteristics:**
- Single synchronous loop
- Direct LLM provider calls
- Simple JSON file session storage
- No sandbox isolation
- No subagent orchestration

### OpenClaw Flow (Complex)

```
CLI → Gateway API → Session Resolution → Auth Profiles → Sandbox → Embedded Agent → Transcript → Compaction → Lifecycle Events
```

**Key characteristics:**
- Gateway-based distributed architecture
- Multi-step auth profile rotation
- Docker container isolation
- Complex session management with compaction
- Subagent spawning with lifecycle events

---

## 3. Core Agent Loop Comparison

### PicoClaw Loop (`pkg/agent/loop.go`)

```go
func (al *AgentLoop) Run(ctx context.Context) error {
    al.running.Store(true)

    for al.running.Load() {
        select {
        case <-ctx.Done():
            return nil
        default:
            msg, ok := al.bus.ConsumeInbound(ctx)
            if !ok {
                continue
            }

            response, err := al.processMessage(ctx, msg)
            if err != nil {
                response = fmt.Sprintf("Error processing message: %v", err)
            }

            if response != "" {
                // Check if message tool already sent response
                al.bus.PublishOutbound(bus.OutboundMessage{
                    Channel: msg.Channel,
                    ChatID:  msg.ChatID,
                    Content: response,
                })
            }
        }
    }
    return nil
}
```

**Single method handles everything:**
- Message consumption from bus
- Context building
- LLM iteration loop
- Tool execution
- Session saving
- Automatic summarization

### OpenClaw Embedded Agent (`pi-embedded-runner.ts`)

```typescript
async runEmbeddedPiAgent(
    input: EmbeddedPiAgentInput
): Promise<EmbeddedPiAgentResult> {
    // 1. Build context from skills + tools
    const context = await buildContext();

    // 2. Sanitize session messages
    const sanitized = sanitizeSessionMessages(input.messages);

    // 3. Auth profile rotation
    const authProfile = await rotateAuthProfile();

    // 4. Call LLM with streaming
    const stream = await callLLMStreaming(context, authProfile);

    // 5. Handle tool calls with sandbox
    for await (const toolCall of stream) {
        const result = await executeInSandbox(toolCall);
        await sendToolResult(result);
    }

    // 6. Transcript compaction
    await compactTranscript(runId);

    // 7. Lifecycle events (spawn, announce, complete)
    await emitLifecycleEvents('complete', { runId });

    return { result: finalContent };
}
```

---

## 4. Session Management

### PicoClaw (Simple)

```
~/.picoclaw/
├── sessions/
│   └── {session-key}.jsonl
├── memory/
│   └── {date}.md
└── MEMORY.md
```

- JSONL file per session
- Simple message append
- Periodic summarization when history exceeds thresholds
- Token estimation via simple heuristic (4 chars/token)

**Session files (`pkg/session/session.go`):**
- `GetHistory()` - Load message history
- `AddMessage()` - Append user/assistant messages
- `GetSummary()` - Retrieve condensed summary
- `TruncateHistory()` - Keep last N messages

### OpenClaw (Complex)

```
~/.openclaw/agents/{agent-id}/
├── sessions/
│   └── {session-key}.jsonl
├── transcripts/
│   └── {run-id}/
│       ├── compact.jsonl     # Compressed history
│       └── raw.jsonl         # Full conversation
├── workspace/
│   └── (agent workspace files)
├── sandbox/
│   └── (Docker containers)
└── skills/
    └── (skill snapshots)
```

- Multi-layer storage (sessions, transcripts, compaction)
- Separate raw vs compacted transcripts
- Docker sandbox per session
- Skills snapshot management
- Session write locks for concurrency

---

## 5. Tool Execution

### PicoClaw (`pkg/tools/`)

```go
type ToolRegistry struct {
    tools map[string]Tool
}

type Tool interface {
    Name() string
    Description() string
    Parameters() string
    Execute(args string) ToolResult
}
```

**Built-in tools (~20):**
- `read` / `write` / `list_dir` / `edit` / `append_file`
- `exec` - Shell execution
- `web_search` / `web_fetch`
- `spawn` / `subagent` - Subagent execution
- `message` - Send to channels
- `cron` / `memory_search`
- Hardware tools: `i2c` / `spi`

**Characteristics:**
- Simple synchronous execution
- No sandbox
- Workspace restriction via `restrict` flag
- Optional async callback for long-running tools

### OpenClaw (`src/agents/tools/` + sandbox)

```typescript
type PiTools = {
    // Sandbox tools (Docker-isolated)
    sandbox: SandboxTools;

    // Execution tools
    exec: BashTools;        // With approval flows
    process: ProcessTools;

    // Agent orchestration
    agent: SubagentTools;   // Spawn with lifecycle events
    sessions: SessionsTools;

    // Communication
    channel: ChannelTools;  // Send to any channel

    // File operations
    read: ReadFileTool;
    write: WriteFileTool;
    glob: GlobTool;

    // 50+ more tools...
}
```

**Characteristics:**
- Docker sandbox execution for exec
- Subagent orchestration with lifecycle events (spawn, announce, complete)
- Complex approval flows for dangerous operations
- Tool policy enforcement
- Tool display with rich metadata

---

## 6. Auth & Configuration

### PicoClaw (`pkg/config/config.go`)

```yaml
workspace_path: ~/.picoclaw
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    base_url: https://api.anthropic.com
  openai:
    api_key: ${OPENAI_API_KEY}
  minimax:
    api_key: ${MINIMAX_API_KEY}

agents:
  defaults:
    model: claude-sonnet-4-20250514
    max_tokens: 16384
    max_tool_iterations: 20
    restrict_to_workspace: true

tools:
  web:
    brave:
      api_key: ${BRAVE_API_KEY}
      max_results: 10
      enabled: true
    duckduckgo:
      enabled: true
      max_results: 10
```

**Characteristics:**
- Single API key per provider
- Simple environment variable substitution
- Per-agent defaults
- Tool-specific configuration

### OpenClaw (`openclaw.json` or `models.json`)

```yaml
providers:
  anthropic:
    profiles:
      - name: "primary"
        api_key: ${ANTHROPIC_API_KEY_1}
        priority: 10
      - name: "fallback"
        api_key: ${ANTHROPIC_API_KEY_2}
        priority: 5
  openai:
    profiles:
      - name: "work"
        api_key: ${OPENAI_WORK_KEY}
      - name: "personal"
        api_key: ${OPENAI_PERSONAL_KEY}

agents:
  main:
    model: claude-sonnet-4-20250514
    timeout: 600
    thinking: medium
    sandbox: docker
  ops:
    model: claude-sonnet-4-20250514
    sandbox: docker
  dev:
    workspace: ~/dev-workspace
    sandbox: docker

channels:
  whatsapp:
    account: personal
    instance: default
```

**Characteristics:**
- Multiple auth profiles per provider with priority
- Profile rotation on failure
- Per-agent sandbox configuration
- Channel routing bindings
- Complex timeout and thinking configuration

---

## 7. Message Bus Architecture

### PicoClaw (`pkg/bus/`)

```go
type MessageBus struct {
    inbound  chan InboundMessage
    outbound chan OutboundMessage
    subscribers []Subscriber
}

type InboundMessage struct {
    Channel    string
    SenderID   string
    ChatID     string
    Content    string
    SessionKey string
}

type OutboundMessage struct {
    Channel string
    ChatID  string
    Content string
}
```

- Simple in-memory channel-based bus
- Direct publish/subscribe
- No persistence
- No retry logic

### OpenClaw (`src/gateway/`)

```
Gateway Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  Gateway    │────▶│   Agent     │
└─────────────┘     │  (WebSocket)│     └─────────────┘
                    │             │
                    ├─────────────┤
                    │   Client    │
                    └─────────────┘

Frame Types:
- req: Request frame
- res: Response frame
- event: Async event frame
- err: Error frame
```

- WebSocket-based gateway
- Frame-based protocol (req/res/event)
- Request/response matching via ID
- Async event delivery
- Connection state management
- Device authentication

---

## 8. Key Architectural Differences

| Aspect | PicoClaw | OpenClaw |
|--------|----------|----------|
| **Architecture** | Monolithic | Microservices via Gateway |
| **Isolation** | Workspace restriction | Docker containers |
| **Sessions** | Simple JSONL | Complex transcripts + compaction |
| **Auth** | Single API key | Auth profiles with rotation |
| **Subagents** | Basic spawn | Lifecycle events + announce queue |
| **Thinking** | Not supported | `--thinking` flag (minimal→high) |
| **Delivery** | Print to stdout | `--deliver` to channels |
| **Timeout** | Infinite | Configurable (default 600s) |
| **State** | Files only | SQLite + files + transcripts |
| **Protocol** | Direct exec | Gateway WebSocket frames |
| **Message Bus** | In-memory channels | WebSocket + frame protocol |
| **Scaling** | Single instance | Distributed via gateway |

---

## 9. File Structure Comparison

### PicoClaw (`pkg/`)

```
pkg/
├── agent/           # Agent loop, context, memory
├── bus/             # Simple message bus
├── channels/        # Channel implementations
├── config/          # Configuration loading
├── constants/       # Constants and helpers
├── cron/            # Cron job management
├── heartbeat/       # Heartbeat service
├── logger/          # Simple logging
├── migrate/         # Workspace migration
├── providers/       # LLM provider interfaces
├── session/         # Session management
├── skills/          # Skills loader/installer
├── state/           # Atomic state persistence
├── tools/           # Tool registry and implementations
└── utils/           # Utilities
```

**Total: ~15 packages, ~50 files**

### OpenClaw (`src/`)

```
src/
├── agents/          # 100+ files (auth, tools, sessions, etc.)
├── browser/         # Browser automation
├── canvas/          # Canvas rendering
├── channels/        # Channel implementations (discord, whatsapp, etc.)
├── cli/             # CLI commands
├── commands/        # Command implementations
├── config/          # Configuration
├── cron/            # Cron management
├── daemon/          # Daemon processes
├── gateway/         # Gateway protocol
├── hooks/           # Lifecycle hooks
├── infra/           # Infrastructure
├── media-understanding/
├── nodes/           # Node management
├── protocol/        # Protocol definitions
├── runtime/         # Runtime context
└── skills/          # Skills management
```

**Total: ~500+ files, ~50 directories**

---

## 10. Design Philosophy

### PicoClaw: "Simple is better"

**Target hardware:**
- Raspberry Pi
- $10-$50 single-board computers
- Embedded systems
- IoT devices

**Design decisions:**
- Ultra-lightweight (~10MB RAM)
- Fast startup (<1s)
- Single agent only
- No external dependencies
- Simple configuration
- Direct LLM calls

### OpenClaw: "Enterprise-grade"

**Target deployment:**
- Production servers
- Cloud environments
- Multi-user systems
- Enterprise security requirements

**Design decisions:**
- Full security isolation (Docker)
- Complex orchestration
- Multi-agent support
- Comprehensive lifecycle management
- Auth profile rotation
- Transcript compaction
- Rich CLI experience

---

## 11. Implementation Details

### Context Building

**PicoClaw (`pkg/agent/context.go`):**
```go
func (cb *ContextBuilder) BuildMessages(
    history []providers.Message,
    summary string,
    userMessage string,
    systemPrompt string,
    channel string,
    chatID string,
) []providers.Message {
    messages := []providers.Message{}

    // System prompt
    if systemPrompt != "" {
        messages = append(messages, providers.Message{
            Role:    "system",
            Content: systemPrompt,
        })
    }

    // Summary if exists
    if summary != "" {
        messages = append(messages, providers.Message{
            Role:    "system",
            Content: "Summary of previous conversation:\n" + summary,
        })
    }

    // History
    messages = append(messages, history...)

    // User message
    messages = append(messages, providers.Message{
        Role:    "user",
        Content: userMessage,
    })

    return messages
}
```

**OpenClaw (`src/agents/pi-embedded-context.ts`):**
```typescript
async buildContext(input: EmbeddedPiInput) {
    // Load skills
    const skillsPrompt = await loadSkillsPrompt();

    // Build tools definitions
    const tools = await createOpenClawTools(this.sandbox);

    // Get memory context
    const memoryContext = await searchMemory(input.message);

    // Load session history
    const history = await loadSessionHistory(input.sessionKey);

    // Apply context window guard
    const trimmed = applyContextWindowGuard(history, this.maxTokens);

    // Build system prompt with skills, tools, memory
    const systemPrompt = buildSystemPrompt({
        skills: skillsPrompt,
        tools,
        memory: memoryContext,
        identity: this.identity,
    });

    return { systemPrompt, messages: trimmed };
}
```

### LLM Iteration Loop

**PicoClaw (`loop.go` runLLMIteration):**
```go
func (al *AgentLoop) runLLMIteration(
    ctx context.Context,
    messages []providers.Message,
    opts processOptions,
) (string, int, error) {
    iteration := 0

    for iteration < al.maxIterations {
        iteration++

        // Build tool definitions
        providerToolDefs := al.tools.ToProviderDefs()

        // Call LLM
        response, err := al.provider.Chat(ctx, messages, providerToolDefs, al.model, nil)

        // No tool calls - done
        if len(response.ToolCalls) == 0 {
            return response.Content, iteration, nil
        }

        // Execute tool calls
        for _, tc := range response.ToolCalls {
            toolResult := al.tools.ExecuteWithContext(ctx, tc.Name, tc.Arguments, opts.Channel, opts.ChatID, nil)

            // Append tool result to messages
            messages = append(messages, providers.Message{
                Role:       "tool",
                Content:    toolResult.ForLLM,
                ToolCallID: tc.ID,
            })
        }
    }

    return finalContent, iteration, nil
}
```

**OpenClaw (`pi-embedded-runner.ts`):**
```typescript
async runLLMIteration(messages: Message[], tools: ToolDefinition[]) {
    const stream = await callLLMStreaming(messages, tools);

    for await (const chunk of stream) {
        if (chunk.type === 'content') {
            // Accumulate content
        } else if (chunk.type === 'tool_call') {
            // Execute in sandbox
            const result = await this.sandbox.execute(chunk.toolCall);

            // Handle async tools with lifecycle events
            if (result.async) {
                await this.emitEvent('tool_start', { toolCall: chunk.toolCall });
                await this.waitForAsyncCompletion(result);
                await this.emitEvent('tool_complete', { toolCall: chunk.toolCall });
            }

            // Send tool result back to LLM
            await this.sendToolResult(result);
        } else if (chunk.type === 'reasoning') {
            // Handle reasoning text (if thinking enabled)
            await this.emitEvent('reasoning', { content: chunk.content });
        }
    }
}
```

---

## 12. Why PicoClaw Exists

PicoClaw was created to bring OpenClaw's agent capabilities to **ultra-constrained environments**:

1. **Hardware constraints:** Raspberry Pi, Pi Zero, ESP32 with external AI chips
2. **Memory constraints:** Systems with <100MB RAM available
3. **Startup time:** Edge devices that need instant-on capability
4. **Cost constraints:** Users who can't run full OpenClaw stack
5. **Simplicity:** Users who don't need enterprise features

### Trade-offs Made

| Feature | PicoClaw | Cost |
|---------|----------|------|
| Docker sandbox | ✗ | Security isolation |
| Auth rotation | ✗ | Reliability under API failures |
| Transcript compaction | ✗ | Long conversation context |
| Subagent lifecycle | ✗ | Complex orchestration |
| Gateway protocol | ✗ | Distributed deployment |
| Multi-agent | ✗ | Team workflows |

---

## 13. Conclusion

PicoClaw is **not a competitor to OpenClaw** — it's a **lightweight derivative** that:

- ✅ **Keeps:** Core agent loop functionality
- ✅ **Keeps:** Tool execution capability
- ✅ **Keeps:** Session management (simplified)
- ✅ **Keeps:** Channel abstraction
- ❌ **Removes:** Gateway architecture
- ❌ **Removes:** Docker sandbox
- ❌ **Removes:** Auth profile rotation
- ❌ **Removes:** Transcript compaction
- ❌ **Removes:** Subagent lifecycle events
- ❌ **Removes:** Complex CLI

### Use PicoClaw when:
- Running on Raspberry Pi or similar
- Need <10MB memory footprint
- Single-user, direct execution
- No Docker available
- Simpler is better

### Use OpenClaw when:
- Production deployment
- Need security isolation
- Multi-agent workflows
- Enterprise auth requirements
- Complex orchestration needs

---

## 14. References

- PicoClaw repository: https://github.com/qidu/picoclaw
- OpenClaw repository: https://github.com/openclaw/openclaw
- Original nanobot inspiration: https://github.com/HKUDS/nanobot

---

*This document was auto-generated as part of the PicoClaw documentation effort.*