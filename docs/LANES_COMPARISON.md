# Lanes - PicoClaw vs OpenClaw Comparison

> Generated: 2026-02-16
> Author: Clawra

## What is a Lane?

A **lane** is a session isolation mechanism that groups related sessions together and manages their priority and concurrency. Think of it like lanes on a highway — higher priority traffic gets its own lane and never has to wait.

---

## OpenClaw Lanes

OpenClaw implements a comprehensive lane system for enterprise multi-agent environments.

### Built-in Lanes

| Lane | Priority | Max Concurrent | Purpose |
|------|----------|----------------|---------|
| **heartbeat** | 100 | 1 | Critical monitoring and check-ins |
| **default** | 50 | 5 | Normal interactive chat |
| **subagent** | 25 | Unlimited | Background spawned subagents |
| **background** | 10 | 10 | Bulk operations and cron jobs |
| **broadcast** | 1 | Unlimited | Low-priority announcements |

### Features

- **DM History Limits:** Per-lane configuration for conversation history retention
- **Dynamic Routing:** Automatic lane assignment based on session characteristics
- **Full Scheduler:** Resource allocation with priority-based preemption
- **Per-User Quotas:** Enterprise features for multi-user environments

### Priority Flow

```
Priority 100 ──► heartbeat ──► Immediate processing (never queued)
     │
     ▼
Priority 50 ──► default ──► Waits if slot available (max 5 concurrent)
     │
     ▼
Priority 25 ──► subagent ──► Background execution (no limits)
     │
     ▼
Priority 10 ──► background ──► Queued and batched
     │
     ▼
Priority 1 ──► broadcast ──► Best-effort delivery only
```

### Implementation Location

- **Main implementation:** `src/agents/pi-embedded-runner/lanes.ts`
- **Session lane resolution:** `src/agents/pi-embedded-runner/lanes.ts`
- **Lane stats:** Per-lane active session tracking

---

## PicoClaw Lanes (After Refactor)

PicoClaw implements a simplified lane system inspired by OpenClaw but adapted for single-user, resource-constrained environments.

### Device Profiles

PicoClaw adds **device-aware lane configuration** that auto-detects hardware capabilities and applies appropriate limits.

| Device | CPU | RAM | heartbeat | default | subagent | background | broadcast |
|--------|-----|-----|-----------|---------|----------|------------|-----------|
| **pi-zero** | 1 | 512MB | 1 | 1 | 2 | 2 | 2 |
| **pi-3** | 4 | 1GB | 1 | 2 | 4 | 5 | 5 |
| **pi-4** | 4 | 4GB | 1 | 3 | 5 | 8 | 10 |
| **phone-8** | 8 | 8GB | 1 | 5 | 8 | 12 | 15 |
| **desktop** | 8 | 16GB | 1 | 10 | 20 | 30 | 50 |
| **server** | 32 | 64GB | 1 | 50 | 100 | 200 | 500 |

### Configuration Modes

**Auto-detection (default):**
```json
{
  "devices": {
    "device_type": "auto"
  }
}
```
Automatically detects CPU cores and RAM, applies appropriate limits.

**Explicit configuration:**
```json
{
  "devices": {
    "device_type": "phone-8"
  }
}
```

**Full manual override:**
```json
{
  "devices": {
    "device_type": "custom",
    "lanes": {
      "heartbeat": { "priority": 100, "max_concurrent": 2 },
      "default": { "priority": 50, "max_concurrent": 8 },
      "background": { "priority": 10, "max_concurrent": 15 }
    }
  }
}
```

### Implementation Components

1. **LaneManager** (`pkg/agent/loop.go`):
   - Session registration/unregistration per lane
   - Active session tracking
   - Lane statistics

2. **Device Profiles** (`pkg/config/config.go`):
   - Predefined profiles for common devices
   - Auto-detection based on system capabilities
   - Explicit profile override

3. **RunManager** (`pkg/agent/loop.go`):
   - Run queue and abort support
   - Active run tracking with stats

---

## Key Differences

| Aspect | OpenClaw | PicoClaw |
|--------|----------|----------|
| **Target Environment** | Enterprise, multi-agent | Personal, single-user |
| **DM History Limits** | Per-lane configuration | Not implemented |
| **Dynamic Routing** | Automatic assignment | Manual assignment |
| **Resource Scheduler** | Full scheduler | Simple count limiting |
| **Device Awareness** | Not supported | Auto-detection |
| **Per-User Quotas** | Enterprise feature | Not implemented |
| **Priority Range** | 1-100 | 1-100 (same) |
| **Config Override** | CLI + config | Env vars + config |

---

## Why PicoClaw Doesn't Need Full OpenClaw Lanes

### When OpenClaw Lanes Are Necessary

- Multi-user environments (different users different lanes)
- Enterprise quotas (per-user resource limits)
- Complex scheduling (priority-based preemption)
- Integration with external schedulers

### When PicoClaw Lanes Are Sufficient

- Single-user personal assistant
- Resource-constrained devices (Pi, phones)
- Simple priority separation (heartbeat vs chat)
- Mobile/edge deployment

---

## Design Philosophy

**OpenClaw:** "Enterprise-grade" - comprehensive features for complex deployments

**PicoClaw:** "Minimalism" - add only what you need, auto-adapt to device

---

## Lane Use Cases

### Heartbeat (Critical)

```go
// "I'm OK" check-ins must never fail
lane := lm.GetLane("heartbeat")
// Max 1 concurrent, Priority 100
```

### Interactive Chat (Normal)

```go
// Normal conversations
lane := lm.GetLane("default")
// Max 5 concurrent, Priority 50
```

### Subagent (Background)

```go
// Background tasks don't block main agent
lane := lm.GetLane("subagent")
// Unlimited concurrent, Priority 25
```

### Broadcast (Best-Effort)

```go
// Low-priority announcements
lane := lm.GetLane("broadcast")
// Unlimited, Priority 1, sent when idle
```

---

## Summary

| | OpenClaw | PicoClaw |
|--|----------|----------|
| **Complexity** | High (enterprise) | Low (minimal) |
| **Device-aware** | ❌ | ✅ Auto-detection |
| **Configurable** | Yes | Yes (with auto-detection) |
| **DM limits** | Per-lane | Not needed |
| **Best for** | Teams, enterprises | Personal, edge devices |

---

## References

- PicoClaw: `pkg/agent/loop.go` (LaneManager), `pkg/config/config.go` (Device profiles)
- OpenClaw: `src/agents/pi-embedded-runner/lanes.ts`, `src/agents/pi-embedded-runner.ts`

---

*This document provides a comparison of lane implementations between OpenClaw and PicoClaw.*