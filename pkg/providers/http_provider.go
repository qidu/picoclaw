// PicoClaw - Ultra-lightweight personal AI agent
// Inspired by and based on nanobot: https://github.com/HKUDS/nanobot
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/auth"
	"github.com/sipeed/picoclaw/pkg/config"
)

// isTermux returns true if running in Termux environment
func isTermux() bool {
	return strings.Contains(os.Getenv("PREFIX"), "com.termux")
}

// tcp4Resolver uses Google DNS (8.8.8.8) for lookups over IPv4
func tcp4Resolver(ctx context.Context, network, address string) (net.Conn, error) {
	d := net.Dialer{}
	return d.DialContext(ctx, "tcp4", "8.8.8.8:53")
}

// resolveIPv4 resolves a hostname to IPv4 using Google DNS
func resolveIPv4(ctx context.Context, host string) ([]net.IP, error) {
	resolver := &net.Resolver{
		Dial: tcp4Resolver,
	}
	return resolver.LookupIP(ctx, "ip", host)
}

type HTTPProvider struct {
	apiKey     string
	apiBase    string
	httpClient *http.Client
}

// ProviderInfo holds the configuration for making API requests
type ProviderInfo struct {
	APIKey        string
	APIBase       string
	Proxy         string
	APIFormat     string // "openai", "anthropic", "custom"
	AuthHeader    string // Custom auth header name (default: "Authorization")
	AuthPrefix    string // Auth header prefix (default: "Bearer")
	Endpoint      string // Custom endpoint (default: "/chat/completions")
	RequestMap    map[string]string
	ResponseMap   ResponseMapConfig
}

// ResponseMapConfig defines custom response field mappings
type ResponseMapConfig = config.ResponseMapConfig

// ResponseMapConfigFields for internal use
type ResponseMapConfigFields struct {
	Content      string
	Role         string
	ToolCalls    string
	FinishReason string
	Usage        string
}

func NewHTTPProvider(apiKey, apiBase, proxy string) *HTTPProvider {
	return NewHTTPProviderWithConfig(ProviderInfo{
		APIKey:     apiKey,
		APIBase:    strings.TrimRight(apiBase, "/"),
		Proxy:      proxy,
		APIFormat:  "openai",
		AuthHeader: "Authorization",
		AuthPrefix: "Bearer",
		Endpoint:   "/chat/completions",
	})
}

// NewHTTPProviderWithConfig creates an HTTPProvider with extended configuration
func NewHTTPProviderWithConfig(info ProviderInfo) *HTTPProvider {
	dialer := &net.Dialer{}

	var transport *http.Transport

	// Use IPv4-only on Termux for compatibility
	if isTermux() {
		transport = &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, port, err := net.SplitHostPort(addr)
				if err != nil {
					return dialer.DialContext(ctx, "tcp4", addr)
				}
				ips, err := resolveIPv4(ctx, host)
				if err != nil || len(ips) == 0 {
					return dialer.DialContext(ctx, "tcp4", addr)
				}
				for _, ip := range ips {
					if ip.To4() != nil {
						return dialer.DialContext(ctx, "tcp4", net.JoinHostPort(ip.String(), port))
					}
				}
				return dialer.DialContext(ctx, "tcp4", addr)
			},
			MaxIdleConns:    100,
			IdleConnTimeout: 90 * time.Second,
		}
	} else {
		// Standard transport for non-Termux
		transport = &http.Transport{
			MaxIdleConns:    100,
			IdleConnTimeout: 90 * time.Second,
		}
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   0,
	}

	if info.Proxy != "" {
		proxyURL, err := url.Parse(info.Proxy)
		if err == nil {
			client.Transport = &http.Transport{
				Proxy:           http.ProxyURL(proxyURL),
				MaxIdleConns:    100,
				IdleConnTimeout: 90 * time.Second,
			}
		}
	}

	return &HTTPProvider{
		apiKey:     info.APIKey,
		apiBase:    info.APIBase,
		httpClient: client,
	}
}

func (p *HTTPProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]interface{}) (*LLMResponse, error) {
	if p.apiBase == "" {
		return nil, fmt.Errorf("API base not configured")
	}

	// Strip provider prefix from model name (e.g., moonshot/kimi-k2.5 -> kimi-k2.5)
	if idx := strings.Index(model, "/"); idx != -1 {
		prefix := model[:idx]
		if prefix == "moonshot" || prefix == "nvidia" {
			model = model[idx+1:]
		}
	}

	requestBody := map[string]interface{}{
		"model":    model,
		"messages": messages,
	}

	if len(tools) > 0 {
		requestBody["tools"] = tools
		requestBody["tool_choice"] = "auto"
	}

	if maxTokens, ok := options["max_tokens"].(int); ok {
		lowerModel := strings.ToLower(model)
		if strings.Contains(lowerModel, "glm") || strings.Contains(lowerModel, "o1") {
			requestBody["max_completion_tokens"] = maxTokens
		} else {
			requestBody["max_tokens"] = maxTokens
		}
	}

	if temperature, ok := options["temperature"].(float64); ok {
		lowerModel := strings.ToLower(model)
		// Kimi k2 models only support temperature=1
		if strings.Contains(lowerModel, "kimi") && strings.Contains(lowerModel, "k2") {
			requestBody["temperature"] = 1.0
		} else {
			requestBody["temperature"] = temperature
		}
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed:\n  Status: %d\n  Body:   %s", resp.StatusCode, string(body))
	}

	return p.parseResponse(body)
}

func (p *HTTPProvider) parseResponse(body []byte) (*LLMResponse, error) {
	var apiResponse struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function *struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *UsageInfo `json:"usage"`
	}

	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(apiResponse.Choices) == 0 {
		return &LLMResponse{
			Content:      "",
			FinishReason: "stop",
		}, nil
	}

	choice := apiResponse.Choices[0]

	toolCalls := make([]ToolCall, 0, len(choice.Message.ToolCalls))
	for _, tc := range choice.Message.ToolCalls {
		arguments := make(map[string]interface{})
		name := ""

		// Handle OpenAI format with nested function object
		if tc.Type == "function" && tc.Function != nil {
			name = tc.Function.Name
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &arguments); err != nil {
					arguments["raw"] = tc.Function.Arguments
				}
			}
		} else if tc.Function != nil {
			// Legacy format without type field
			name = tc.Function.Name
			if tc.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &arguments); err != nil {
					arguments["raw"] = tc.Function.Arguments
				}
			}
		}

		toolCalls = append(toolCalls, ToolCall{
			ID:        tc.ID,
			Name:      name,
			Arguments: arguments,
		})
	}

	return &LLMResponse{
		Content:      choice.Message.Content,
		ToolCalls:    toolCalls,
		FinishReason: choice.FinishReason,
		Usage:        apiResponse.Usage,
	}, nil
}

func (p *HTTPProvider) GetDefaultModel() string {
	return ""
}

func createClaudeAuthProvider() (LLMProvider, error) {
	cred, err := auth.GetCredential("anthropic")
	if err != nil {
		return nil, fmt.Errorf("loading auth credentials: %w", err)
	}
	if cred == nil {
		return nil, fmt.Errorf("no credentials for anthropic. Run: picoclaw auth login --provider anthropic")
	}
	return NewClaudeProviderWithTokenSource(cred.AccessToken, createClaudeTokenSource()), nil
}

func createCodexAuthProvider() (LLMProvider, error) {
	cred, err := auth.GetCredential("openai")
	if err != nil {
		return nil, fmt.Errorf("loading auth credentials: %w", err)
	}
	if cred == nil {
		return nil, fmt.Errorf("no credentials for openai. Run: picoclaw auth login --provider openai")
	}
	return NewCodexProviderWithTokenSource(cred.AccessToken, cred.AccountID, createCodexTokenSource()), nil
}

func CreateProvider(cfg *config.Config) (LLMProvider, error) {
	model := cfg.Agents.Defaults.Model
	providerName := strings.ToLower(cfg.Agents.Defaults.Provider)

	var apiKey, apiBase, proxy string

	lowerModel := strings.ToLower(model)

	// First, try to use explicitly configured provider
	if providerName != "" {
		switch providerName {
		case "groq":
			if cfg.Providers.Groq.APIKey != "" {
				apiKey = cfg.Providers.Groq.APIKey
				apiBase = cfg.Providers.Groq.APIBase
				if apiBase == "" {
					apiBase = "https://api.groq.com/openai/v1"
				}
			}
		case "openai", "gpt":
			if cfg.Providers.OpenAI.APIKey != "" || cfg.Providers.OpenAI.AuthMethod != "" {
				if cfg.Providers.OpenAI.AuthMethod == "oauth" || cfg.Providers.OpenAI.AuthMethod == "token" {
					return createCodexAuthProvider()
				}
				apiKey = cfg.Providers.OpenAI.APIKey
				apiBase = cfg.Providers.OpenAI.APIBase
				if apiBase == "" {
					apiBase = "https://api.openai.com/v1"
				}
			}
		case "anthropic", "claude":
			if cfg.Providers.Anthropic.APIKey != "" || cfg.Providers.Anthropic.AuthMethod != "" {
				if cfg.Providers.Anthropic.AuthMethod == "oauth" || cfg.Providers.Anthropic.AuthMethod == "token" {
					return createClaudeAuthProvider()
				}
				apiKey = cfg.Providers.Anthropic.APIKey
				apiBase = cfg.Providers.Anthropic.APIBase
				if apiBase == "" {
					apiBase = "https://api.anthropic.com/v1"
				}
			}
		case "openrouter":
			if cfg.Providers.OpenRouter.APIKey != "" {
				apiKey = cfg.Providers.OpenRouter.APIKey
				if cfg.Providers.OpenRouter.APIBase != "" {
					apiBase = cfg.Providers.OpenRouter.APIBase
				} else {
					apiBase = "https://openrouter.ai/api/v1"
				}
			}
		case "zhipu", "glm":
			if cfg.Providers.Zhipu.APIKey != "" {
				apiKey = cfg.Providers.Zhipu.APIKey
				apiBase = cfg.Providers.Zhipu.APIBase
				if apiBase == "" {
					apiBase = "https://open.bigmodel.cn/api/paas/v4"
				}
			}
		case "gemini", "google":
			if cfg.Providers.Gemini.APIKey != "" {
				apiKey = cfg.Providers.Gemini.APIKey
				apiBase = cfg.Providers.Gemini.APIBase
				if apiBase == "" {
					apiBase = "https://generativelanguage.googleapis.com/v1beta"
				}
			}
		case "vllm":
			if cfg.Providers.VLLM.APIBase != "" {
				apiKey = cfg.Providers.VLLM.APIKey
				apiBase = cfg.Providers.VLLM.APIBase
			}
		case "shengsuanyun":
			if cfg.Providers.ShengSuanYun.APIKey != "" {
				apiKey = cfg.Providers.ShengSuanYun.APIKey
				apiBase = cfg.Providers.ShengSuanYun.APIBase
				if apiBase == "" {
					apiBase = "https://router.shengsuanyun.com/api/v1"
				}
			}
		case "claude-cli", "claudecode", "claude-code":
			workspace := cfg.Agents.Defaults.Workspace
			if workspace == "" {
				workspace = "."
			}
			return NewClaudeCliProvider(workspace), nil
		case "deepseek":
			if cfg.Providers.DeepSeek.APIKey != "" {
				apiKey = cfg.Providers.DeepSeek.APIKey
				apiBase = cfg.Providers.DeepSeek.APIBase
				if apiBase == "" {
					apiBase = "https://api.deepseek.com/v1"
				}
				if model != "deepseek-chat" && model != "deepseek-reasoner" {
					model = "deepseek-chat"
				}
			}

		// Custom provider (user-defined in config)
		default:
			if customCfg, ok := cfg.Providers.CustomProviders[providerName]; ok {
				apiKey = customCfg.APIKey
				apiBase = customCfg.APIBase
				proxy = customCfg.Proxy
			}
		}
	}

	// Fallback: detect provider from model name
	if apiKey == "" && apiBase == "" {
		switch {
		case (strings.Contains(lowerModel, "kimi") || strings.Contains(lowerModel, "moonshot") || strings.HasPrefix(model, "moonshot/")) && cfg.Providers.Moonshot.APIKey != "":
			apiKey = cfg.Providers.Moonshot.APIKey
			apiBase = cfg.Providers.Moonshot.APIBase
			proxy = cfg.Providers.Moonshot.Proxy
			if apiBase == "" {
				apiBase = "https://api.moonshot.cn/v1"
			}

		case strings.HasPrefix(model, "openrouter/") || strings.HasPrefix(model, "anthropic/") || strings.HasPrefix(model, "openai/") || strings.HasPrefix(model, "meta-llama/") || strings.HasPrefix(model, "deepseek/") || strings.HasPrefix(model, "google/"):
			apiKey = cfg.Providers.OpenRouter.APIKey
			proxy = cfg.Providers.OpenRouter.Proxy
			if cfg.Providers.OpenRouter.APIBase != "" {
				apiBase = cfg.Providers.OpenRouter.APIBase
			} else {
				apiBase = "https://openrouter.ai/api/v1"
			}

		case (strings.Contains(lowerModel, "claude") || strings.HasPrefix(model, "anthropic/")) && (cfg.Providers.Anthropic.APIKey != "" || cfg.Providers.Anthropic.AuthMethod != ""):
			if cfg.Providers.Anthropic.AuthMethod == "oauth" || cfg.Providers.Anthropic.AuthMethod == "token" {
				return createClaudeAuthProvider()
			}
			apiKey = cfg.Providers.Anthropic.APIKey
			apiBase = cfg.Providers.Anthropic.APIBase
			proxy = cfg.Providers.Anthropic.Proxy
			if apiBase == "" {
				apiBase = "https://api.anthropic.com/v1"
			}

		case (strings.Contains(lowerModel, "gpt") || strings.HasPrefix(model, "openai/")) && (cfg.Providers.OpenAI.APIKey != "" || cfg.Providers.OpenAI.AuthMethod != ""):
			if cfg.Providers.OpenAI.AuthMethod == "oauth" || cfg.Providers.OpenAI.AuthMethod == "token" {
				return createCodexAuthProvider()
			}
			apiKey = cfg.Providers.OpenAI.APIKey
			apiBase = cfg.Providers.OpenAI.APIBase
			proxy = cfg.Providers.OpenAI.Proxy
			if apiBase == "" {
				apiBase = "https://api.openai.com/v1"
			}

		case (strings.Contains(lowerModel, "gemini") || strings.HasPrefix(model, "google/")) && cfg.Providers.Gemini.APIKey != "":
			apiKey = cfg.Providers.Gemini.APIKey
			apiBase = cfg.Providers.Gemini.APIBase
			proxy = cfg.Providers.Gemini.Proxy
			if apiBase == "" {
				apiBase = "https://generativelanguage.googleapis.com/v1beta"
			}

		case (strings.Contains(lowerModel, "glm") || strings.Contains(lowerModel, "zhipu") || strings.Contains(lowerModel, "zai")) && cfg.Providers.Zhipu.APIKey != "":
			apiKey = cfg.Providers.Zhipu.APIKey
			apiBase = cfg.Providers.Zhipu.APIBase
			proxy = cfg.Providers.Zhipu.Proxy
			if apiBase == "" {
				apiBase = "https://open.bigmodel.cn/api/paas/v4"
			}

		case (strings.Contains(lowerModel, "groq") || strings.HasPrefix(model, "groq/")) && cfg.Providers.Groq.APIKey != "":
			apiKey = cfg.Providers.Groq.APIKey
			apiBase = cfg.Providers.Groq.APIBase
			proxy = cfg.Providers.Groq.Proxy
			if apiBase == "" {
				apiBase = "https://api.groq.com/openai/v1"
			}

		case (strings.Contains(lowerModel, "nvidia") || strings.HasPrefix(model, "nvidia/")) && cfg.Providers.Nvidia.APIKey != "":
			apiKey = cfg.Providers.Nvidia.APIKey
			apiBase = cfg.Providers.Nvidia.APIBase
			proxy = cfg.Providers.Nvidia.Proxy
			if apiBase == "" {
				apiBase = "https://integrate.api.nvidia.com/v1"
			}

		case cfg.Providers.VLLM.APIBase != "":
			apiKey = cfg.Providers.VLLM.APIKey
			apiBase = cfg.Providers.VLLM.APIBase
			proxy = cfg.Providers.VLLM.Proxy

		// Auto-detect custom provider from model name (e.g., "sufy/llama-3.1")
		default:
			// Check if model has provider prefix (e.g., "sufy/model-name")
			if idx := strings.Index(model, "/"); idx != -1 {
				customProviderName := strings.ToLower(model[:idx])
				if customCfg, ok := cfg.Providers.CustomProviders[customProviderName]; ok {
					apiKey = customCfg.APIKey
					apiBase = customCfg.APIBase
					proxy = customCfg.Proxy
					// Strip provider prefix from model
					model = model[idx+1:]
				}
			}

			// Fallback to OpenRouter if still no config
			if apiKey == "" && apiBase == "" {
				if cfg.Providers.OpenRouter.APIKey != "" {
					apiKey = cfg.Providers.OpenRouter.APIKey
					proxy = cfg.Providers.OpenRouter.Proxy
					if cfg.Providers.OpenRouter.APIBase != "" {
						apiBase = cfg.Providers.OpenRouter.APIBase
					} else {
						apiBase = "https://openrouter.ai/api/v1"
					}
				} else {
					return nil, fmt.Errorf("no API key configured for model: %s", model)
				}
			}
		}
	}

	if apiKey == "" && !strings.HasPrefix(model, "bedrock/") {
		return nil, fmt.Errorf("no API key configured for provider (model: %s)", model)
	}

	if apiBase == "" {
		return nil, fmt.Errorf("no API base configured for provider (model: %s)", model)
	}

	// Build provider info with extended config
	providerInfo := ProviderInfo{
		APIKey:      apiKey,
		APIBase:     strings.TrimRight(apiBase, "/"),
		Proxy:       proxy,
		APIFormat:   "openai",
		AuthHeader:  "Authorization",
		AuthPrefix:  "Bearer",
		Endpoint:    "/chat/completions",
		ResponseMap: ResponseMapConfig{},
	}

	// Check if this is a custom provider with extended config
	if customCfg, ok := cfg.Providers.CustomProviders[providerName]; ok {
		providerInfo.APIKey = customCfg.APIKey
		providerInfo.APIBase = strings.TrimRight(customCfg.APIBase, "/")
		providerInfo.Proxy = customCfg.Proxy
		providerInfo.APIFormat = customCfg.APIFormat
		providerInfo.AuthHeader = customCfg.AuthHeader
		providerInfo.AuthPrefix = customCfg.AuthPrefix
		providerInfo.Endpoint = customCfg.Endpoint
		if providerInfo.Endpoint == "" {
			providerInfo.Endpoint = "/chat/completions"
		}
		providerInfo.RequestMap = customCfg.RequestMap
		providerInfo.ResponseMap = customCfg.ResponseMap
	}

	return NewHTTPProviderWithConfig(providerInfo), nil
}