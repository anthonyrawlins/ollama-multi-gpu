package multigpu

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server"
)

// MultiGPUIntegration provides integration with Ollama's existing server
type MultiGPUIntegration struct {
	scheduler      *MultiGPUScheduler
	enabled        bool
	fallbackScheduler interface{} // Original Ollama scheduler
	mu             sync.RWMutex
}

// NewMultiGPUIntegration creates a new integration instance
func NewMultiGPUIntegration(strategy LoadBalancerStrategy) *MultiGPUIntegration {
	return &MultiGPUIntegration{
		scheduler: NewMultiGPUScheduler(strategy),
		enabled:   false,
	}
}

// Initialize sets up the multi-GPU integration
func (i *MultiGPUIntegration) Initialize() error {
	if err := i.scheduler.Initialize(); err != nil {
		slog.Warn("Failed to initialize multi-GPU scheduler, falling back to single GPU", "error", err)
		return err
	}
	
	i.mu.Lock()
	i.enabled = true
	i.mu.Unlock()
	
	slog.Info("Multi-GPU integration enabled successfully")
	return nil
}

// IsEnabled returns whether multi-GPU is currently enabled
func (i *MultiGPUIntegration) IsEnabled() bool {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.enabled
}

// InterceptLlmRequest intercepts and handles LLM requests for multi-GPU processing
func (i *MultiGPUIntegration) InterceptLlmRequest(ctx context.Context, model *server.Model, opts api.Options) (*MultiGPURunner, error) {
	if !i.IsEnabled() {
		return nil, fmt.Errorf("multi-GPU not enabled")
	}
	
	// Use our multi-GPU scheduler
	runner, err := i.scheduler.ScheduleRequest(ctx, model.ModelPath, opts)
	if err != nil {
		slog.Error("Multi-GPU scheduling failed", "error", err, "model", model.ModelPath)
		return nil, err
	}
	
	return runner, nil
}

// GetSchedulerStats returns multi-GPU scheduler statistics
func (i *MultiGPUIntegration) GetSchedulerStats() map[string]interface{} {
	if !i.IsEnabled() {
		return map[string]interface{}{"enabled": false}
	}
	
	stats := i.scheduler.GetStats()
	stats["enabled"] = true
	return stats
}

// CreateMultiGPULoadFunction creates a load function for Ollama's scheduler
func (i *MultiGPUIntegration) CreateMultiGPULoadFunction() func(req *server.LlmRequest, ggml *ggml.GGML, gpus discover.GpuInfoList, numParallel int) {
	return func(req *server.LlmRequest, ggml *ggml.GGML, gpus discover.GpuInfoList, numParallel int) {
		if !i.IsEnabled() {
			// Fall back to original load function
			slog.Debug("Multi-GPU not enabled, using original load function")
			return
		}
		
		// Custom multi-GPU loading logic
		slog.Debug("Multi-GPU load function called",
			"model", req.model.ModelPath,
			"gpus_available", len(gpus.GpuInfoList),
			"num_parallel", numParallel)
		
		// TODO: Implement custom loading logic that distributes models across GPUs
		// This would involve:
		// 1. Analyzing model size and GPU capabilities
		// 2. Deciding whether to use model parallelism or instance parallelism
		// 3. Loading model components onto appropriate GPUs
		// 4. Setting up communication between GPU instances
	}
}

// CreateMultiGPUServerFunction creates a server creation function for Ollama's scheduler
func (i *MultiGPUIntegration) CreateMultiGPUServerFunction() func(gpus discover.GpuInfoList, model string, ggml *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
	return func(gpus discover.GpuInfoList, model string, ggml *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		if !i.IsEnabled() {
			return nil, fmt.Errorf("multi-GPU not enabled")
		}
		
		slog.Debug("Multi-GPU server creation",
			"model", model,
			"gpus", len(gpus.GpuInfoList),
			"adapters", len(adapters),
			"projectors", len(projectors))
		
		// For now, create a wrapper that can handle multi-GPU operations
		return NewMultiGPULlamaServer(gpus, model, ggml, adapters, projectors, opts, numParallel)
	}
}

// MultiGPULlamaServer wraps Ollama's LlamaServer for multi-GPU support
type MultiGPULlamaServer struct {
	primaryServer llm.LlamaServer
	gpuServers    map[string]llm.LlamaServer
	gpuList       discover.GpuInfoList
	model         string
	mu            sync.RWMutex
}

// NewMultiGPULlamaServer creates a new multi-GPU LLama server
func NewMultiGPULlamaServer(gpus discover.GpuInfoList, model string, ggml *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (*MultiGPULlamaServer, error) {
	server := &MultiGPULlamaServer{
		gpuServers: make(map[string]llm.LlamaServer),
		gpuList:    gpus,
		model:      model,
	}
	
	// For now, create servers for each GPU
	// In a full implementation, this would be more sophisticated
	for _, gpu := range gpus.GpuInfoList {
		singleGPUList := discover.GpuInfoList{
			GpuInfoList: []discover.GpuInfo{gpu},
		}
		
		// TODO: Use actual Ollama server creation function
		// gpuServer, err := llm.NewLlamaServer(singleGPUList, model, ggml, adapters, projectors, opts, numParallel)
		// if err != nil {
		//     slog.Error("Failed to create server for GPU", "gpu_id", gpu.ID, "error", err)
		//     continue
		// }
		// server.gpuServers[gpu.ID] = gpuServer
		
		slog.Info("Created server placeholder for GPU", "gpu_id", gpu.ID, "model", model)
	}
	
	return server, nil
}

// Completion handles completion requests across multiple GPUs
func (s *MultiGPULlamaServer) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	// Route to appropriate GPU based on load balancing
	// For now, use primary server or first available
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	if s.primaryServer != nil {
		return s.primaryServer.Completion(ctx, req, fn)
	}
	
	// Use first available GPU server
	for _, server := range s.gpuServers {
		if server != nil {
			return server.Completion(ctx, req, fn)
		}
	}
	
	return fmt.Errorf("no GPU servers available for completion")
}

// Embedding handles embedding requests
func (s *MultiGPULlamaServer) Embedding(ctx context.Context, prompt string) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	if s.primaryServer != nil {
		return s.primaryServer.Embedding(ctx, prompt)
	}
	
	// Use first available GPU server
	for _, server := range s.gpuServers {
		if server != nil {
			return server.Embedding(ctx, prompt)
		}
	}
	
	return nil, fmt.Errorf("no GPU servers available for embedding")
}

// Tokenize handles tokenization requests
func (s *MultiGPULlamaServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	if s.primaryServer != nil {
		return s.primaryServer.Tokenize(ctx, content)
	}
	
	// Use first available GPU server
	for _, server := range s.gpuServers {
		if server != nil {
			return server.Tokenize(ctx, content)
		}
	}
	
	return nil, fmt.Errorf("no GPU servers available for tokenization")
}

// Detokenize handles detokenization requests
func (s *MultiGPULlamaServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	if s.primaryServer != nil {
		return s.primaryServer.Detokenize(ctx, tokens)
	}
	
	// Use first available GPU server
	for _, server := range s.gpuServers {
		if server != nil {
			return server.Detokenize(ctx, tokens)
		}
	}
	
	return "", fmt.Errorf("no GPU servers available for detokenization")
}

// Close closes all GPU servers
func (s *MultiGPULlamaServer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	var lastErr error
	
	if s.primaryServer != nil {
		if err := s.primaryServer.Close(); err != nil {
			slog.Error("Error closing primary server", "error", err)
			lastErr = err
		}
	}
	
	for gpuID, server := range s.gpuServers {
		if server != nil {
			if err := server.Close(); err != nil {
				slog.Error("Error closing GPU server", "gpu_id", gpuID, "error", err)
				lastErr = err
			}
		}
	}
	
	s.gpuServers = make(map[string]llm.LlamaServer)
	s.primaryServer = nil
	
	return lastErr
}

// EstimateVRAM estimates VRAM usage
func (s *MultiGPULlamaServer) EstimateVRAM() uint64 {
	// Simple estimation - should be more sophisticated in practice
	return 2 * 1024 * 1024 * 1024 // 2GB default
}

// Close shuts down the integration
func (i *MultiGPUIntegration) Close() error {
	i.mu.Lock()
	i.enabled = false
	i.mu.Unlock()
	
	if i.scheduler != nil {
		return i.scheduler.Close()
	}
	
	return nil
}