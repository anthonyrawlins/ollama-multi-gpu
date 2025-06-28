package multigpu

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llm"
)

// MultiGPUScheduler extends Ollama's scheduler for multi-GPU support
type MultiGPUScheduler struct {
	manager       *MultiGPUManager
	activeRunners map[string]*MultiGPURunner
	runnersMu     sync.RWMutex
	
	// Request tracking
	totalRequests   int64
	activeRequests  int64
	completedReqs   int64
	failedRequests  int64
	
	// Performance metrics
	avgResponseTime time.Duration
	responseTimes   []time.Duration
	metricsMu       sync.RWMutex
}

// MultiGPURunner manages a model instance on a specific GPU
type MultiGPURunner struct {
	ID           string
	GPUID        string
	ModelName    string
	Server       llm.LlamaServer
	LastUsed     time.Time
	RequestCount int64
	mu           sync.RWMutex
}

// NewMultiGPUScheduler creates a new multi-GPU scheduler
func NewMultiGPUScheduler(strategy LoadBalancerStrategy) *MultiGPUScheduler {
	manager := NewMultiGPUManager(strategy)
	
	scheduler := &MultiGPUScheduler{
		manager:       manager,
		activeRunners: make(map[string]*MultiGPURunner),
		responseTimes: make([]time.Duration, 0, 1000),
	}
	
	return scheduler
}

// Initialize sets up the multi-GPU scheduler
func (s *MultiGPUScheduler) Initialize() error {
	if err := s.manager.InitializeGPUs(); err != nil {
		return fmt.Errorf("failed to initialize multi-GPU manager: %w", err)
	}
	
	slog.Info("Multi-GPU scheduler initialized successfully",
		"gpu_count", len(s.manager.gpuStates))
	
	return nil
}

// ScheduleRequest schedules an inference request across available GPUs
func (s *MultiGPUScheduler) ScheduleRequest(ctx context.Context, modelName string, opts api.Options) (*MultiGPURunner, error) {
	atomic.AddInt64(&s.totalRequests, 1)
	atomic.AddInt64(&s.activeRequests, 1)
	defer atomic.AddInt64(&s.activeRequests, -1)
	
	startTime := time.Now()
	
	// Create inference request
	req := &InferenceRequest{
		ID:             fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), atomic.LoadInt64(&s.totalRequests)),
		ModelName:      modelName,
		RequiredMemory: s.estimateModelMemory(modelName, opts),
		Priority:       s.calculatePriority(opts),
		CreatedAt:      startTime,
		Context:        ctx,
	}
	
	// Assign GPU
	if err := s.manager.AssignGPUToRequest(req); err != nil {
		atomic.AddInt64(&s.failedRequests, 1)
		return nil, fmt.Errorf("failed to assign GPU: %w", err)
	}
	
	// Get or create runner for the assigned GPU
	runner, err := s.getOrCreateRunner(req.AssignedGPU, modelName, opts)
	if err != nil {
		atomic.AddInt64(&s.failedRequests, 1)
		return nil, fmt.Errorf("failed to get runner for GPU %s: %w", req.AssignedGPU, err)
	}
	
	// Update metrics
	duration := time.Since(startTime)
	s.updateMetrics(duration)
	atomic.AddInt64(&s.completedReqs, 1)
	
	slog.Debug("Request scheduled successfully",
		"request_id", req.ID,
		"model", modelName,
		"gpu_id", req.AssignedGPU,
		"duration_ms", duration.Milliseconds())
	
	return runner, nil
}

// getOrCreateRunner gets an existing runner or creates a new one for the GPU
func (s *MultiGPUScheduler) getOrCreateRunner(gpuID, modelName string, opts api.Options) (*MultiGPURunner, error) {
	runnerKey := fmt.Sprintf("%s_%s", gpuID, modelName)
	
	s.runnersMu.RLock()
	if runner, exists := s.activeRunners[runnerKey]; exists {
		runner.mu.Lock()
		runner.LastUsed = time.Now()
		runner.RequestCount++
		runner.mu.Unlock()
		s.runnersMu.RUnlock()
		return runner, nil
	}
	s.runnersMu.RUnlock()
	
	// Need to create new runner
	s.runnersMu.Lock()
	defer s.runnersMu.Unlock()
	
	// Double-check in case another goroutine created it
	if runner, exists := s.activeRunners[runnerKey]; exists {
		runner.mu.Lock()
		runner.LastUsed = time.Now()
		runner.RequestCount++
		runner.mu.Unlock()
		return runner, nil
	}
	
	// Create new runner
	runner, err := s.createRunner(gpuID, modelName, opts)
	if err != nil {
		return nil, err
	}
	
	s.activeRunners[runnerKey] = runner
	
	slog.Info("Created new multi-GPU runner",
		"runner_id", runner.ID,
		"gpu_id", gpuID,
		"model", modelName)
	
	return runner, nil
}

// createRunner creates a new runner instance for a specific GPU
func (s *MultiGPUScheduler) createRunner(gpuID, modelName string, opts api.Options) (*MultiGPURunner, error) {
	// Get GPU info for the specific GPU
	gpuList := discover.GetGPUInfo()
	var targetGPU *discover.GpuInfo
	
	for _, gpu := range gpuList.GpuInfoList {
		if gpu.ID == gpuID {
			targetGPU = &gpu
			break
		}
	}
	
	if targetGPU == nil {
		return nil, fmt.Errorf("GPU %s not found", gpuID)
	}
	
	// Create GPU list with only the target GPU
	singleGPUList := discover.GpuInfoList{
		GpuInfoList: []discover.GpuInfo{*targetGPU},
	}
	
	// Create the LLM server for this specific GPU
	// This would typically involve loading the model onto the specific GPU
	// For now, we'll create a placeholder that can be expanded
	runner := &MultiGPURunner{
		ID:           fmt.Sprintf("runner_%s_%s_%d", gpuID, modelName, time.Now().UnixNano()),
		GPUID:        gpuID,
		ModelName:    modelName,
		LastUsed:     time.Now(),
		RequestCount: 1,
	}
	
	// TODO: Integrate with actual model loading on specific GPU
	// server, err := llm.NewLlamaServer(singleGPUList, modelName, ggmlFile, adapters, projectors, opts, numParallel)
	// if err != nil {
	//     return nil, fmt.Errorf("failed to create LLM server: %w", err)
	// }
	// runner.Server = server
	
	return runner, nil
}

// estimateModelMemory estimates the memory required for a model
func (s *MultiGPUScheduler) estimateModelMemory(modelName string, opts api.Options) uint64 {
	// Simple heuristic based on model name and context size
	// This should be enhanced with actual model size information
	
	baseMemory := uint64(1024 * 1024 * 1024) // 1GB base
	
	// Estimate based on model name patterns
	if contains(modelName, "7b") {
		baseMemory = uint64(4 * 1024 * 1024 * 1024) // 4GB for 7B models
	} else if contains(modelName, "13b") {
		baseMemory = uint64(8 * 1024 * 1024 * 1024) // 8GB for 13B models
	} else if contains(modelName, "33b") || contains(modelName, "30b") {
		baseMemory = uint64(16 * 1024 * 1024 * 1024) // 16GB for 30B+ models
	} else if contains(modelName, "70b") {
		baseMemory = uint64(32 * 1024 * 1024 * 1024) // 32GB for 70B models
	}
	
	// Add context size overhead
	if opts.NumCtx != nil && *opts.NumCtx > 0 {
		contextOverhead := uint64(*opts.NumCtx * 2 * 1024) // 2KB per context token
		baseMemory += contextOverhead
	}
	
	return baseMemory
}

// calculatePriority calculates request priority based on options
func (s *MultiGPUScheduler) calculatePriority(opts api.Options) int {
	priority := 5 // Default priority
	
	// Increase priority for smaller requests (faster to process)
	if opts.NumCtx != nil && *opts.NumCtx < 2048 {
		priority += 2
	}
	
	// Could add more priority logic based on other factors
	
	return priority
}

// updateMetrics updates performance metrics
func (s *MultiGPUScheduler) updateMetrics(duration time.Duration) {
	s.metricsMu.Lock()
	defer s.metricsMu.Unlock()
	
	s.responseTimes = append(s.responseTimes, duration)
	
	// Keep only last 1000 response times
	if len(s.responseTimes) > 1000 {
		s.responseTimes = s.responseTimes[1:]
	}
	
	// Calculate average
	var total time.Duration
	for _, rt := range s.responseTimes {
		total += rt
	}
	s.avgResponseTime = total / time.Duration(len(s.responseTimes))
}

// GetStats returns scheduler statistics
func (s *MultiGPUScheduler) GetStats() map[string]interface{} {
	s.metricsMu.RLock()
	avgResponse := s.avgResponseTime
	responseCount := len(s.responseTimes)
	s.metricsMu.RUnlock()
	
	stats := map[string]interface{}{
		"total_requests":    atomic.LoadInt64(&s.totalRequests),
		"active_requests":   atomic.LoadInt64(&s.activeRequests),
		"completed_requests": atomic.LoadInt64(&s.completedReqs),
		"failed_requests":   atomic.LoadInt64(&s.failedRequests),
		"avg_response_time_ms": avgResponse.Milliseconds(),
		"response_samples":  responseCount,
		"active_runners":    len(s.activeRunners),
		"gpu_stats":        s.manager.GetGPUStats(),
	}
	
	return stats
}

// CleanupIdleRunners removes idle runners to free up GPU memory
func (s *MultiGPUScheduler) CleanupIdleRunners(maxIdleTime time.Duration) {
	s.runnersMu.Lock()
	defer s.runnersMu.Unlock()
	
	now := time.Now()
	toRemove := []string{}
	
	for key, runner := range s.activeRunners {
		runner.mu.RLock()
		isIdle := now.Sub(runner.LastUsed) > maxIdleTime
		runner.mu.RUnlock()
		
		if isIdle {
			toRemove = append(toRemove, key)
			
			// Close the runner's server if it exists
			if runner.Server != nil {
				if err := runner.Server.Close(); err != nil {
					slog.Warn("Error closing runner server", "runner_id", runner.ID, "error", err)
				}
			}
			
			slog.Info("Cleaned up idle runner",
				"runner_id", runner.ID,
				"gpu_id", runner.GPUID,
				"model", runner.ModelName,
				"idle_time", now.Sub(runner.LastUsed))
		}
	}
	
	for _, key := range toRemove {
		delete(s.activeRunners, key)
	}
}

// Close gracefully shuts down the scheduler
func (s *MultiGPUScheduler) Close() error {
	// Close all active runners
	s.runnersMu.Lock()
	for _, runner := range s.activeRunners {
		if runner.Server != nil {
			if err := runner.Server.Close(); err != nil {
				slog.Warn("Error closing runner during shutdown", "runner_id", runner.ID, "error", err)
			}
		}
	}
	s.activeRunners = make(map[string]*MultiGPURunner)
	s.runnersMu.Unlock()
	
	// Close the manager
	if err := s.manager.Close(); err != nil {
		return fmt.Errorf("error closing multi-GPU manager: %w", err)
	}
	
	slog.Info("Multi-GPU scheduler shut down successfully")
	return nil
}

// Helper function to check if string contains substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(s) > len(substr) && 
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
				findInString(s, substr))))
}

func findInString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}