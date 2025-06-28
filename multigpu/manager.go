package multigpu

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/ollama/ollama/discover"
)

// GPUState represents the current state of a GPU
type GPUState struct {
	ID                string
	Name              string
	TotalMemory       uint64
	FreeMemory        uint64
	UsedMemory        uint64
	Utilization       float64
	ActiveSessions    int
	QueuedRequests    int
	LastUpdated       time.Time
	ComputeCapability string
	Library           string
	mu                sync.RWMutex
}

// LoadBalancerStrategy defines how requests are distributed across GPUs
type LoadBalancerStrategy int

const (
	RoundRobin LoadBalancerStrategy = iota
	MemoryAware
	UtilizationAware
	Hybrid
)

// LoadBalancer manages request distribution across multiple GPUs
type LoadBalancer struct {
	strategy    LoadBalancerStrategy
	gpus        map[string]*GPUState
	currentGPU  int
	weights     map[string]float64
	mu          sync.RWMutex
}

// MultiGPUManager orchestrates multi-GPU inference operations
type MultiGPUManager struct {
	balancer     *LoadBalancer
	gpuStates    map[string]*GPUState
	requestQueue chan *InferenceRequest
	ctx          context.Context
	cancel       context.CancelFunc
	mu           sync.RWMutex
}

// InferenceRequest represents a request for GPU inference
type InferenceRequest struct {
	ID              string
	ModelName       string
	RequiredMemory  uint64
	Priority        int
	CreatedAt       time.Time
	AssignedGPU     string
	ResponseChan    chan *InferenceResponse
	Context         context.Context
}

// InferenceResponse contains the result of GPU inference
type InferenceResponse struct {
	RequestID string
	Success   bool
	Error     error
	GPUUsed   string
	Duration  time.Duration
	Data      interface{}
}

// NewMultiGPUManager creates a new multi-GPU manager
func NewMultiGPUManager(strategy LoadBalancerStrategy) *MultiGPUManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	manager := &MultiGPUManager{
		balancer:     NewLoadBalancer(strategy),
		gpuStates:    make(map[string]*GPUState),
		requestQueue: make(chan *InferenceRequest, 1000),
		ctx:          ctx,
		cancel:       cancel,
	}
	
	return manager
}

// NewLoadBalancer creates a new load balancer with the specified strategy
func NewLoadBalancer(strategy LoadBalancerStrategy) *LoadBalancer {
	return &LoadBalancer{
		strategy: strategy,
		gpus:     make(map[string]*GPUState),
		weights:  make(map[string]float64),
	}
}

// InitializeGPUs discovers and initializes all available GPUs
func (m *MultiGPUManager) InitializeGPUs() error {
	// Discover available GPUs using Ollama's existing discovery
	gpuList := discover.GetGPUInfo()
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	for _, gpu := range gpuList.GpuInfoList {
		state := &GPUState{
			ID:                gpu.ID,
			Name:              gpu.Name,
			TotalMemory:       gpu.TotalMemory,
			FreeMemory:        gpu.FreeMemory,
			UsedMemory:        gpu.TotalMemory - gpu.FreeMemory,
			Utilization:       0.0,
			ActiveSessions:    0,
			QueuedRequests:    0,
			LastUpdated:       time.Now(),
			ComputeCapability: gpu.Compute,
			Library:           gpu.Library,
		}
		
		m.gpuStates[gpu.ID] = state
		m.balancer.AddGPU(state)
		
		slog.Info("Initialized GPU for multi-GPU management",
			"gpu_id", gpu.ID,
			"name", gpu.Name,
			"memory", gpu.TotalMemory,
			"library", gpu.Library)
	}
	
	if len(m.gpuStates) == 0 {
		return fmt.Errorf("no GPUs found for multi-GPU management")
	}
	
	slog.Info("Multi-GPU manager initialized",
		"gpu_count", len(m.gpuStates),
		"strategy", m.getStrategyName())
	
	return nil
}

// AddGPU adds a GPU to the load balancer
func (lb *LoadBalancer) AddGPU(gpu *GPUState) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	
	lb.gpus[gpu.ID] = gpu
	
	// Set default weights based on memory and compute capability
	weight := float64(gpu.TotalMemory) / 1024 / 1024 / 1024 // GB
	if gpu.ComputeCapability != "" {
		// Boost weight for newer architectures
		weight *= 1.2
	}
	lb.weights[gpu.ID] = weight
}

// SelectGPU selects the best GPU for a request based on the current strategy
func (lb *LoadBalancer) SelectGPU(req *InferenceRequest) (string, error) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	
	if len(lb.gpus) == 0 {
		return "", fmt.Errorf("no GPUs available")
	}
	
	switch lb.strategy {
	case RoundRobin:
		return lb.selectRoundRobin()
	case MemoryAware:
		return lb.selectMemoryAware(req)
	case UtilizationAware:
		return lb.selectUtilizationAware()
	case Hybrid:
		return lb.selectHybrid(req)
	default:
		return lb.selectRoundRobin()
	}
}

// selectRoundRobin implements round-robin GPU selection
func (lb *LoadBalancer) selectRoundRobin() (string, error) {
	gpuIDs := make([]string, 0, len(lb.gpus))
	for id := range lb.gpus {
		gpuIDs = append(gpuIDs, id)
	}
	sort.Strings(gpuIDs) // Ensure consistent ordering
	
	if len(gpuIDs) == 0 {
		return "", fmt.Errorf("no GPUs available")
	}
	
	selectedID := gpuIDs[lb.currentGPU%len(gpuIDs)]
	lb.currentGPU++
	
	return selectedID, nil
}

// selectMemoryAware implements memory-aware GPU selection
func (lb *LoadBalancer) selectMemoryAware(req *InferenceRequest) (string, error) {
	var bestGPU string
	var maxFreeMemory uint64
	
	for id, gpu := range lb.gpus {
		gpu.mu.RLock()
		freeMemory := gpu.FreeMemory
		gpu.mu.RUnlock()
		
		// Ensure GPU has enough memory for the request
		if freeMemory >= req.RequiredMemory && freeMemory > maxFreeMemory {
			maxFreeMemory = freeMemory
			bestGPU = id
		}
	}
	
	if bestGPU == "" {
		return "", fmt.Errorf("no GPU has sufficient memory for request (required: %d MB)", req.RequiredMemory/1024/1024)
	}
	
	return bestGPU, nil
}

// selectUtilizationAware implements utilization-aware GPU selection
func (lb *LoadBalancer) selectUtilizationAware() (string, error) {
	var bestGPU string
	var lowestLoad float64 = 1000.0 // High initial value
	
	for id, gpu := range lb.gpus {
		gpu.mu.RLock()
		// Calculate combined load (utilization + queue factor)
		queueFactor := float64(gpu.QueuedRequests) * 10.0
		combinedLoad := gpu.Utilization + queueFactor
		gpu.mu.RUnlock()
		
		if combinedLoad < lowestLoad {
			lowestLoad = combinedLoad
			bestGPU = id
		}
	}
	
	if bestGPU == "" {
		return "", fmt.Errorf("no suitable GPU found")
	}
	
	return bestGPU, nil
}

// selectHybrid implements hybrid GPU selection considering multiple factors
func (lb *LoadBalancer) selectHybrid(req *InferenceRequest) (string, error) {
	var bestGPU string
	var bestScore float64 = -1.0
	
	for id, gpu := range lb.gpus {
		gpu.mu.RLock()
		
		// Check if GPU has enough memory
		if gpu.FreeMemory < req.RequiredMemory {
			gpu.mu.RUnlock()
			continue
		}
		
		// Calculate composite score (higher is better)
		memoryRatio := float64(gpu.FreeMemory) / float64(gpu.TotalMemory)
		utilizationFactor := (100.0 - gpu.Utilization) / 100.0
		queueFactor := 1.0 / (1.0 + float64(gpu.QueuedRequests))
		priorityFactor := float64(req.Priority) / 10.0
		
		score := (memoryRatio * 0.4) + (utilizationFactor * 0.3) + (queueFactor * 0.2) + (priorityFactor * 0.1)
		
		// Apply GPU-specific weight
		if weight, exists := lb.weights[id]; exists {
			score *= weight / 8.0 // Normalize around 8GB baseline
		}
		
		gpu.mu.RUnlock()
		
		if score > bestScore {
			bestScore = score
			bestGPU = id
		}
	}
	
	if bestGPU == "" {
		return "", fmt.Errorf("no suitable GPU found for hybrid selection")
	}
	
	return bestGPU, nil
}

// UpdateGPUState updates the state of a specific GPU
func (m *MultiGPUManager) UpdateGPUState(gpuID string, freeMemory uint64, utilization float64) {
	m.mu.RLock()
	gpu, exists := m.gpuStates[gpuID]
	m.mu.RUnlock()
	
	if !exists {
		slog.Warn("Attempted to update unknown GPU", "gpu_id", gpuID)
		return
	}
	
	gpu.mu.Lock()
	gpu.FreeMemory = freeMemory
	gpu.UsedMemory = gpu.TotalMemory - freeMemory
	gpu.Utilization = utilization
	gpu.LastUpdated = time.Now()
	gpu.mu.Unlock()
}

// AssignGPUToRequest assigns a GPU to an inference request
func (m *MultiGPUManager) AssignGPUToRequest(req *InferenceRequest) error {
	gpuID, err := m.balancer.SelectGPU(req)
	if err != nil {
		return fmt.Errorf("failed to select GPU: %w", err)
	}
	
	req.AssignedGPU = gpuID
	
	// Update GPU state
	m.mu.RLock()
	gpu, exists := m.gpuStates[gpuID]
	m.mu.RUnlock()
	
	if exists {
		gpu.mu.Lock()
		gpu.QueuedRequests++
		gpu.mu.Unlock()
	}
	
	slog.Debug("Assigned GPU to request",
		"request_id", req.ID,
		"gpu_id", gpuID,
		"model", req.ModelName)
	
	return nil
}

// GetGPUStats returns current statistics for all GPUs
func (m *MultiGPUManager) GetGPUStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	stats := make(map[string]interface{})
	
	for id, gpu := range m.gpuStates {
		gpu.mu.RLock()
		stats[id] = map[string]interface{}{
			"name":             gpu.Name,
			"total_memory_mb":  gpu.TotalMemory / 1024 / 1024,
			"free_memory_mb":   gpu.FreeMemory / 1024 / 1024,
			"used_memory_mb":   gpu.UsedMemory / 1024 / 1024,
			"utilization":      gpu.Utilization,
			"active_sessions":  gpu.ActiveSessions,
			"queued_requests":  gpu.QueuedRequests,
			"last_updated":     gpu.LastUpdated.Format(time.RFC3339),
		}
		gpu.mu.RUnlock()
	}
	
	return stats
}

// getStrategyName returns the name of the current load balancing strategy
func (m *MultiGPUManager) getStrategyName() string {
	switch m.balancer.strategy {
	case RoundRobin:
		return "round_robin"
	case MemoryAware:
		return "memory_aware"
	case UtilizationAware:
		return "utilization_aware"
	case Hybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// Close gracefully shuts down the multi-GPU manager
func (m *MultiGPUManager) Close() error {
	m.cancel()
	close(m.requestQueue)
	
	slog.Info("Multi-GPU manager shut down successfully")
	return nil
}