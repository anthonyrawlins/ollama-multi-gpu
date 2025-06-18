# Ollama Multi-GPU Fork

This is a comprehensive fork of Ollama designed to provide advanced multi-GPU capabilities including true tensor parallelism, pipeline parallelism, and cross-vendor GPU coordination.

## Key Features

- **True Multi-GPU Parallelism**: Tensor and pipeline parallelism that actually improves performance
- **Cross-Vendor Support**: Simultaneous use of NVIDIA, AMD, and Intel GPUs
- **Advanced Scheduling**: Performance-aware scheduling algorithms
- **Enterprise Features**: Dynamic load balancing, model migration, resource optimization
- **Backward Compatibility**: Maintains existing API and user experience

## Development Plan

See [MULTI_GPU_OLLAMA_PLAN.md](./MULTI_GPU_OLLAMA_PLAN.md) for the comprehensive development plan.

## Performance Targets

- 2-4x throughput improvement with proper tensor parallelism
- 50-70% latency reduction for large models
- Support for 400B+ models across consumer hardware
- 90%+ GPU utilization vs current 40-60%

## Quick Start

```bash
# Clone the repository
git clone https://github.com/anthonyrawlins/ollama-multi-gpu.git
cd ollama-multi-gpu

# Build (when implementation begins)
make build

# Run with multi-GPU support
./ollama serve --multi-gpu
```

## Contributing

This fork is in planning phase. See the development plan for implementation roadmap.

## Status

🚧 **Planning Phase** - Comprehensive development plan completed
🚧 **Phase 1** - Foundation & Backend Integration (Upcoming)
🚧 **Phase 2** - Multi-GPU Parallelism Implementation (Upcoming)
🚧 **Phase 3** - Advanced Features & Optimization (Upcoming)

---

Based on the original [Ollama](https://github.com/ollama/ollama) project.
