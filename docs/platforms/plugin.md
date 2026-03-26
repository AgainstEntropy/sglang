# SGLang Plugin

## Overview

Inspired by vLLM's platform abstraction. Allows hardware vendors to extend SGLang **without modifying the main repository code**.

The framework provides two plugin types, both discovered and loaded via Python's standard `setuptools` entry_points mechanism:

| Plugin Type | Entry Point Group | Purpose |
|---|---|---|
| **Hardware Platform Plugin** | `sglang.platform_plugins` | Register a custom hardware platform (device operations, KV cache pools, attention backends, CUDA Graph, compilation backends, etc.) |
| **General Function Plugin** | `sglang.general_plugins` | Inject hooks (before/after/around/replace) into any function/method in sglang, or replace entire classes |

### Principles

- **Non-intrusive**: Existing CUDA/ROCm/NPU/XPU code remains unchanged. OOT code paths are added as `elif` branches alongside existing hardware-specific logic.
- **Zero configuration**: Plugins are automatically discovered after `pip install`, no sglang code changes required.
- **SGLANG_PLUGINS environment variable**: Controls which plugins to load via a comma-separated whitelist. Only one hardware plugin can be loaded, while multiple general plugins can be loaded simultaneously.

## Plugin Type 1: Hardware Platform Plugin

### Description

A hardware platform plugin registers a `Platform` subclass that tells SGLang how to interact with a specific hardware backend. The platform controls the following:

- **Device operations**: `set_device()`, `get_device_name()`, `get_device_total_memory()`, etc.
- **Capability flags**: `support_cuda_graph()`, `support_cublas()`, `support_kernel_warmup()`, `supports_fp8()`, etc.
- **Subsystem factory methods**: Specify which KV cache pool, Graph Runner, memory allocator, compilation backend, and attention backend to use
- **Configuration lifecycle hooks**: `apply_server_args_defaults()`, `init_backend()`, `apply_worker_patches()`, etc.
- **MultiPlatformOp dispatch**: Compatible with the existing MultiPlatformOp mechanism, dispatching to custom methods via key. Specifies which `forward_<key>()` method to call for fused operators

### Platform Interface

```python
from sglang.srt.platforms.interface import Platform, PlatformEnum

class MyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "my_device"
    device_type: str = "cuda"          # torch device type
    dispatch_key: str = "CUDA"         # PyTorch dispatch key
    device_control_env_var: str = "MY_VISIBLE_DEVICES"
    dist_backend: str = "nccl"         # or "hccl", "gloo", etc.
```

#### Identity Queries (Instance Methods)

| Method | Default | Description |
|---|---|---|
| `is_cuda_alike()` | `False` for OOT | Override to `True` if the hardware supports CUDA-like APIs (enables alt_stream, etc.) |
| `is_out_of_tree()` | `True` for OOT | Automatically detected based on `_enum = PlatformEnum.OOT` |

#### Capability Flags (Class Methods)

| Method | Default | Description |
|---|---|---|
| `support_cuda_graph()` | `True` | Whether device Graph capture is supported |
| `support_cublas()` | `False` | Whether to initialize cuBLAS |
| `support_kernel_warmup()` | `False` | Whether to run kernel warmup |
| `support_torch_compile()` | `True` | Whether torch.compile is available |
| `supports_fp8()` | `False` | Whether FP8 quantization is supported |
| `is_pin_memory_available()` | `True` | Whether pinned memory is available |

#### Subsystem Factory Methods (Class Methods)

| Method | Default Return Value | Description |
|---|---|---|
| `get_default_attention_backend()` | `"flashinfer"` | Default attention backend name |
| `get_graph_runner_cls()` | `CudaGraphRunner` | Graph Runner class |
| `get_mha_kv_pool_cls()` | `MHATokenToKVPool` | MHA KV cache pool class |
| `get_mla_kv_pool_cls()` | `MLATokenToKVPool` | MLA KV cache pool class |
| `get_nsa_kv_pool_cls()` | `NSATokenToKVPool` | NSA KV cache pool class (DeepSeek V3.2) |
| `get_paged_allocator_cls()` | `PagedTokenToKVPoolAllocator` | Paged allocator class |
| `get_piecewise_backend_cls()` | `CUDAPiecewiseBackend` | Piecewise compilation backend class |
| `get_compile_backend(mode)` | `"inductor"` | Compilation backend string |
| `get_dispatch_key_name()` | `"native"` | MultiPlatformOp dispatch key name |

#### Lifecycle Hooks (Class Methods)

| Method | Invocation Timing | Purpose |
|---|---|---|
| `pre_register_and_update()` | Process startup (phase 1) | Custom CLI, etc. |
| `apply_global_patches()` | Process startup | Global monkey patches |
| `apply_server_args_defaults(server_args)` | After ServerArgs parsing (phase 2) | Set platform-specific defaults |
| `check_and_update_config(server_args)` | Configuration validation (phase 3) | Raise exceptions for incompatible configurations |
| `init_backend()` | On model runner import (before ModelRunner construction) |
| `apply_worker_patches()` | After Model Runner initialization (per worker) | Worker-level monkey patches |



## Plugin Type 2: General Function Plugin

### Description

General function plugins inject behavior into sglang **without requiring a custom platform**. Use cases include:

- **Observability**: Add logging, metrics, and tracing to any function
- **Behavior modification**: Modify function arguments or return values
- **Performance profiling**: Add timing to critical functions
- **A/B testing**: Replace implementations at runtime

### Insertion Points in the Main Framework

Two locations where `load_general_plugins()` + `HookRegistry.apply_hooks()` are called:

| Call Site | Process | Timing | Code Location |
|---------|------|------|---------|
| `_launch_subprocesses()` | Main process | After `_set_envs_and_config()`, before `server_args.check_server_args()` | `engine.py:998-1002` |
| `TpModelWorker.__init__()` | Worker subprocess | After `apply_worker_patches()`, before model inference | `tp_worker.py:251-255` |

### Hook Types

`HookRegistry` supports four hook types:

| Hook Type | Function Signature | Description |
|---|---|---|
| **BEFORE** | `fn(*args, **kwargs) -> (args, kwargs) \| None` | Runs before the original function. Return `None` to keep arguments unchanged, or return `(args, kwargs)` to modify them. |
| **AFTER** | `fn(result, *args, **kwargs) -> new_result \| None` | Runs after the original function. Return `None` to keep the result unchanged, or return a new value to replace it. |
| **AROUND** | `fn(original_fn, *args, **kwargs) -> result` | Wraps the original function. You must call `original_fn` yourself. Gives full control over execution. |
| **REPLACE** | `fn(*args, **kwargs) -> result` | Completely replaces the original function. |

Typical usage examples:

```python
# 1. Timing decorator
def time_it(original_fn, *args, **kwargs):
    start = time.perf_counter()
    result = original_fn(*args, **kwargs)   # Call original function
    print(f"Elapsed {time.perf_counter() - start:.3f}s")
    return result

# 2. Cache short-circuit (may skip calling the original function)
def cache_hit(original_fn, *args, **kwargs):
    cached = cache.get(args)
    if cached is not None:
        return cached              # Skip original_fn, return cached result
    result = original_fn(*args, **kwargs)
    cache.set(args, result)
    return result

# 3. Retry on exception (may call the original function multiple times)
def retry_on_error(original_fn, *args, **kwargs):
    for i in range(3):
        try:
            return original_fn(*args, **kwargs)
        except Exception:
            if i == 2: raise
```

### ClassReplacer

Used to replace entire classes (rather than individual functions):

```python
from sglang.srt.plugins.class_replacer import ClassReplacer

ClassReplacer.register(
    "sglang.srt.managers.scheduler.Scheduler",         # Original class
    "my_plugin.custom_scheduler.EnhancedScheduler"      # Replacement class
)
```
