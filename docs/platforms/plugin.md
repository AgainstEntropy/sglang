# SGLang plugin

## 概述

参考 vLLM 的平台抽象。允许硬件厂商**无需修改主仓库代码**扩展 SGLang。

框架提供两种插件类型，均通过 Python 标准的 `setuptools` entry_points 机制发现和加载：

| 插件类型 | Entry Point Group | 用途 |
|---|---|---|
| **硬件平台插件** | `sglang.platform_plugins` | 注册自定义硬件平台（设备操作、KV 缓存池、注意力后端、CUDA Graph、编译后端等） |
| **通用函数插件** | `sglang.general_plugins` | 向 sglang 中的任意函数/方法注入钩子（before/after/around/replace），或替换整个类 |

### 原则

- **非侵入式**: 现有 CUDA/ROCm/NPU/XPU 代码保持不变。OOT 代码路径以 `elif` 分支的形式添加在已有硬件特定逻辑旁边。
- **零配置**: 插件在 `pip install` 后自动被发现，无需修改 sglang 代码。
- **SGLANG_PLUGINS 环境变量**: 通过逗号分隔的白名单控制加载哪些插件，硬件插件只能加载1个，通用插件可加载多个。

## 插件类型一：硬件平台插件

### 功能说明

硬件平台插件注册一个 `Platform` 子类，告诉 SGLang 如何与特定硬件后端交互。平台控制以下内容：

- **设备操作**: `set_device()`、`get_device_name()`、`get_device_total_memory()` 等
- **能力标志位**: `support_cuda_graph()`、`support_cublas()`、`support_kernel_warmup()`、`supports_fp8()` 等
- **子系统工厂方法**: 指定使用哪种 KV 缓存池、Graph Runner、内存分配器、编译后端和attention后端
- **配置生命周期钩子**: `apply_server_args_defaults()`、`init_backend()`、`apply_worker_patches()` 等
- **MultiPlatformOp 分发**: 兼容现有的MultiPlatformOp，通过key，分发到自定义的方法上。指定调用哪个 `forward_<key>()` 方法来执行融合算子

### Platform 接口

```python
from sglang.srt.platforms.interface import Platform, PlatformEnum

class MyPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "my_device"
    device_type: str = "cuda"          # torch 设备类型
    dispatch_key: str = "CUDA"         # PyTorch dispatch key
    device_control_env_var: str = "MY_VISIBLE_DEVICES"
    dist_backend: str = "nccl"         # 或 "hccl"、"gloo" 等
```

#### 身份查询（实例方法）

| 方法 | 默认值 | 说明 |
|---|---|---|
| `is_cuda_alike()` | OOT 默认为 `False` | 如果硬件支持类 CUDA API（启用 alt_stream 等），覆盖为 `True` |
| `is_out_of_tree()` | OOT 默认为 `True` | 根据 `_enum = PlatformEnum.OOT` 自动检测 |

#### 能力标志位（类方法）

| 方法 | 默认值 | 说明 |
|---|---|---|
| `support_cuda_graph()` | `True` | 是否支持设备 Graph 捕获 |
| `support_cublas()` | `False` | 是否初始化 cuBLAS |
| `support_kernel_warmup()` | `False` | 是否运行 kernel 预热 |
| `support_torch_compile()` | `True` | torch.compile 是否可用 |
| `supports_fp8()` | `False` | 是否支持 FP8 量化 |
| `is_pin_memory_available()` | `True` | 锁页内存是否可用 |

#### 子系统工厂方法（类方法）

| 方法 | 默认返回值 | 说明 |
|---|---|---|
| `get_default_attention_backend()` | `"flashinfer"` | 默认注意力后端名称 |
| `get_graph_runner_cls()` | `CudaGraphRunner` | Graph Runner 类 |
| `get_mha_kv_pool_cls()` | `MHATokenToKVPool` | MHA KV 缓存池类 |
| `get_mla_kv_pool_cls()` | `MLATokenToKVPool` | MLA KV 缓存池类 |
| `get_nsa_kv_pool_cls()` | `NSATokenToKVPool` | NSA KV 缓存池类（DeepSeek V3.2） |
| `get_paged_allocator_cls()` | `PagedTokenToKVPoolAllocator` | 分页分配器类 |
| `get_piecewise_backend_cls()` | `CUDAPiecewiseBackend` | 分段编译后端类 |
| `get_compile_backend(mode)` | `"inductor"` | 编译后端字符串 |
| `get_dispatch_key_name()` | `"native"` | MultiPlatformOp 分发键名 |

#### 生命周期钩子（类方法）

| 方法 | 调用时机 | 用途 |
|---|---|---|
| `pre_register_and_update()` | 进程启动（阶段 1） | 自定义CLI等 |
| `apply_global_patches()` | 进程启动时 | 全局 Monkey Patch |
| `apply_server_args_defaults(server_args)` | ServerArgs 解析后（阶段 2） | 设置平台特定的默认值 |
| `check_and_update_config(server_args)` | 配置校验（阶段 3） | 对不兼容的配置抛出异常 |
| `init_backend()` | import model runner时（ModelRunner构造前） |
| `apply_worker_patches()` | Model Runner 初始化后（每个 worker） | Worker 级 Monkey Patch |



## 插件类型二：通用函数插件

### 功能说明

通用函数插件可以在**不需要自定义平台**的情况下向 sglang 注入行为。适用场景包括：

- **可观测性**: 为任意函数添加日志、指标、链路追踪
- **行为修改**: 修改函数的参数或返回值
- **性能检测**: 为关键函数添加计时
- **A/B 测试**: 在运行时替换实现

### 主框架中的插入位置

两处调用 `load_general_plugins()` + `HookRegistry.apply_hooks()` 的位置：

| 调用位置 | 进程 | 时机 | 代码位置 |
|---------|------|------|---------|
| `_launch_subprocesses()` | 主进程 | `_set_envs_and_config()` 之后、`server_args.check_server_args()` 之前 | `engine.py:998-1002` |
| `TpModelWorker.__init__()` | Worker 子进程 | `apply_worker_patches()` 之后、模型推理之前 | `tp_worker.py:251-255` |

### Hook 类型

`HookRegistry` 支持四种 Hook 类型：

| Hook 类型 | 函数签名 | 说明 |
|---|---|---|
| **BEFORE** | `fn(*args, **kwargs) -> (args, kwargs) \| None` | 在原函数之前运行。返回 `None` 保持参数不变，或返回 `(args, kwargs)` 修改参数。 |
| **AFTER** | `fn(result, *args, **kwargs) -> new_result \| None` | 在原函数之后运行。返回 `None` 保持结果不变，或返回新值替换结果。 |
| **AROUND** | `fn(original_fn, *args, **kwargs) -> result` | 包裹原函数。你必须自行调用 `original_fn`。拥有对执行过程的完全控制。 |
| **REPLACE** | `fn(*args, **kwargs) -> result` | 完全替换原函数。 |

典型用法示例：

```python
# 1. 计时装饰器
def time_it(original_fn, *args, **kwargs):
    start = time.perf_counter()
    result = original_fn(*args, **kwargs)   # 调用原函数
    print(f"耗时 {time.perf_counter() - start:.3f}s")
    return result

# 2. 缓存短路（可以不调用原函数）
def cache_hit(original_fn, *args, **kwargs):
    cached = cache.get(args)
    if cached is not None:
        return cached              # 不调用 original_fn，直接返回缓存
    result = original_fn(*args, **kwargs)
    cache.set(args, result)
    return result

# 3. 异常重试（可以多次调用原函数）
def retry_on_error(original_fn, *args, **kwargs):
    for i in range(3):
        try:
            return original_fn(*args, **kwargs)
        except Exception:
            if i == 2: raise
```

### ClassReplacer

用于替换整个类（而非单个函数）：

```python
from sglang.srt.plugins.class_replacer import ClassReplacer

ClassReplacer.register(
    "sglang.srt.managers.scheduler.Scheduler",         # 原始类
    "my_plugin.custom_scheduler.EnhancedScheduler"      # 替换类
)
```

