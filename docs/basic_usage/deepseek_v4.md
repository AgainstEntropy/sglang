# DeepSeek V4 Usage

This page covers the original [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) checkpoint on AMD MI355X.
The original checkpoint keeps routed experts in MXFP4 and other weights in FP8.
It is different from FP8-converted checkpoints such as `sgl-project/DeepSeek-V4-Flash-FP8`.

## Launch on MI355X

The original checkpoint needs FP4 expert handling enabled and the ROCm/AITER MoE path active:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SHM_DISABLE=1

export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_DSV4_FP4_EXPERTS=true
export SGLANG_FORCE_TRITON_MOE_FP8=0

export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false

python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tp 4 \
  --ep 4 \
  --dp 4 \
  --enable-dp-attention \
  --disable-radix-cache \
  --attention-backend compressed \
  --moe-runner-backend triton \
  --max-running-request 256 \
  --page-size 256 \
  --chunked-prefill-size 8192 \
  --disable-shared-experts-fusion \
  --disable-cuda-graph \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4
```

For FP8-converted checkpoints, set `SGLANG_DSV4_FP4_EXPERTS=false` and `SGLANG_FORCE_TRITON_MOE_FP8=1`.
