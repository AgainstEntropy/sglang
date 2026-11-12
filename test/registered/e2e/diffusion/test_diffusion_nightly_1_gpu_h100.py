from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# Ming-Image (six cold starts, 31 min on B200) plus Qwen-Image 2.1 (3 min on B200).
register_cuda_ci(
    est_time=3000,
    stage="nightly",
    runner_config="diffusion-1-gpu-h100",
)


if __name__ == "__main__":
    run_diffusion_suite("nightly-1-gpu-h100")
