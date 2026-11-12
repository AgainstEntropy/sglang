from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# 6.5 min warm on 4x B200; not yet measured on H100.
register_cuda_ci(
    est_time=1500,
    stage="nightly",
    runner_config="diffusion-4-gpu-h100",
)


if __name__ == "__main__":
    run_diffusion_suite("nightly-4-gpu-h100")
