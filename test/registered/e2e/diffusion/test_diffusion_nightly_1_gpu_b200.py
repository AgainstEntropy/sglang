from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# 2 min warm; the rest covers a cold ~80 GB download (server wait is 2400 s).
register_cuda_ci(
    est_time=900,
    stage="nightly",
    runner_config="diffusion-1-gpu-b200",
)


if __name__ == "__main__":
    run_diffusion_suite("nightly-1-gpu-b200")
