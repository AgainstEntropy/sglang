from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# About 12 min warm (Kandinsky 6 plus the 11 family cases); the rest covers
# a cold download of ~510 GB of checkpoints.
register_cuda_ci(
    est_time=3000,
    stage="nightly",
    runner_config="diffusion-1-gpu-b200",
)


if __name__ == "__main__":
    run_diffusion_suite("nightly-1-gpu-b200")
