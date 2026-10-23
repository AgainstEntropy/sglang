"""EAGLE / MTP on an out-of-tree platform with the native speculative kernels.

A throwaway platform plugin re-registers the CUDA platform as out-of-tree and
routes every speculative-decoding fused op to its forward_native, so the
built-in EAGLE worker serves a draft tree end to end the way it would on a
device that brings no speculative kernels of its own.
"""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

MODEL = "XiaomiMiMo/MiMo-7B-RL"
PLUGIN_NAME = "oot_eagle_native_test"

_PLUGIN_SOURCE = """
import os

from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.device_mixin import PlatformEnum

SPEC_OPS = (
    ("sglang.kernels.ops.speculative.spec_tree", "BuildEagleTreeOp"),
    ("sglang.kernels.ops.speculative.spec_tree", "VerifyEagleTreeGreedyOp"),
    ("sglang.kernels.ops.speculative.cache_locs", "AssignDraftCacheLocsOp"),
    ("sglang.kernels.ops.speculative.cache_locs", "FillFrontOutCacheLocOp"),
    ("sglang.kernels.ops.speculative.cache_locs", "AssignReqToTokenPoolOp"),
    ("sglang.kernels.ops.speculative.eagle", "FillBonusTokensOp"),
    ("sglang.kernels.ops.speculative.gather_spec_extras", "GatherSpecExtrasOp"),
    ("sglang.kernels.ops.speculative.verify_sampling", "TreeSpeculativeSamplingOp"),
    ("sglang.kernels.ops.speculative.verify_sampling", "ChainSpeculativeSamplingOp"),
    ("sglang.kernels.ops.sampling.renorm", "TopKRenormProbOp"),
    ("sglang.kernels.ops.sampling.renorm", "TopPRenormProbOp"),
)

class CudaAsOutOfTreePlatform(CudaSRTPlatform):
    _enum = PlatformEnum.OOT

    def get_default_attention_backend(self):
        return "fa3"

    def get_default_speculative_draft_attention_backend(self, algorithm):
        return "fa3"

    def supports_speculative_algorithm(self, algorithm):
        return algorithm == "EAGLE"

    def get_dispatch_key_name(self):
        # Model layers keep their CUDA forwards; only the spec ops go native.
        return "cuda"

    def init_backend(self):
        import importlib

        from sglang.kernels.fused_op import BaseFusedOp

        super().init_backend()
        for module, name in SPEC_OPS:
            op_cls = getattr(importlib.import_module(module), name)
            BaseFusedOp.register_oot_forward(op_cls, op_cls.forward_native, "cuda")
        with open(os.environ["OOT_EAGLE_NATIVE_MARKER"], "a") as f:
            f.write(f"{os.getpid()}\\n")


def activate():
    return __name__ + ".CudaAsOutOfTreePlatform"
"""


def _write_plugin(root: Path) -> None:
    (root / f"{PLUGIN_NAME}.py").write_text(_PLUGIN_SOURCE)
    dist_info = root / f"{PLUGIN_NAME}-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {PLUGIN_NAME}\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        textwrap.dedent(
            f"""\
            [sglang.srt.platforms]
            {PLUGIN_NAME} = {PLUGIN_NAME}:activate
            """
        )
    )


class TestOutOfTreeEagleNative(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.plugin_dir = tempfile.TemporaryDirectory()
        root = Path(cls.plugin_dir.name)
        _write_plugin(root)
        cls.marker = root / "registered.txt"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(root), env.get("PYTHONPATH")])
        )
        env["SGLANG_PLATFORM"] = PLUGIN_NAME
        env["OOT_EAGLE_NATIVE_MARKER"] = str(cls.marker)
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=900,
            other_args=[
                "--trust-remote-code",
                "--speculative-algorithm",
                "EAGLE",
                # A draft tree (topk > 1), so tree construction, tree greedy
                # verification and tree sampling all run their native forwards.
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "2",
                "--speculative-num-draft-tokens",
                "4",
                "--mem-fraction-static",
                "0.7",
                # The native references sync with the host; they cannot be captured.
                "--disable-cuda-graph",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.plugin_dir.cleanup()

    def _gsm8k(self, **sampling):
        requests.get(self.base_url + "/flush_cache")
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=MODEL,
                eval_name="gsm8k",
                num_examples=64,
                num_threads=64,
                max_tokens=4096,
                seed=0,
                **sampling,
            )
        )
        server_info = requests.get(self.base_url + "/server_info").json()
        accept_length = server_info["internal_states"][0]["avg_spec_accept_length"]
        print(f"{sampling=} {metrics=} {accept_length=}")
        return metrics["score"], accept_length

    def test_a_runs_on_the_out_of_tree_platform(self):
        self.assertTrue(self.marker.exists())

    def test_b_greedy_verification(self):
        score, accept_length = self._gsm8k(temperature=0.0)
        self.assertGreater(score, 0.85)
        self.assertGreater(accept_length, 1.5)

    def test_c_sampled_verification(self):
        # MiMo-7B-RL's recommended sampling settings.
        score, accept_length = self._gsm8k(temperature=0.6, top_p=0.95)
        self.assertGreater(score, 0.85)
        self.assertGreater(accept_length, 1.5)


if __name__ == "__main__":
    unittest.main()
