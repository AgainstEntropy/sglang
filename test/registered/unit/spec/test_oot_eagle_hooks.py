import contextlib
import sys
import types
import unittest
from unittest.mock import Mock, patch, sentinel

from sglang.srt.arg_groups.speculative_hook import _check_oot_speculative_algorithm
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_worker_v2 import _select_spec_graph_runner_cls
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"
WORKER_MODULE = "sglang.srt.speculative.eagle_worker_v2"
DRAFT_UTILS_MODULE = "sglang.srt.speculative.draft_utils"


def _platform(
    *,
    out_of_tree: bool = True,
    supported: bool = True,
    worker_cls=None,
    graph: bool = True,
    runner_cls=None,
) -> Mock:
    platform = Mock(spec=SRTPlatform)
    platform.is_out_of_tree.return_value = out_of_tree
    platform.supports_speculative_algorithm.return_value = supported
    platform.get_speculative_worker_cls.return_value = worker_cls
    platform.support_cuda_graph.return_value = graph
    platform.get_speculative_graph_runner_cls.return_value = runner_cls
    return platform


def _unreachable_builtin():
    raise KeyError("custom")


class TestOOTEagleWorkerSelection(CustomTestCase):
    def _create_worker(self, platform: Mock):
        args = ServerArgs(model_path="dummy", device="custom")
        builtin = types.ModuleType(WORKER_MODULE)
        builtin.EAGLEWorkerV2 = sentinel.builtin_worker
        with (
            patch("sglang.srt.platforms.current_platform", platform),
            patch.dict(sys.modules, {WORKER_MODULE: builtin}),
        ):
            return SpeculativeAlgorithm.EAGLE.create_worker(args)

    def test_platform_worker_replaces_builtin(self):
        platform = _platform(worker_cls=sentinel.platform_worker)
        self.assertIs(self._create_worker(platform), sentinel.platform_worker)
        platform.get_speculative_worker_cls.assert_called_once_with(algorithm="EAGLE")

    def test_builtin_worker_when_platform_declines(self):
        declining = _platform(worker_cls=None)
        self.assertIs(self._create_worker(declining), sentinel.builtin_worker)
        in_tree = _platform(out_of_tree=False, worker_cls=sentinel.platform_worker)
        self.assertIs(self._create_worker(in_tree), sentinel.builtin_worker)
        in_tree.get_speculative_worker_cls.assert_not_called()


class TestOOTEagleAlgorithmGate(CustomTestCase):
    def test_unsupported_out_of_tree_platform_is_rejected(self):
        with patch(f"{HOOK_MODULE}.current_platform", _platform(supported=False)):
            with self.assertRaisesRegex(ValueError, "EAGLE speculative decoding"):
                _check_oot_speculative_algorithm(algorithm="EAGLE", device="custom")

    def test_builtin_platforms_are_not_gated(self):
        platform = _platform(out_of_tree=False, supported=False)
        with patch(f"{HOOK_MODULE}.current_platform", platform):
            _check_oot_speculative_algorithm(algorithm="EAGLE", device="cuda")
        platform.supports_speculative_algorithm.assert_not_called()


class TestOOTSpecGraphRunnerSelection(CustomTestCase):
    def _select(self, platform: Mock, phase: str = "draft_decode"):
        with patch(f"{WORKER_MODULE}.current_platform", platform):
            return _select_spec_graph_runner_cls(
                builtin_runner_cls=_unreachable_builtin,
                algorithm="EAGLE",
                phase=phase,
            )

    def test_out_of_tree_device_never_indexes_builtin_runners(self):
        """An OOT device name is not a key of the built-in runner maps."""
        self.assertIsNone(self._select(_platform(graph=False)))
        self.assertIsNone(self._select(_platform(runner_cls=None)))

    def test_platform_runner_per_phase(self):
        for phase in ("draft_decode", "draft_extend", "target_verify"):
            with self.subTest(phase=phase):
                platform = _platform(runner_cls=sentinel.runner)
                self.assertIs(self._select(platform, phase), sentinel.runner)
                platform.get_speculative_graph_runner_cls.assert_called_once_with(
                    algorithm="EAGLE", phase=phase
                )

    def test_builtin_platform_keeps_device_map(self):
        platform = _platform(out_of_tree=False)
        with patch(f"{WORKER_MODULE}.current_platform", platform):
            selected = _select_spec_graph_runner_cls(
                builtin_runner_cls=lambda: sentinel.builtin_runner,
                algorithm="EAGLE",
                phase="draft_decode",
            )
        self.assertIs(selected, sentinel.builtin_runner)
        platform.get_speculative_graph_runner_cls.assert_not_called()


class TestOOTDraftAttentionBackend(CustomTestCase):
    def _factory(self, backend_name: str, platform: Mock):
        runner = Mock()
        runner.draft_attention_backend = backend_name
        spec = Mock(speculative_algorithm="EAGLE", speculative_attention_mode="prefill")
        patches = (
            patch("sglang.srt.platforms.current_platform", platform),
            patch(
                f"{DRAFT_UTILS_MODULE}.attention_backends", return_value=(None, None)
            ),
            patch(f"{DRAFT_UTILS_MODULE}.get_spec", return_value=spec),
            patch(f"{DRAFT_UTILS_MODULE}._assert_draft_needs_no_conv_sidecar"),
            patch(
                "sglang.srt.layers.attention.attention_registry."
                "attn_backend_wrapper_for_draft_extend",
                side_effect=lambda _runner, backend: backend,
            ),
        )
        return DraftBackendFactory(runner, topk=1, speculative_num_steps=3), patches

    def _run(self, backend_name: str, platform: Mock, method: str):
        factory, patches = self._factory(backend_name, platform)
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            return getattr(factory, method)()

    def test_unknown_backend_is_created_by_platform(self):
        backend = Mock()
        backend.attn_backends = [Mock(), Mock()]
        platform = _platform()
        platform.create_speculative_draft_attention_backend.return_value = backend
        for method, phase in (
            ("create_draft_extend_backend", "draft_extend"),
            ("create_decode_backend", "draft_decode"),
        ):
            with self.subTest(phase=phase):
                platform.create_speculative_draft_attention_backend.reset_mock()
                created = self._run("custom", platform, method)
                self.assertIs(created, backend)
                self.assertEqual(created.prefill_attention_backend_str, "custom")
                kwargs = (
                    platform.create_speculative_draft_attention_backend.call_args.kwargs
                )
                self.assertEqual(kwargs["phase"], phase)
                self.assertEqual(kwargs["speculative_num_steps"], 3)

    def test_unimplemented_platform_keeps_builtin_error(self):
        platform = _platform()
        platform.create_speculative_draft_attention_backend.side_effect = (
            NotImplementedError
        )
        with self.assertRaisesRegex(
            ValueError, "not supported in attention backend custom"
        ):
            self._run("custom", platform, "create_draft_extend_backend")


if __name__ == "__main__":
    unittest.main()
