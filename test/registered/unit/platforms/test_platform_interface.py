"""
Unit tests for SGLang platform abstraction layer.

Tests DeviceMixin, SRTPlatform, PlatformEnum, CpuArchEnum, and DeviceCapability.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.platforms.device_mixin import (
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestPlatformEnum(unittest.TestCase):
    """Tests for PlatformEnum enumeration."""

    def test_all_expected_values_exist(self):
        """Test that all expected platform enum values exist."""
        expected_values = [
            "CUDA",
            "ROCM",
            "CPU",
            "XPU",
            "MUSA",
            "NPU",
            "TPU",
            "MPS",
            "OOT",
            "UNSPECIFIED",
        ]
        actual_values = [member.name for member in PlatformEnum]
        for expected in expected_values:
            self.assertIn(expected, actual_values)


class TestCpuArchEnum(unittest.TestCase):
    """Tests for CpuArchEnum enumeration."""

    def test_all_expected_values_exist(self):
        """Test that all expected CPU architecture enum values exist."""
        expected_values = ["X86", "ARM", "UNSPECIFIED"]
        actual_values = [member.name for member in CpuArchEnum]
        for expected in expected_values:
            self.assertIn(expected, actual_values)


class TestDeviceCapability(unittest.TestCase):
    """Tests for DeviceCapability NamedTuple."""

    def test_constructor(self):
        """Test DeviceCapability constructor with valid values."""
        cap = DeviceCapability(major=9, minor=0)
        self.assertEqual(cap.major, 9)
        self.assertEqual(cap.minor, 0)

    def test_constructor_edge_cases(self):
        """Test DeviceCapability with edge case minor values."""
        # Test minimum minor
        cap_min = DeviceCapability(major=8, minor=0)
        self.assertEqual(cap_min.minor, 0)
        # Test maximum minor (single digit)
        cap_max = DeviceCapability(major=8, minor=9)
        self.assertEqual(cap_max.minor, 9)

    def test_as_version_str(self):
        """Test as_version_str returns correct format."""
        cap = DeviceCapability(major=9, minor=0)
        self.assertEqual(cap.as_version_str(), "9.0")

        cap2 = DeviceCapability(major=8, minor=9)
        self.assertEqual(cap2.as_version_str(), "8.9")

    def test_to_int(self):
        """Test to_int returns correct integer representation."""
        cap = DeviceCapability(major=9, minor=0)
        self.assertEqual(cap.to_int(), 90)

        cap2 = DeviceCapability(major=8, minor=9)
        self.assertEqual(cap2.to_int(), 89)

    def test_to_int_min_values(self):
        """Test to_int with minimum values."""
        cap = DeviceCapability(major=0, minor=0)
        self.assertEqual(cap.to_int(), 0)

    def test_comparison(self):
        """Test DeviceCapability comparison operations."""
        cap_90 = DeviceCapability(9, 0)
        cap_89 = DeviceCapability(8, 9)
        cap_80 = DeviceCapability(8, 0)
        cap_90_copy = DeviceCapability(9, 0)

        # Test greater than
        self.assertTrue(cap_90 > cap_89)
        self.assertTrue(cap_89 > cap_80)

        # Test less than
        self.assertTrue(cap_89 < cap_90)
        self.assertTrue(cap_80 < cap_89)

        # Test equality
        self.assertEqual(cap_90, cap_90_copy)
        self.assertNotEqual(cap_90, cap_89)


class TestDeviceMixin(unittest.TestCase):
    """Tests for DeviceMixin base class."""

    def setUp(self):
        """Create a concrete subclass for testing."""
        # Create a concrete implementation of DeviceMixin for testing
        class ConcreteDeviceMixin(DeviceMixin):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        self.mixin = ConcreteDeviceMixin()

    def test_is_cuda_true(self):
        """Test is_cuda returns True for CUDA platform."""
        self.assertTrue(self.mixin.is_cuda())

    def test_is_cuda_false(self):
        """Test is_cuda returns False for non-CUDA platform."""
        class NonCudaMixin(DeviceMixin):
            _enum = PlatformEnum.CPU
            device_name = "cpu"
            device_type = "cpu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = NonCudaMixin()
        self.assertFalse(mixin.is_cuda())

    def test_is_rocm_true(self):
        """Test is_rocm returns True for ROCm platform."""
        class RocmMixin(DeviceMixin):
            _enum = PlatformEnum.ROCM
            device_name = "rocm"
            device_type = "hip"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = RocmMixin()
        self.assertTrue(mixin.is_rocm())

    def test_is_cpu_true(self):
        """Test is_cpu returns True for CPU platform."""
        class CpuMixin(DeviceMixin):
            _enum = PlatformEnum.CPU
            device_name = "cpu"
            device_type = "cpu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = CpuMixin()
        self.assertTrue(mixin.is_cpu())

    def test_is_xpu_true(self):
        """Test is_xpu returns True for XPU platform."""
        class XpuMixin(DeviceMixin):
            _enum = PlatformEnum.XPU
            device_name = "xpu"
            device_type = "xpu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = XpuMixin()
        self.assertTrue(mixin.is_xpu())

    def test_is_musa_true(self):
        """Test is_musa returns True for MUSA platform."""
        class MusaMixin(DeviceMixin):
            _enum = PlatformEnum.MUSA
            device_name = "musa"
            device_type = "musa"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = MusaMixin()
        self.assertTrue(mixin.is_musa())

    def test_is_npu_true(self):
        """Test is_npu returns True for NPU platform."""
        class NpuMixin(DeviceMixin):
            _enum = PlatformEnum.NPU
            device_name = "npu"
            device_type = "npu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = NpuMixin()
        self.assertTrue(mixin.is_npu())

    def test_is_tpu_true(self):
        """Test is_tpu returns True for TPU platform."""
        class TpuMixin(DeviceMixin):
            _enum = PlatformEnum.TPU
            device_name = "tpu"
            device_type = "tpu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = TpuMixin()
        self.assertTrue(mixin.is_tpu())

    def test_is_mps_true(self):
        """Test is_mps returns True for MPS platform."""
        class MpsMixin(DeviceMixin):
            _enum = PlatformEnum.MPS
            device_name = "mps"
            device_type = "mps"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = MpsMixin()
        self.assertTrue(mixin.is_mps())

    def test_is_cuda_alike_true_for_cuda(self):
        """Test is_cuda_alike returns True for CUDA."""
        self.assertTrue(self.mixin.is_cuda_alike())

    def test_is_cuda_alike_true_for_rocm(self):
        """Test is_cuda_alike returns True for ROCm."""
        class RocmMixin(DeviceMixin):
            _enum = PlatformEnum.ROCM
            device_name = "rocm"
            device_type = "hip"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = RocmMixin()
        self.assertTrue(mixin.is_cuda_alike())

    def test_is_cuda_alike_true_for_musa(self):
        """Test is_cuda_alike returns True for MUSA."""
        class MusaMixin(DeviceMixin):
            _enum = PlatformEnum.MUSA
            device_name = "musa"
            device_type = "musa"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = MusaMixin()
        self.assertTrue(mixin.is_cuda_alike())

    def test_is_cuda_alike_false_for_cpu(self):
        """Test is_cuda_alike returns False for CPU."""
        class CpuMixin(DeviceMixin):
            _enum = PlatformEnum.CPU
            device_name = "cpu"
            device_type = "cpu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = CpuMixin()
        self.assertFalse(mixin.is_cuda_alike())

    def test_is_out_of_tree_true(self):
        """Test is_out_of_tree returns True for OOT platform."""
        class OotMixin(DeviceMixin):
            _enum = PlatformEnum.OOT
            device_name = "custom"
            device_type = "custom"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

        mixin = OotMixin()
        self.assertTrue(mixin.is_out_of_tree())

    def test_is_out_of_tree_false(self):
        """Test is_out_of_tree returns False for non-OOT platform."""
        self.assertFalse(self.mixin.is_out_of_tree())

    def test_empty_cache_noop(self):
        """Test empty_cache is a no-op and does not raise."""
        # Should not raise any exception
        self.mixin.empty_cache()

    def test_synchronize_noop(self):
        """Test synchronize is a no-op and does not raise."""
        # Should not raise any exception
        self.mixin.synchronize()

    def test_get_device_raises_not_implemented(self):
        """Test get_device raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_device(0)

    def test_set_device_raises_not_implemented(self):
        """Test set_device raises NotImplementedError."""
        import torch
        with self.assertRaises(NotImplementedError):
            self.mixin.set_device(torch.device("cuda"))

    def test_get_device_name_raises_not_implemented(self):
        """Test get_device_name raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_device_name(0)

    def test_get_device_uuid_raises_not_implemented(self):
        """Test get_device_uuid raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_device_uuid(0)

    def test_get_device_capability_raises_not_implemented(self):
        """Test get_device_capability raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_device_capability(0)

    def test_get_available_memory_raises_not_implemented(self):
        """Test get_available_memory raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_available_memory(0)

    def test_get_torch_distributed_backend_str_raises_not_implemented(self):
        """Test get_torch_distributed_backend_str raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.mixin.get_torch_distributed_backend_str()

    def test_get_communicator_class_returns_none(self):
        """Test get_communicator_class returns None by default."""
        result = self.mixin.get_communicator_class()
        self.assertIsNone(result)

    def test_inference_mode_returns_context_manager(self):
        """Test inference_mode returns a context manager."""
        cm = self.mixin.inference_mode()
        # Should be usable as a context manager
        self.assertTrue(hasattr(cm, "__enter__"))
        self.assertTrue(hasattr(cm, "__exit__"))

    def test_seed_everything_with_none(self):
        """Test seed_everything with None does nothing."""
        # Should not raise
        self.mixin.seed_everything(None)

    def test_seed_everything_with_valid_seed(self):
        """Test seed_everything with a valid seed sets random states."""
        # Should not raise
        self.mixin.seed_everything(42)

    def test_verify_quantization_noop(self):
        """Test verify_quantization is a no-op."""
        # Should not raise any exception
        self.mixin.verify_quantization("fp8")
        self.mixin.verify_quantization("int8")

    @patch("platform.machine")
    def test_get_cpu_architecture_x86(self, mock_machine):
        """Test get_cpu_architecture returns X86 for x86_64."""
        mock_machine.return_value = "x86_64"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.X86)

    @patch("platform.machine")
    def test_get_cpu_architecture_amd64(self, mock_machine):
        """Test get_cpu_architecture returns X86 for amd64."""
        mock_machine.return_value = "amd64"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.X86)

    @patch("platform.machine")
    def test_get_cpu_architecture_i386(self, mock_machine):
        """Test get_cpu_architecture returns X86 for i386."""
        mock_machine.return_value = "i386"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.X86)

    @patch("platform.machine")
    def test_get_cpu_architecture_i686(self, mock_machine):
        """Test get_cpu_architecture returns X86 for i686."""
        mock_machine.return_value = "i686"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.X86)

    @patch("platform.machine")
    def test_get_cpu_architecture_arm64(self, mock_machine):
        """Test get_cpu_architecture returns ARM for arm64."""
        mock_machine.return_value = "arm64"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.ARM)

    @patch("platform.machine")
    def test_get_cpu_architecture_aarch64(self, mock_machine):
        """Test get_cpu_architecture returns ARM for aarch64."""
        mock_machine.return_value = "aarch64"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.ARM)

    @patch("platform.machine")
    def test_get_cpu_architecture_unknown(self, mock_machine):
        """Test get_cpu_architecture returns UNSPECIFIED for unknown arch."""
        mock_machine.return_value = "unknown_arch"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.UNSPECIFIED)

    @patch("platform.machine")
    def test_get_cpu_architecture_case_insensitive(self, mock_machine):
        """Test get_cpu_architecture is case insensitive."""
        mock_machine.return_value = "X86_64"
        result = DeviceMixin.get_cpu_architecture()
        self.assertEqual(result, CpuArchEnum.X86)

    def test_repr(self):
        """Test __repr__ returns expected format."""
        repr_str = repr(self.mixin)
        self.assertIn("ConcreteDeviceMixin", repr_str)
        self.assertIn("cuda", repr_str)


class TestSRTPlatform(unittest.TestCase):
    """Tests for SRTPlatform base class."""

    def setUp(self):
        """Create a concrete subclass for testing."""
        class ConcreteSRTPlatform(SRTPlatform):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

        self.platform = ConcreteSRTPlatform()

    def test_apply_server_args_defaults_noop(self):
        """Test apply_server_args_defaults is a no-op."""
        mock_args = MagicMock()
        self.platform.apply_server_args_defaults(mock_args)
        # Should not raise and not modify args

    def test_supported_quantization_empty_list(self):
        """Test supported_quantization defaults to empty list."""
        self.assertEqual(self.platform.supported_quantization, [])

    def test_get_default_attention_backend_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_default_attention_backend()

    def test_get_graph_runner_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_graph_runner_cls()

    def test_get_mha_kv_pool_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_mha_kv_pool_cls()

    def test_get_mla_kv_pool_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_mla_kv_pool_cls()

    def test_get_nsa_kv_pool_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_nsa_kv_pool_cls()

    def test_get_paged_allocator_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_paged_allocator_cls()

    def test_get_piecewise_backend_cls_raises_not_implemented(self):
        """Test base class raises NotImplementedError."""
        base = SRTPlatform()
        with self.assertRaises(NotImplementedError):
            base.get_piecewise_backend_cls()

    def test_get_compile_backend_default(self):
        """Test get_compile_backend returns 'inductor' by default."""
        base = SRTPlatform()
        result = base.get_compile_backend()
        self.assertEqual(result, "inductor")

    def test_get_compile_backend_with_mode(self):
        """Test get_compile_backend accepts mode parameter."""
        base = SRTPlatform()
        result = base.get_compile_backend(mode="npugraph_ex")
        self.assertEqual(result, "inductor")

    def test_supports_fp8_default_false(self):
        """Test supports_fp8 returns False by default."""
        base = SRTPlatform()
        self.assertFalse(base.supports_fp8())

    def test_is_pin_memory_available_default_true(self):
        """Test is_pin_memory_available returns True by default."""
        base = SRTPlatform()
        self.assertTrue(base.is_pin_memory_available())

    def test_support_cuda_graph_default_false(self):
        """Test support_cuda_graph returns False by default."""
        base = SRTPlatform()
        self.assertFalse(base.support_cuda_graph())

    def test_support_piecewise_cuda_graph_default_false(self):
        """Test support_piecewise_cuda_graph returns False by default."""
        base = SRTPlatform()
        self.assertFalse(base.support_piecewise_cuda_graph())

    def test_init_backend_noop(self):
        """Test init_backend is a no-op."""
        base = SRTPlatform()
        # Should not raise
        base.init_backend()

    def test_get_dispatch_key_name_default_native(self):
        """Test get_dispatch_key_name returns 'native' by default."""
        base = SRTPlatform()
        result = base.get_dispatch_key_name()
        self.assertEqual(result, "native")


class TestSRTPlatformOverrides(unittest.TestCase):
    """Tests for SRTPlatform methods that can be overridden."""

    def test_custom_supports_fp8(self):
        """Test platform can override supports_fp8."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

            def supports_fp8(self) -> bool:
                return True

        platform = CustomPlatform()
        self.assertTrue(platform.supports_fp8())

    def test_custom_support_cuda_graph(self):
        """Test platform can override support_cuda_graph."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

            def support_cuda_graph(self) -> bool:
                return True

        platform = CustomPlatform()
        self.assertTrue(platform.support_cuda_graph())

    def test_custom_support_piecewise_cuda_graph(self):
        """Test platform can override support_piecewise_cuda_graph."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

            def support_piecewise_cuda_graph(self) -> bool:
                return True

        platform = CustomPlatform()
        self.assertTrue(platform.support_piecewise_cuda_graph())

    def test_custom_get_dispatch_key_name(self):
        """Test platform can override get_dispatch_key_name."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.NPU
            device_name = "npu"
            device_type = "npu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

            def get_dispatch_key_name(self) -> str:
                return "npu"

        platform = CustomPlatform()
        self.assertEqual(platform.get_dispatch_key_name(), "npu")

    def test_custom_get_compile_backend(self):
        """Test platform can override get_compile_backend."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.NPU
            device_name = "npu"
            device_type = "npu"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

            def get_compile_backend(self, mode: str | None = None) -> str:
                return "inductor"

        platform = CustomPlatform()
        result = platform.get_compile_backend(mode="npugraph_ex")
        self.assertEqual(result, "inductor")


class TestSRTPlatformInheritance(unittest.TestCase):
    """Tests for SRTPlatform inheritance from DeviceMixin."""

    def test_inherits_device_mixin_methods(self):
        """Test SRTPlatform inherits DeviceMixin methods."""

        class CustomPlatform(SRTPlatform):
            _enum = PlatformEnum.CUDA
            device_name = "cuda"
            device_type = "cuda"

            def get_device_total_memory(self, device_id: int = 0) -> int:
                return 1000000000

            def get_current_memory_usage(self, device=None) -> float:
                return 500000000.0

            def get_default_attention_backend(self) -> str:
                return "flashinfer"

            def get_graph_runner_cls(self) -> type:
                return object

            def get_mha_kv_pool_cls(self) -> type:
                return object

            def get_mla_kv_pool_cls(self) -> type:
                return object

            def get_nsa_kv_pool_cls(self) -> type:
                return object

            def get_paged_allocator_cls(self) -> type:
                return object

            def get_piecewise_backend_cls(self) -> type:
                return object

        platform = CustomPlatform()
        # Test inherited methods
        self.assertTrue(platform.is_cuda())
        self.assertFalse(platform.is_cpu())
        self.assertTrue(platform.is_cuda_alike())
        self.assertFalse(platform.is_out_of_tree())

    def test_srt_platform_has_all_device_mixin_attrs(self):
        """Test SRTPlatform has all DeviceMixin class attributes."""
        # Check class attributes exist
        self.assertTrue(hasattr(SRTPlatform, "_enum"))
        self.assertTrue(hasattr(SRTPlatform, "device_name"))
        self.assertTrue(hasattr(SRTPlatform, "device_type"))
        self.assertTrue(hasattr(SRTPlatform, "supported_quantization"))


if __name__ == "__main__":
    unittest.main()
