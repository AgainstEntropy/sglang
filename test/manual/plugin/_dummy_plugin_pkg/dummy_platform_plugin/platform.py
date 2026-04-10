"""Dummy SRTPlatform subclass for integration tests."""

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class DummyDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "dummy"
    device_type = "cpu"

    def get_device_total_memory(self, device_id=0):
        return 16 * (1 << 30)  # 16 GiB

    def get_current_memory_usage(self, device=None):
        return 0.0


class DummySRTPlatform(SRTPlatform, DummyDeviceMixin):

    def get_default_attention_backend(self):
        return "dummy_attn"

    def support_cuda_graph(self):
        return False

    def support_piecewise_cuda_graph(self):
        return False

    def supports_fp8(self):
        return True

    def get_compile_backend(self, mode=None):
        return "dummy_inductor"

    def get_dispatch_key_name(self):
        return "dummy"

    def apply_server_args_defaults(self, server_args):
        server_args._dummy_plugin_applied = True
