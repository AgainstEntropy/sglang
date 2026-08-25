# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.vla import VLAObservationSamplingParams


@dataclass
class LingbotVlaV2SamplingParams(VLAObservationSamplingParams):
    """Sampling parameters for LingBot-VLA-V2 flow-matching action inference."""

    num_inference_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 55

    def _set_output_file_name(self):
        if self.output_file_name is None:
            self.output_file_name = "lingbot_vla_v2_action"
        super()._set_output_file_name()
