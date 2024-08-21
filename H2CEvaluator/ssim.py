from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import PartialState

from .dist_utils import gather_all_tensors

try:
    from pytorch_msssim import SSIM as SSIM_Metric

except ImportError:
    print(
        "pytorch_msssim in not installed, please install it via "
        "`pip install pytorch-msssim`"
    )

SAMPLE_TYPE = Union[torch.Tensor, dict]


class SSIM:
    def __init__(self):
        self.ssim = SSIM_Metric(data_range=1.0, size_average=True, channel=3).cuda()
        self.fake_list = []
        self.score_list = []

    def prepare(self, *args, **kwargs):
        """Do not need prepare. Do nothing."""
        return

    def run_evaluation(self):
        score_list = torch.cat(self.score_list)
        if PartialState().use_distributed:
            score = gather_all_tensors(score_list)
            score = torch.cat(score)
        else:
            score = score_list

        score = torch.mean(score).item()
        self.score_list.clear()
        result_dict = {"ssim": score}
        return result_dict

    @torch.no_grad()
    def feed_one_sample(self, sample: SAMPLE_TYPE, mode: str):
        """Feed one sample.
        Args:
            sample (torch.Tensor | dict): If sample is tensor, sample should be
                [F, C, H, W], order in RGB, range in (0, 1). Otherwise, is dict
                with list of np.ndarray. The length of list is F and all elements
                are un-processed, in [0, 255], [B, H, W, C].
        """
        if mode == "fake":
            self.fake_list.append(sample)  # [0, 1]

        elif mode == "real":
            src_samples = sample["driving_video"]
            src_samples = [
                torch.from_numpy(np.array(src_sample)).float()
                for src_sample in src_samples
            ]
            src_samples = torch.stack(src_samples).permute(0, 3, 1, 2).float()
            src_samples = src_samples / 255  # [0, 1]

            assert len(self.fake_list) == 1, (
                "When call feed_one_sample with mode `real`, "
                "SSIM.fake_list should only contain one element. "
                "Please check your code!"
            )

            fake_samples = self.fake_list.pop().cuda()
            if fake_samples.shape[-2:] != src_samples.shape[-2]:
                fake_samples = F.interpolate(fake_samples, src_samples.shape[-2:])

            score = self.ssim(src_samples.cuda(), fake_samples)
            self.score_list.append(score[None])

        else:
            raise ValueError(f"Do not support mode {mode}.")

        # no intermedia results for visualization, return empty dict
        return {}
