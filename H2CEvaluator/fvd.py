from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .fid import FID

SAMPLE_TYPE = Union[torch.Tensor, dict]


class FVD(FID):
    def __init__(
        self,
        model_path: str = "./work_dirs/eval/i3d_torchscript.pt",
    ):
        self.inception = torch.load(model_path).cuda()

        self.inception_kwargs = {
            "rescale": False,
            "resize": False,
            "return_features": True,
        }

        self._is_prepared = False

        self.real_feat_list = []
        self.fake_feat_list = []

        self.real_mean = self.real_cov = None

    @torch.no_grad()
    def feed_one_sample(self, sample: SAMPLE_TYPE, mode: str):
        """
        Args:
            sample (torch.Tensor | dict): If sample is tensor, sample should be
                [F, C, H, W], order in RGB, range in (0, 1). Otherwise, is dict
                with list of np.ndarray. The length of list is F and all elements
                are un-processed, in [0, 255], [B, H, W, C].
        """
        # NOTE: input sample should be (b, c, f, 224, 224) in [-1, 1]
        # https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/styleganv/fvd.py

        if mode == "fake":
            assert (
                self._is_prepared
            ), "FVD is not prepared. Please check your evaluator."
            fake_sample = (
                sample.cuda().permute(1, 0, 2, 3) / 255
            )  # [0, 1], [c, f, h, w]
            fake_sample = F.interpolate(
                fake_sample,
                (224, 224),
                mode="bilinear",
            )[None]  # [1, c, f, 224, 224]
            fake_sample = fake_sample * 2 - 1
            fake_feat = self.inception(fake_sample, **self.inception_kwargs)
            self.fake_feat_list.append(fake_feat)

            return {}

        elif mode == "real":
            driving_sample = np.stack(sample["driving_video"]) / 255
            driving_sample = (
                torch.from_numpy(driving_sample)
                .permute(3, 0, 1, 2)
                .to(dtype=torch.float32, device="cuda")
            )  # [c, f, h, w]
            driving_sample = F.interpolate(
                driving_sample,
                (224, 224),
                mode="bilinear",
                align_corners=False,
            )[None]  # [1, c, f, 224, 224]
            driving_sample = driving_sample * 2 - 1
            real_feat = self.inception(driving_sample, **self.inception_kwargs)
            self.real_feat_list.append(real_feat)

            return {}

        else:
            raise ValueError(f"Do not support mode {mode}.")
