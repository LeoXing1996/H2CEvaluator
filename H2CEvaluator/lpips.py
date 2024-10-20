import numpy as np
import torch
import torch.nn.functional as F

from .dist_utils import gather_tensor_list
from .metric_utils import SAMPLE_TYPE

try:
    import lpips
except ImportError:
    print("lpips in not installed, please install it via `pip install lpips`")


class LPIPS:
    def __init__(self, model_type: str = "alex"):
        self.lpips = lpips.LPIPS(net=model_type).cuda()

        self.fake_list = []
        self.score_list = []

    def run_evaluation(self):
        score_list = gather_tensor_list(self.score_list)
        score = torch.mean(score_list).item()
        self.score_list.clear()
        result_dict = {"lpips": score}
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
            fake_sample = sample * 2 - 1  # [-1., 1]
            self.fake_list.append(fake_sample)

            # no intermedia results for visualization, return empty dict
            return {}, {}

        elif mode == "real":
            src_samples = sample["driving_video"]
            src_samples = [
                torch.from_numpy(np.array(src_sample)).float()
                for src_sample in src_samples
            ]
            src_samples = torch.stack(src_samples).permute(0, 3, 1, 2).float()
            src_samples = (src_samples / 255 - 0.5) / 0.5  # [-1, 1]

            assert len(self.fake_list) == 1, (
                "When call feed_one_sample with mode `real`, "
                "LPIPS.fake_list should only contain one element. Please your code!"
            )

            fake_samples = self.fake_list.pop().cuda()
            if fake_samples.shape[-2:] != src_samples.shape[-2]:
                fake_samples = F.interpolate(fake_samples, src_samples.shape[-2:])

            score = self.lpips(src_samples.cuda(), fake_samples)
            self.score_list.append(score)

            # no intermedia results for visualization, return empty dict
            return {}, {"lpips": score.item()}

        else:
            raise ValueError(f"Do not support mode {mode}.")
