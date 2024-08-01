from typing import Union

import torch
import torch.nn.functional as F
from accelerate import PartialState

from .arcface_torch.backbones import get_model
from .dist_utils import gather_all_tensors

SAMPLE_TYPE = Union[torch.Tensor, dict]


class ArcFace:
    def __init__(
        self,
        model_type="r100",
        model_path="./models/glint360k_cosface_r100_fp16_0.1",
    ):
        self.face_model = get_model(model_type, fp16=True)
        self.face_model.load_state_dict(torch.load(model_path))
        self.face_model.cuda()
        self.face_model.train(False)

        self.real_feat = []
        self.fake_feat = []

    def prepare(self, *args, **kwargs):
        """Do not need prepare. Do nothing."""
        return

    def run_evaluation(self):
        if len(self.real_feat) > 1:
            fake_feat = torch.cat(self.fake_feat)
            real_feat = torch.cat(self.real_feat)
        else:
            fake_feat = self.fake_feat[0]
            real_feat = self.real_feat[0]

        if PartialState().use_distributed:
            real_feat = gather_all_tensors(real_feat)
            fake_feat = gather_all_tensors(fake_feat)

            fake_feat = torch.cat(fake_feat)
            real_feat = torch.cat(real_feat)

        cosine_dist = F.cosine_similarity(real_feat, fake_feat, dim=1).mean()

        self.real_feat.clear()
        self.fake_feat.clear()

        result_dict = {"arcface_score": cosine_dist.item()}
        return result_dict

    @torch.no_grad()
    def feed_one_sample(self, sample: Union[torch.Tensor, dict], mode: str):
        """Feed one sample.

        Args:
            sample (torch.Tensor): [F, C, H, W], order in RGB, range in (0, 1).
        """

        if mode == "fake":
            fake_sample = sample * 2 - 1  # [-1, 1]
            fake_feat = self.face_model(
                F.interpolate(fake_sample, size=(112, 112)).cuda().half()
            )
            self.fake_feat.append(F.normalize(fake_feat))

        elif mode == "real":
            import numpy as np

            ref_samples = (
                sample["ref_image"] * self.fake_feat[-1].shape[0]
            )  # list of pillow image
            ref_samples = [s.resize((112, 112)) for s in ref_samples]
            ref_samples = [torch.from_numpy(np.array(s)) for s in ref_samples]
            ref_samples = torch.stack(ref_samples).permute(0, 3, 1, 2).float()
            ref_samples = (ref_samples / 255 - 0.5) / 0.5  #  [-1, 1]

            real_feat = self.face_model(ref_samples.cuda().half())
            self.real_feat.append(real_feat / torch.norm(real_feat, dim=1)[:, None])

        else:
            raise ValueError(f"Do not support mode {mode}.")

        # no intermedia results for visualization, return empty dict
        return {}
