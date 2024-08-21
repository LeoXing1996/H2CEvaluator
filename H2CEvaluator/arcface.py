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
    def feed_one_sample(self, sample: SAMPLE_TYPE, mode: str):
        """Feed one sample.

        Args:
            sample (torch.Tensor | dict): If sample is tensor, sample should be
                [F, C, H, W], order in RGB, range in (0, 1). Otherwise, is dict
                with list of np.ndarray. The length of list is F and all elements
                are un-processed, in [0, 255], [B, H, W, C].
        """

        if mode == "fake":
            fake_sample = sample * 2 - 1  # [-1, 1]
            fake_feat = self.face_model(
                F.interpolate(fake_sample, size=(112, 112)).cuda().half()
            )
            self.fake_feat.append(F.normalize(fake_feat))

        elif mode == "real":
            ref_samples = torch.from_numpy(sample["reference_image"])  # [H, W, 3]
            ref_samples = ref_samples[None].permute(0, 3, 1, 2)  # [1, 3, H, W]
            ref_samples = ref_samples / 127.5 - 1  # [-1, 1]
            ref_samples = F.interpolate(ref_samples, size=(112, 112)).repeat(
                self.fake_feat[-1].shape[0], 1, 1, 1
            )

            real_feat = self.face_model(ref_samples.cuda().half())
            self.real_feat.append(real_feat / torch.norm(real_feat, dim=1)[:, None])

        else:
            raise ValueError(f"Do not support mode {mode}.")

        # no intermedia results for visualization, return empty dict
        return {}
