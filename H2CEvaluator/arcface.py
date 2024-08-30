import os.path as osp

import torch
import torch.nn.functional as F

from .arcface_torch.backbones import get_model
from .dist_utils import gather_tensor_list
from .metric_utils import DEFAULT_CACHE_DIR, FileHashItem, MetricModelItems, SAMPLE_TYPE


class ArcFace:
    metric_items = MetricModelItems(
        file_list=[
            FileHashItem(
                "glint360k_cosface_r100_fp16_0.1.pth",
                sha256="5f631718e783448b41631e15073bdc622eaeef56509bbad4e5085f23bd32db83",
            )
        ],
        remote_subfolder="arcface",
    )

    def __init__(
        self,
        model_dir: str = osp.join(DEFAULT_CACHE_DIR, "arcface"),
        model_type="r100",
    ):
        self.metric_items.prepare_model(model_dir)
        model_path = f"{model_dir}/glint360k_cosface_r100_fp16_0.1.pth"
        self.face_model = get_model(model_type, fp16=True)
        self.face_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.face_model = self.face_model.cuda()
        self.face_model.train(False)

        self.fake_feat = []
        self.arcface_dist_list = []

    def run_evaluation(self):
        arcface_dist = gather_tensor_list(self.arcface_dist_list)
        arcface_dist = torch.mean(arcface_dist).item()

        self.arcface_dist_list.clear()

        result_dict = {"arcface_score": arcface_dist}
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

            assert len(self.fake_feat) == 1, (
                "When call feed_one_sample with mode `real`, "
                "ArcFace.fake_list should only contain one element. "
                "Please check your code!"
            )

            arcface_dist = F.cosine_similarity(
                self.fake_feat.pop(), F.normalize(real_feat), dim=1
            ).mean()
            self.arcface_dist_list.append(arcface_dist[None])

        else:
            raise ValueError(f"Do not support mode {mode}.")

        # no intermedia results for visualization, return empty dict
        return {}
