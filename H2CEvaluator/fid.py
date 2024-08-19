import os.path as osp
from typing import List, Optional

import numpy as np
import torch
from accelerate import PartialState
from scipy import linalg

from .dist_utils import gather_all_tensors
from .metric_utils import get_dataset_meta


class FID:
    def __init__(self, model_path="./models/inception-2015-12-05.pt"):
        self.inception = torch.load(model_path).cuda()

        self.inception_kwargs = {"return_features": True}

        self._is_prepared = False

        self.real_feat_list = []
        self.fake_feat_list = []

        self.real_mean = self.real_cov = None

    def prepare(self, dataset, feat_cache_path):
        """Prepare metric"""
        # solve cache path
        self.real_feat_path = self.get_real_feat_cache_path(dataset, feat_cache_path)

        # attempt to load cache feature
        if osp.exists(self.real_feat_path):
            real_feat = torch.load(self.real_feat_path)
            real_mean, real_cov = real_feat["mean"], real_feat["cov"]
        else:
            raise FileExistsError(
                f'Real feature cache for FID not found ("{feat_cache_path}"). '
                f"Please run scripts/preprocess.py to generate the cache."
            )
        self.real_mean, self.real_cov = real_mean, real_cov

        self._is_prepared = True

    @staticmethod
    def get_real_feat_cache_path(dataset, feat_cache_path):
        _, md5 = get_dataset_meta(dataset)
        real_feat_path = osp.join(feat_cache_path, f"fid_real_cache_{md5}.pt")

        return real_feat_path

    def dump_real_feature_cache(
        self,
        cache_path: str,
        dataset_meta: Optional[dict] = None,
    ):
        real_feat_mean, real_feat_cov = self._get_mean_cov(self.real_feat_list)

        torch.save(
            {
                "mean": real_feat_mean,
                "cov": real_feat_cov,
                "dataset_meta": dataset_meta,
            },
            cache_path,
        )

    @staticmethod
    def _get_mean_cov(feat_list: List[torch.Tensor]):
        if len(feat_list) > 1:
            feat = torch.cat(feat_list)
        else:
            feat = feat_list[0]

        if PartialState().use_distributed:
            feat = gather_all_tensors(feat)
            feat = torch.cat(feat)

        feat_np = feat.cpu().numpy()
        feat_mean = np.mean(feat_np, 0)
        feat_cov_np = np.cov(feat_np, rowvar=False)

        return feat_mean, feat_cov_np

    def run_evaluation(self):
        # all gather
        fake_mean, fake_cov = self._get_mean_cov(self.fake_feat_list)

        self.fake_feat_list.clear()
        self.real_feat_list.clear()

        fid, mean, trace = self._calc_fid(
            fake_mean,
            fake_cov,
            self.real_mean,
            self.real_cov,
        )

        return {"fid": fid, "mean": mean, "trace": trace}

    @torch.no_grad()
    def feed_one_sample(self, sample: torch.Tensor, mode: str):
        """
        Feed one sample, forward inception network, and save to the feature list.

        Args:
            sample (torch.Tensor | dict): If sample is tensor, sample should be
                [F, C, H, W], order in RGB, range in (0, 1). Otherwise, is dict
                with list of np.ndarray. The length of list is F and all elements
                are un-processed, in [0, 255], [B, H, W, C].
        """
        # NOTE: input sample should be (b, c, h, w), in [0, 255] and **BGR**

        if mode == "fake":
            assert (
                self._is_prepared
            ), "FID is not prepared. Please check your evaluator."
            fake_sample = sample / 255  # [f, c, h, w]
            fake_sample = fake_sample[:, [2, 1, 0], ...]  # [RGB] -> [BGR]
            fake_feat = self.inception(fake_sample)
            self.fake_feat.append(fake_feat)
        elif mode == "real":
            driving_sample = np.stack(sample["driving_video"])  # [f, h, w, c]
            driving_sample = (
                torch.from_numpy(driving_sample)
                .to(dtype=torch.float32, device="cuda")
                .permute(0, 3, 1, 2)
            )  # [f, c, h, w]
            driving_sample = driving_sample[:, [2, 1, 0], ...]  # [RGB] -> [BGR]
            real_feat = self.inception(driving_sample, **self.inception_kwargs)
            self.real_feat_list.append(real_feat)
        else:
            raise ValueError(f"Do not support mode {mode}.")

    def _calc_fid(self, fake_mean, fake_cov, real_mean, real_cov, eps=1e-6):
        cov_sqrt, _ = linalg.sqrtm(fake_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print("product of cov matrices is singular")
            offset = np.eye(fake_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm((fake_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f"Imaginary component {m}")

            cov_sqrt = cov_sqrt.real

        mean_diff = fake_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(fake_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return float(fid), float(mean_norm), float(trace)
