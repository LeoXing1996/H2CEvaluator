import hashlib
import os.path as osp

import numpy as np
import torch
from scipy import linalg
from .dist_utils import gather_all_tensors


class FVD:
    def __init__(self, model_path="./work_dirs/eval/i3d_torchscript.pt"):
        self.inception = torch.load(model_path).cuda()

        self.inception_kwargs = {
            "rescale": True,
            "resize": True,
            "return_features": True,
        }

        self._is_prepared = False

        self.real_feat_list = []
        self.fake_feat_list = []

    def prepare(self, dataset, feat_cache_path):
        """Prepare metric"""
        # solve cache path
        self.real_feat_path = self.get_real_feat_cache_path(dataset, feat_cache_path)

        # attempt to load cache feature
        if osp.exists(self.real_feat_path):
            real_feat = torch.load(self.real_feat_path)
            real_mean, real_cov = real_feat["mean"], real_feat["cov"]
        else:
            real_mean = real_cov = None
        self.real_mean, self.real_cov = real_mean, real_cov

        self._is_prepared = True

    @staticmethod
    def get_real_feat_cache_path(dataset, feat_cache_path):
        dataset_identity = [
            "sample_rate",
            "n_sample_frame",
            "width",
            "height",
            "img_size",
            "cond_size",
            "img_scale",
            "img_ratio",
            "rtmpose_drop_rate",
            "patch_drop_rate",
            "patch_aug_prob",
            "data_meta_paths",
        ]
        dataset_kwargs = {k: dataset.__dict__.get(k, None) for k in dataset_identity}
        md5 = hashlib.md5(str(dataset_kwargs).encode()).hexdigest()
        real_feat_path = osp.join(feat_cache_path, f"fvd_real_cache_{md5}.pt")

        return real_feat_path

    def run_evaluation(self):
        # all gather
        fake_feat = gather_all_tensors(self.fake_feat_list)

        fake_feat_mean = np.mean(fake_feat, 0)
        fake_feat_cov = np.cov(fake_feat, rowvar=False)

        if self.real_mean is None:
            real_feat = gather_all_tensors(self.real_feat_list)
            real_feat_mean = np.mean(real_feat, 0)
            real_feat_cov = np.cov(real_feat, rowvar=False)
            self.real_mean, self.real_cov = real_feat_mean, real_feat_cov
            torch.save(
                {"mean": self.real_mean, "cov": self.real_cov},
                self.real_feat_path,
            )

        self.fake_feat_list.clear()
        self.real_feat_list.clear()

        fid, mean, trace = self._calc_fid(
            fake_feat_mean,
            fake_feat_cov,
            self.real_mean,
            self.real_cov,
        )

        return {"fvd": fid, "mean": mean, "trace": trace}

    @torch.no_grad()
    def feed_one_sample(self, sample: torch.Tensor, mode: str):
        """
        Feed one sample, forward inception network, and save to the feature list.
        """
        # TODO: we need some pre-process function, check this later.
        if mode == "fake":
            fake_feat = self.inception(sample)
            self.fake_feat.append(fake_feat)
        elif mode == "real" and self.real_mean is None:
            real_feat = self.inception(sample)
            self.real_feat.append(real_feat)
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
