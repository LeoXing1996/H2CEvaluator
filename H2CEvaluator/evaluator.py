import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils.tqdm import tqdm
from omegaconf import OmegaConf
import imageio
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

METRIC_CONFIG_TYPE = Optional[List[Union[str, dict]]]


def is_list_of(seq: List, target):
    return all([isinstance(ele, target) for ele in seq])


def custom_collate_fn(batch):
    """
    Collate function for pillow and string.
    """
    assert len(batch) == 1, "Only support batch_size as 1."
    assert isinstance(batch[0], dict), "The value of batch must be a dict."
    return batch[0]


class Evaluator:
    """
    Evaluator for Human2Character task.

    Args:
        dataset (Dataset): Dataset to evaluation.
        pipeline_kwargs (Optional[dict]): Default kwargs for pipeline forward.
            Seed for generator may be contained in this dict and be popped to
            create a new generator for evaluation.
        metric_list (METRIC_CONFIG_TYPE): List of config for metrics.

        save_image (bool): Whether save generated results and intermedia results
            from metrics.
        save_keys (Optional[List]): Keys to save. If not passed, only generated
            image (video) will be saved.
        save_as_frames (bool): Save video for individual frames.
        save_path (Optional[str]): Path to save results.
    """

    def __init__(
        self,
        dataset: Dataset,
        pipeline_kwargs: Optional[dict] = None,
        metric_list: METRIC_CONFIG_TYPE = None,
        save_image: bool = False,
        save_keys: Optional[List] = None,
        save_as_frames: bool = False,
        save_path: Optional[str] = None,
    ):
        self.metric_list = self.build_metrics(metric_list, dataset)

        self.dataloader = self.build_dataloader(dataset)

        self.save_image = save_image
        self.save_keys = save_keys
        self.save_as_frames = save_as_frames
        self.save_path = save_path

        self.pipeline_kwargs = deepcopy(pipeline_kwargs)

    def build_metrics(
        self,
        metric_list: METRIC_CONFIG_TYPE = None,
        dataset: Optional[Dataset] = None,
    ):
        if not metric_list:
            return []
        # NOTE: we do not have much metrics,
        # therefore, let's hardcode here (@_@).

        has_self_metric = has_cross_metric = False
        metrics = []
        for metric in metric_list:
            try:
                metric = OmegaConf.to_container(metric)
            except Exception:
                pass

            if isinstance(metric, str):
                metric_type = metric
                metric_kwargs = {}
            elif isinstance(metric, dict):
                metric_type = metric.pop("type")
                metric_kwargs = deepcopy(metric)
            else:
                raise ValueError(
                    f"Do not support {type(metric)} type for metric config."
                )

            if metric_type.upper() == "HYPERIQA":
                from .hyper_iqa import HyperIQA

                metrics.append(HyperIQA(**metric_kwargs))

            elif metric_type.upper() == "ARCFACE":
                from .arcface import ArcFace

                metrics.append(ArcFace(**metric_kwargs))

            elif metric_type.upper() == "SMIRK":
                from .smirk import SMIRK

                metrics.append(SMIRK(**metric_kwargs))

            elif metric_type.upper() == "LPIPS":
                from .lpips import LPIPS

                metrics.append(LPIPS(**metric_kwargs))

                has_self_metric = True

            elif metric_type.upper() == "SSIM":
                from .ssim import SSIM

                metrics.append(SSIM(**metric_kwargs))

                has_self_metric = True

            elif metric_type.upper() == "AESTHETIC":
                from .aesthetic import Aesthetic

                metrics.append(Aesthetic(**metric_kwargs))

            elif metric_type.upper() == "FID":
                from .fid import FID

                cache_dir = metric_kwargs.pop("cache_dir", None)
                cache_path = metric_kwargs.pop("cache_path", None)
                metrics.append(
                    FID(**metric_kwargs).prepare(
                        dataset,
                        cache_dir,
                        cache_path,
                    )
                )

                has_self_metric = True

            elif metric_type.upper() == "FVD":
                from .fvd import FVD

                cache_dir = metric_kwargs.pop("cache_dir", None)
                cache_path = metric_kwargs.pop("cache_path", None)
                metrics.append(
                    FVD(**metric_kwargs).prepare(
                        dataset,
                        cache_dir,
                        cache_path,
                    )
                )

                has_self_metric = True

            else:
                raise ValueError("Do not support metric {}".format(metric))

        if has_self_metric and has_cross_metric:
            print(
                "WARNING: You are using both Self-Reconstruction-Metric and "
                "Cross-ID-Generation-Metric. Please make sure you know what you are doing!"
            )

        return metrics

    def build_dataloader(self, dataset):
        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, range(os.cpu_count()))

        dataloader_kwargs = dict(
            dataset=dataset,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn,
        )
        if PartialState().use_distributed:
            dataloader_kwargs["sampler"] = DistributedSampler(dataset, shuffle=False)

        dataloader = DataLoader(**dataloader_kwargs)

        return dataloader

    def run_eval(self, pipeline):
        """
        Here we assume all metrics use the same types of data pair
        (e.g., face generation / face reconstruction).
        """
        pbar = tqdm(total=len(self.dataloader))

        for idx, data in enumerate(self.dataloader):
            # handle save_name
            save_name = data.pop("save_name", None)
            save_name_list = data.pop("save_name_list", None)

            video = pipeline(
                **data,
                **self.pipeline_kwargs,
            ).videos[0]  # [F, 3, H, W]

            # build a vis dict
            extra_vis_dict = {}

            for metric in self.metric_list:
                fake_vis_dict = metric.feed_one_sample(video, mode="fake")
                real_vis_dict = metric.feed_one_sample(data, mode="real")
                extra_vis_dict.update(fake_vis_dict)
                extra_vis_dict.update(real_vis_dict)

            if self.save_image:
                if self.save_as_frames:
                    if save_name_list is None:
                        if PartialState().use_distributed:
                            save_name_template = f"{idx}_{{}}_rank{PartialState().local_process_index}.png"
                        else:
                            save_name_template = f"{idx}_{{}}.png"
                        save_name_list = [
                            save_name_template.format(j) for j in range(video.shape[0])
                        ]
                    self.save_video_frames(video, data, extra_vis_dict, save_name_list)
                else:
                    if save_name is None:
                        if PartialState().use_distributed:
                            save_name = (
                                f"{idx}_rank{PartialState().local_process_index}.mp4"
                            )
                        else:
                            save_name = f"{idx}.mp4"
                    self.save_video(video, data, extra_vis_dict, save_name)

            pbar.update(1)

        result = {}
        for metric in self.metric_list:
            metric_res = metric.run_evaluation()
            result.update(metric_res)

        return result

    def save_video_frames(
        self,
        video: torch.Tensor,
        data: Dict[str, List[np.ndarray]],
        extra_vis_dict: Dict[str, List[np.ndarray]],
        save_name_list: List[str],
    ):
        """
        Save one video as frames.

        Default
        +-------+---------+-----+-------+
        | Ref   | Driving | Gen | Extra |
        +-------+---------+-----+-------+
        """
        video = video.permute(0, 2, 3, 1)
        extra_keys = list(extra_vis_dict.keys())
        n_extra_items = len(extra_keys)

        for i, image in enumerate(video):
            res_image_pil = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            # Save ref_image, src_fimage and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * (3 + n_extra_items), h), "white")

            ref_image_pil = (
                Image.fromarray(data["reference_image"]).convert("RGB").resize((w, h))
            )
            origin_image_pil = (
                Image.fromarray(data["driving_video"][i]).convert("RGB").resize((w, h))
            )

            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(origin_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            for k in extra_keys:
                canvas.paste(
                    extra_vis_dict[k][i].resize((w, h)),
                    (w * (3 + extra_keys.index(k)), 0),
                )

            sample_name = save_name_list[i]
            img = canvas
            out_file = os.path.join(self.save_path, sample_name)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            img.save(out_file)

    def save_video(
        self,
        video: torch.Tensor,
        data: Dict[str, List[np.ndarray]],
        extra_vis_dict: Dict[str, List[np.ndarray]],
        save_name: str,
    ):
        """
        Save one video.

        Default
        +-------+---------+-----+-------+
        | Ref   | Driving | Gen | Extra |
        +-------+---------+-----+-------+
        """
        video = video.permute(0, 2, 3, 1)
        extra_keys = list(extra_vis_dict.keys())
        n_extra_items = len(extra_keys)

        out_file = os.path.join(self.save_path, save_name)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        writer = imageio.get_writer(out_file, fps=24)

        for i, image in enumerate(video):
            res_image_pil = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            # Save ref_image, src_fimage and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * (3 + n_extra_items), h), "white")

            ref_image_pil = (
                Image.fromarray(data["reference_image"]).convert("RGB").resize((w, h))
            )
            origin_image_pil = (
                Image.fromarray(data["driving_video"][i]).convert("RGB").resize((w, h))
            )

            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(origin_image_pil, (w, 0))
            canvas.paste(res_image_pil, (w * 2, 0))

            for k in extra_keys:
                canvas.paste(
                    extra_vis_dict[k][i].resize((w, h)),
                    (w * (3 + extra_keys.index(k)), 0),
                )

            img = canvas
            writer.append_data(np.array(img))

        writer.close()
