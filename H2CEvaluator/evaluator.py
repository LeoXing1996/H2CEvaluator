import os
from copy import deepcopy
from typing import Dict, List, Optional, Union
import simplejson as json

import cv2
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
import logging

METRIC_CONFIG_TYPE = Optional[List[Union[str, dict]]]

logger = logging.getLogger(__file__)


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
    """

    def __init__(
        self,
        dataset: Dataset,
        pipeline_kwargs: Optional[dict] = None,
        metric_list: METRIC_CONFIG_TYPE = None,
    ):
        self.metric_list = self.build_metrics(metric_list, dataset)

        self.dataloader = self.build_dataloader(dataset)

        if pipeline_kwargs is None:
            pipeline_kwargs = {}

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

                has_cross_metric = True

            elif metric_type.upper() == "SMIRK":
                from .smirk import SMIRK

                metrics.append(SMIRK(**metric_kwargs))

                has_cross_metric = True

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

            elif metric_type.upper() == "POINTTRACKING":
                from .point_tracking import PointTracking

                metrics.append(PointTracking(**metric_kwargs))

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

    @staticmethod
    def get_save_name(
        vis_save_path: str,
        save_as_frames: bool,
        idx: int,
        save_name: Optional[str] = None,
    ):
        """Function to get save name for current sample."""

        if save_name is None:
            if PartialState().use_distributed:
                save_name = f"{idx}_rank{PartialState().local_process_index}.mp4"
            else:
                save_name = f"{idx}.mp4"
        return os.path.join(vis_save_path, save_name)

    @staticmethod
    def get_save_name_list(
        vis_save_path: str,
        save_as_frames: bool,
        idx: int,
        save_name_list: Optional[List[str]] = None,
        n_frames: Optional[int] = None,
    ):
        """Function to get save name list for current sample."""

        if save_name_list is None:
            if PartialState().use_distributed:
                save_name_template = (
                    f"{idx}_{{}}_rank{PartialState().local_process_index}.png"
                )
            else:
                save_name_template = f"{idx}_{{}}.png"
            save_name_list = [save_name_template.format(j) for j in range(n_frames)]
        return [os.path.join(vis_save_path, save_name) for save_name in save_name_list]

    @staticmethod
    def resume_from_saved_samples(
        save_as_frames: bool,
        save_name: Optional[str] = None,
        save_name_list: Optional[List[str]] = None,
        n_frames: Optional[int] = None,
    ):
        resumed_video = None
        if save_as_frames:
            # TODO: do not support now
            pass
        else:
            if os.path.exists(save_name):
                cap = cv2.VideoCapture(save_name)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                cap.release()
                resumed_video = torch.from_numpy(np.array(frames))  # [F, H, W, C]
                resumed_video = (
                    resumed_video.permute(0, 3, 1, 2) / 255.0
                )  # [F, C, H, W]

        if n_frames is not None and resumed_video is not None:
            assert resumed_video.shape[0] == n_frames, ()

        if resumed_video is None:
            print(f"Did not find {save_name}")

        return resumed_video.contiguous() if resumed_video is not None else None

    def run_eval(
        self,
        pipeline,
        save_path: str,
        no_eval: bool = False,
        no_vis: bool = False,
        eval_samples: int = -1,
        vis_samples: int = -1,
        save_as_frames: bool = False,
    ):
        """
        Here we assume all metrics use the same types of data pair
        (e.g., face generation / face reconstruction).

        Args:
            pipeline: Inference pipeline to generate results.
            save_path (str): Path to save results. The evaluation and visualization
                results will be saved to this folder.
            no_eval (bool): If true, will not run evaluation.
            no_vis (bool): If true, will not save visualization results.
            eval_samples (int): Number of samples to evaluate. If -1, will evaluate all samples.
            vis_samples (int): Number of samples to visualize. If -1, will visualize all samples.
            save_meta_info (bool): Whether save meta info. If true, will save a meta info
            save_as_frames (bool): Whether save results as individual frames.
        """
        pbar_length = len(self.dataloader)
        if max(vis_samples, eval_samples) > 0:
            pbar_length = min(max(vis_samples, eval_samples), pbar_length)
        pbar = tqdm(total=pbar_length)

        if no_eval and no_vis:
            logger.warning("Both no_eval and no_vis are True. Nothing to do.")
            return {}

        meta_info_list = []
        for idx, data in enumerate(self.dataloader):
            should_eval = (eval_samples == -1 or idx < eval_samples) and not no_eval
            should_vis = (vis_samples == -1 or idx < vis_samples) and not no_vis

            if not should_eval and not should_vis:
                break

            # handle save_name
            save_name = data.pop("save_name", None)
            save_name_list = data.pop("save_name_list", None)
            reference_name = data.pop("reference_filename", "null")
            driving_name = data.pop("driving_filename", "null")

            # handle resume name
            if save_as_frames:
                save_name_list = self.get_save_name_list(
                    save_path,
                    save_as_frames,
                    idx,
                    save_name_list,
                    len(data["driving_video"]),
                )
            else:
                save_name = self.get_save_name(
                    save_path,
                    save_as_frames,
                    idx,
                    save_name,
                )
            resumed_video = self.resume_from_saved_samples(
                save_as_frames,
                save_name,
                save_name_list,
                len(data["driving_video"]),
            )

            meta_info = {
                "reference_name": reference_name,
                "driving_name": driving_name,
                "used_for_eval": should_eval,
            }

            meta_info_list.append(meta_info)

            if resumed_video is not None:
                video = resumed_video
                cond = None
                meta_info["success"] = True
                meta_info["is_resumed"] = True

            else:
                try:
                    output = pipeline(
                        **data,
                        **self.pipeline_kwargs,
                    )
                    video = output.videos[0]  # [F, 3, H, W]
                    cond = output.conditions[0]  # [F, 3, H, W]
                    meta_info["success"] = True

                except Exception as e:
                    meta_info["success"] = False
                    meta_info["exception"] = str(e)
                    continue

            # build a vis dict
            extra_vis_dict = {}

            if should_eval:
                for metric in self.metric_list:
                    fake_vis_dict = metric.feed_one_sample(video, mode="fake")
                    real_vis_dict = metric.feed_one_sample(data, mode="real")
                    extra_vis_dict.update(fake_vis_dict)
                    extra_vis_dict.update(real_vis_dict)

            if should_vis:
                if save_as_frames:
                    self.save_video_frames(
                        video, cond, data, extra_vis_dict, save_name_list
                    )
                    meta_info["save_name_list"] = save_name_list
                else:
                    self.save_video(
                        video,
                        cond,
                        data,
                        extra_vis_dict,
                        save_name,
                    )
                    meta_info["save_name"] = save_name

            pbar.update(1)

        result = {}
        for metric in self.metric_list:
            metric_res = metric.run_evaluation()
            result.update(metric_res)

        # gather sync meta-info across device
        if PartialState().use_distributed:
            import torch.distributed as dist

            gathered_info_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_info_list, meta_info_list)
            meta_info_list = []
            for info_list in gathered_info_list:
                meta_info_list.extend(info_list)
        if PartialState().is_main_process:
            result_path = os.path.join(save_path, "result.json")
            with open(result_path, "w") as file:
                json.dump(
                    {"result": result, "meta_info": meta_info_list},
                    file,
                    indent=4,
                )

        return result

    def save_video_frames(
        self,
        video: torch.Tensor,
        cond: torch.Tensor,
        data: Dict[str, List[np.ndarray]],
        extra_vis_dict: Dict[str, List[np.ndarray]],
        save_name_list: List[str],
    ):
        """
        Save one video as frames.

        Default
        +-------+---------+------+-----+-------+
        | Ref   | Driving | Cond | Gen | Extra |
        +-------+---------+------+-----+-------+
        """
        video = video.permute(0, 2, 3, 1)
        if cond is not None:
            cond = cond.permute(0, 2, 3, 1)
        else:
            cond = torch.zeros_like(video)
        extra_keys = list(extra_vis_dict.keys())
        n_extra_items = len(extra_keys)

        for i, (image, cond_image) in enumerate(zip(video, cond)):
            res_image_pil = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            cond_image_pil = Image.fromarray(
                (cond_image.numpy() * 255).astype(np.uint8)
            )
            # Save ref_image, src_fimage and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * (4 + n_extra_items), h), "white")

            ref_image_pil = (
                Image.fromarray(data["reference_image"]).convert("RGB").resize((w, h))
            )
            origin_image_pil = (
                Image.fromarray(data["driving_video"][i]).convert("RGB").resize((w, h))
            )

            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(origin_image_pil, (w, 0))
            canvas.paste(cond_image_pil.resize((w, h)), (w * 2, 0))
            canvas.paste(res_image_pil, (w * 3, 0))

            for k in extra_keys:
                canvas.paste(
                    Image.fromarray(extra_vis_dict[k][i]).resize((w, h)),
                    (w * (4 + extra_keys.index(k)), 0),
                )

            sample_name = save_name_list[i]
            sample_suffix = sample_name.split(".")[-1]
            sample_combine_name = sample_name.replace(
                f".{sample_suffix}", f"_comb.{sample_suffix}"
            )
            os.makedirs(os.path.dirname(sample_name, exist_ok=True))

            res_image_pil.save(sample_name)
            canvas.save(sample_combine_name)

    def save_video(
        self,
        video: torch.Tensor,
        cond: torch.Tensor,
        data: Dict[str, List[np.ndarray]],
        extra_vis_dict: Dict[str, List[np.ndarray]],
        save_name: str,
    ):
        """
        Save one video.

        Default
        +-------+---------+------+-----+-------+
        | Ref   | Driving | Cond | Gen | Extra |
        +-------+---------+------+-----+-------+
        """
        video = video.permute(0, 2, 3, 1)
        if cond is not None:
            cond = cond.permute(0, 2, 3, 1)
        else:
            cond = torch.zeros_like(video)
        extra_keys = list(extra_vis_dict.keys())
        n_extra_items = len(extra_keys)

        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        writer = imageio.get_writer(save_name, fps=24)

        save_suffix = save_name.split(".")[-1]
        combine_name = save_name.replace(f".{save_suffix}", f"_comb.{save_suffix}")
        writer_comb = imageio.get_writer(combine_name)

        for i, (image, cond_image) in enumerate(zip(video, cond)):
            res_image_pil = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            cond_image_pil = Image.fromarray(
                (cond_image.numpy() * 255).astype(np.uint8)
            )
            # Save ref_image, src_fimage and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * (4 + n_extra_items), h), "white")

            ref_image_pil = (
                Image.fromarray(data["reference_image"]).convert("RGB").resize((w, h))
            )
            dri_image_pil = (
                Image.fromarray(data["driving_video"][i]).convert("RGB").resize((w, h))
            )

            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(dri_image_pil, (w, 0))
            canvas.paste(cond_image_pil.resize((w, h)), (w * 2, 0))
            canvas.paste(res_image_pil, (w * 3, 0))

            for k in extra_keys:
                canvas.paste(
                    Image.fromarray(extra_vis_dict[k][i]).resize((w, h)),
                    (w * (4 + extra_keys.index(k)), 0),
                )

            img = canvas
            writer_comb.append_data(np.array(img))
            writer.append_data(np.array(res_image_pil))

        writer.close()
