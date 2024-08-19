import os
import os.path as osp
from argparse import ArgumentParser
from omegaconf import OmegaConf

import cv2
import imageio
from accelerate.utils.tqdm import tqdm
from torch.utils.data import DataLoader

from H2CEvaluator.fid import FID
from H2CEvaluator.fvd import FVD
from H2CEvaluator.metric_utils import get_dataset_meta
from H2CEvaluator.evaluator import custom_collate_fn


class TestDataset(object):
    def __init__(
        self,
        config: OmegaConf,
        reference_image_root: str,
        driving_video_root: str,
        eval_mode: str = "normal",
    ):
        self.width = config.width
        self.height = config.height
        self.max_frames = config.max_frames
        self.interval = config.interval
        self.default_fps = config.default_fps

        samples = []

        reference_image_paths = sorted(os.listdir(reference_image_root))
        driving_video_paths = sorted(os.listdir(driving_video_root))
        if eval_mode == "debug":
            reference_image_paths = reference_image_paths[:3]
            driving_video_paths = driving_video_paths[:1]
        elif eval_mode == "fast":
            reference_image_paths = reference_image_paths[:15]
            driving_video_paths = driving_video_paths[:1]
        else:
            assert (
                eval_mode == "normal"
            ), "Only support 'debug', 'fast' and 'normal' mode."
        for src in reference_image_paths:
            for driving in driving_video_paths:
                if "chenglong" in driving:
                    continue
                if "huangxiaoming" in driving:
                    continue
                if "zhouxingchi" in driving:
                    continue
                reference_image_path = os.path.join(reference_image_root, src)
                driving_video_path = os.path.join(driving_video_root, driving)
                samples.append((reference_image_path, driving_video_path))

        self.samples = samples

    def conditioned_resize(self, img):
        if (
            self.width > 0
            and self.height > 0
            and img.shape[:2] != (self.height, self.width)
        ):
            img = cv2.resize(img, (self.width, self.height))
        return img

    def __getitem__(self, index):
        """
        Returns:
            dict: contains the reference image, driving video and other meta information.

            - reference_image (numpy.ndarray): image read from imageio.read.
            - driving_video (List[numpy.ndarray]): video frames read from imageio.read.
            - fps (float): fps of driving video.
            - raw_reference_width (int): width of the raw reference image.
            - raw_reference_height (int): height of the raw referece image.
            - raw_driving_width (int): width of the raw driving video.
            - raw_driving_height (int): height of the raw driving video.
            - target_width (int): width of the returned driving_video.
            - target_height (int): height of the returned driving_video.
            - save_name (str): filename to be saved.
            - save_name_list (List[str]): filenames of all frames to be saved.
        """

        reference_image_path, driving_video_path = self.samples[index]
        reference_image = imageio.v3.imread(reference_image_path)
        reference_image_width = reference_image.shape[1]
        reference_image_height = reference_image.shape[0]
        reference_image = self.conditioned_resize(reference_image)

        if os.path.isfile(driving_video_path):
            reader = imageio.get_reader(driving_video_path)
            fps = reader.get_meta_data()["fps"]
            driving_width = reader.get_meta_data()["size"][0]
            driving_height = reader.get_meta_data()["size"][1]
            driving_video = []
            reference_basename = os.path.basename(reference_image_path).split(".")[0]
            driving_basename = os.path.basename(driving_video_path).split(".")[0]
            save_basename = reference_basename + "_" + driving_basename
            save_name = save_basename + ".mp4"
            save_name_list = []
            counter = 0
            for im in reader:
                if self.max_frames > 0 and len(driving_video) >= self.max_frames:
                    break
                if counter % self.interval != 0:
                    counter += 1
                    continue
                else:
                    save_name_list.append(
                        os.path.join(save_name, "%08d.jpg" % len(driving_video))
                    )
                    im = self.conditioned_resize(im)
                    driving_video.append(im)
                    counter += 1
            reader.close()
        else:
            driving_video = []
            fps = self.default_fps
            for frame_name in sorted(os.listdir(driving_video_path))[:: self.interval]:
                if self.max_frames > 0 and len(driving_video) >= self.max_frames:
                    break
                save_name_list.append(
                    os.path.join(save_name, "%08d.jpg" % len(driving_video))
                )
                im = imageio.imread(
                    os.path.join(driving_video_path, frame_name), pilmode="RGB"
                )
                driving_width = im.shape[1]
                driving_height = im.shape[0]
                im = self.conditioned_resize(im)
                driving_video.append(im)
            if len(driving_video) < 10:
                print(
                    f"Warning: {driving_video_path} has {len(os.listdir(driving_video_path))} frames."
                )
        width = driving_video[0].shape[1]
        height = driving_video[0].shape[0]

        datapoint = dict(
            reference_image=reference_image,
            driving_video=driving_video,
            fps=fps,
            raw_reference_width=reference_image_width,
            raw_reference_height=reference_image_height,
            raw_driving_width=driving_width,
            raw_driving_height=driving_height,
            target_width=width,
            target_height=height,
            save_name=save_name,
            save_name_list=save_name_list,
        )

        return datapoint

    def __len__(self):
        return len(self.samples)


def main(args):
    cache_path = args.cache_path
    os.makedirs(cache_path, exist_ok=True)

    config = OmegaConf.load(args.config)

    dataset = TestDataset(
        config,
        args.reference_image_root,
        args.driving_video_root,
    )

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=16,
        worker_init_fn=worker_init_fn,
    )

    metric_list = []
    metric_cache_list = []
    if args.fid:
        fid = FID(args.fid_model_path)
        fid_cache_path = fid.get_real_feat_cache_path(dataset, cache_path)

        if osp.exists(fid_cache_path):
            if args.force:
                print("Cache for FID exists, but force is set. Overwriting...")
                metric_list.append(fid)
                metric_cache_list.append(fid_cache_path)
            else:
                print(f'Cache for FID exists ("{fid_cache_path}"). Skipping...')
        else:
            metric_list.append(fid)
            metric_cache_list.append(fid_cache_path)

    if args.fvd:
        fvd = FVD(args.fvd_model_path)
        fvd_cache_path = fvd.get_real_feat_cache_path(dataset, cache_path)

        if osp.exists(fvd_cache_path):
            if args.force:
                print("Cache for FVD exists, but force is set. Overwriting...")
                metric_list.append(fvd)
                metric_cache_list.append(fvd_cache_path)
            else:
                print(f'Cache for FVD exists ("{fvd_cache_path}"). Skipping...')
        else:
            metric_list.append(fvd)
            metric_cache_list.append(fvd_cache_path)

    if len(metric_list) == 0:
        print("No metric to compute. Exiting...")

    pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        for metric in metric_list:
            metric.feed_one_sample(data, mode="real")

        pbar.update(1)
    pbar.close()

    dataset_meta, _ = get_dataset_meta(dataset)
    for metric, save_path in zip(metric_list, metric_cache_list):
        metric.dump_real_feature_cache(
            save_path,
            dataset_meta=dataset_meta,
        )
        print(f'Dumped cache for {metric.__class__.__name__} ("{save_path}").')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--reference-image-root",
        type=str,
        help="Path to reference image.",
    )
    parser.add_argument(
        "--driving-video-root",
        type=str,
        help="Path to driving video.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for extract inception feature.",
    )

    parser.add_argument(
        "--fid",
        action="store_true",
        help="whether preprocess feature for FID",
    )
    parser.add_argument(
        "--fvd",
        action="store_true",
        help="whether preprocess feature for FVD",
    )
    parser.add_argument(
        "--fid-model-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--fvd-model-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=".cache",
        help="path to save cache feature",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether overwriote cache when exists.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["debug", "fast", "normal"],
        default="normal",
        help="Evaluation mode. If not normal, only use part of the data to evaluate the script.",
    )
    args = parser.parse_args()
    main(args)
