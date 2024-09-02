import os
import os.path as osp
from argparse import ArgumentParser
from omegaconf import OmegaConf

import cv2
import imageio
from accelerate.utils.tqdm import tqdm
from torch.utils.data import DataLoader

from ..fid import FID
from ..fvd import FVD
from ..metric_utils import get_dataset_meta, DEFAULT_CACHE_DIR
from ..evaluator import custom_collate_fn


def get_args():
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
        "--fid-model-dir",
        type=str,
        default=osp.join(DEFAULT_CACHE_DIR, "fid"),
        help="Path to FID model directory.",
    )
    parser.add_argument(
        "--fvd-model-dir",
        type=str,
        default=osp.join(DEFAULT_CACHE_DIR, "fvd"),
        help="Path to FVD model directory.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="path to save cache feature.",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers for dataloader.",
    )
    args = parser.parse_args()
    return args


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
        self.min_frames = config.min_frames
        assert self.min_frames >= 10, "The number of frames must be no smaller than 10."
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
            nframes = len(reader)
            adaptive_interval = min(self.interval, nframes // self.min_frames)
            assert (
                adaptive_interval > 0
            ), f"Unsupported interval {adaptive_interval} (must be greater than 0)"
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
                if counter % adaptive_interval != 0:
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
            filenames = sorted(os.listdir(driving_video_path))
            nframes = len(filenames)
            adaptive_interval = min(self.interval, nframes // self.min_frames)
            assert (
                adaptive_interval > 0
            ), f"Unsupported interval {adaptive_interval} (must be greater than 0)"
            for frame_name in filenames[::adaptive_interval]:
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


def main():
    args = get_args()
    cache_path = args.cache_path
    os.makedirs(cache_path, exist_ok=True)

    if args.config is None:
        config = osp.join(osp.dirname(__file__), "default_config.yaml")
    else:
        config = args.config

    config = OmegaConf.load(config)

    if args.reference_image_root is not None:
        reference_image_root = args.reference_image_root
    else:
        reference_image_root = config.reference_image_root

    if args.driving_video_root is not None:
        driving_video_root = args.driving_video_root
    else:
        driving_video_root = config.driving_video_root

    dataset = TestDataset(
        config,
        reference_image_root,
        driving_video_root,
    )

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    metric_list = []
    metric_cache_list = []
    if args.fid:
        fid = FID(args.fid_model_dir)
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
        fvd = FVD(args.fvd_model_dir)
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
    main()
