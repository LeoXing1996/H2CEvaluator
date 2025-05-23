from typing import Tuple, Union

import os.path as osp
import numpy as np
import torch

from .dist_utils import gather_tensor_list
from .metric_utils import FileHashItem, MetricModelItems, DEFAULT_CACHE_DIR
from .point_tracking_utils import (
    Visualizer,
    get_local_patch_tracks,
    init_model,
    process_one_image,
)

DETECTOR_CONFIG = osp.abspath(
    osp.join(__file__, "..", "configs/yolox-s_8xb8-300e_coco-face.py")
)
POSE_CONFIG = osp.abspath(
    osp.join(__file__, "..", "configs/rtmpose-m_8xb256-120e_face6-256x256.py")
)


SAMPLE_TYPE = Union[torch.Tensor, dict]


class PointTracking:
    """
    Metric for face region point tracking.
    """

    metric_items = MetricModelItems(
        file_list=[
            FileHashItem(
                "yolo-x_8xb8-300e_coco-face_13274d7c.pth",
                sha256="13274d7c0cf00381b8cd1186f0aea849582f83c53f8a9c5cdf0227fc0042d283",
            ),
            FileHashItem(
                "rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth",
                sha256="72a37400a40946db314ec8287e2e77cddc0b1f4a9e1bced3af633355cb72bd03",
            ),
        ],
        remote_subfolder="rtmpose",
    )

    def __init__(
        self,
        model_dir: str = osp.join(DEFAULT_CACHE_DIR, "rtmpose"),
        eye_grid_size: int = 20,
        mouse_grid_size: int = 10,
        enable_vis: bool = True,
    ):
        self.metric_items.prepare_model(model_dir)

        self.eye_grid_size = eye_grid_size
        self.mouse_grid_size = mouse_grid_size
        self.enable_vis = enable_vis
        if self.enable_vis:
            self.visualier = Visualizer(linewidth=1, show_first_frame=0)

        self.tracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker2_online",
        ).cuda()

        detector_checkpoint = f"{model_dir}/yolo-x_8xb8-300e_coco-face_13274d7c.pth"
        pose_checkpoint = f"{model_dir}/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"

        self.detector, self.pose_estimator = init_model(
            DETECTOR_CONFIG,
            POSE_CONFIG,
            detector_checkpoint,
            pose_checkpoint,
        )
        self.detector = self.detector.cuda()
        self.pose_estimator = self.pose_estimator.cuda()

        self.fake_track_list = []
        self.distance_list = []

    def run_evaluation(self):
        tracking_dist = gather_tensor_list(self.distance_list)
        tracking_dist = torch.mean(tracking_dist).item()
        self.distance_list.clear()
        result_dict = {"point_tracking": tracking_dist}
        return result_dict

    @torch.no_grad()
    def _get_query(self, first_frame: np.ndarray) -> torch.Tensor:
        """First frame to query.
        1. detection & pose estimation
        2. generate grid

        Args:
            first_frame (np.ndarray): [H, W, C], RGB, range in [0, 255].
        """
        data_samples = process_one_image(
            first_frame, self.detector, self.pose_estimator
        )
        initial_tracks = get_local_patch_tracks(
            data_samples,
            mouse_grid_size=self.mouse_grid_size,
            eye_grid_size=self.eye_grid_size,
        )
        query = torch.cat(
            [
                torch.zeros(initial_tracks.shape[0], 1),
                torch.from_numpy(initial_tracks),
            ],
            dim=1,
        )[None].to(dtype=torch.float32, device="cuda")  # [1, N, 3]

        return query

    def _preprocess_one_clip(
        self,
        video_frames: torch.Tensor,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the tracking result of one video clip.

        Args:
            video_frames: Video frames, [F, C, H, W], RGB, range in [0, 1].
        """
        is_first_step = True

        for idx in range(video_frames.shape[0]):
            if idx != 0 and idx % self.tracker.step == 0:
                video_chunk = video_frames[idx - self.tracker.step * 2 : idx][None]

                pred_tracks, pred_visibility = self.tracker(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=query,
                )

                is_first_step = False

        # final frame
        video_chunk = video_frames[
            -(idx % self.tracker.step) - self.tracker.step - 1 :
        ][None]
        pred_tracks, pred_visibility = self.tracker(
            video_chunk,
            is_first_step=is_first_step,
            queries=query,
        )

        return pred_tracks, pred_visibility

    @staticmethod
    def _calc_score(
        real_tracks: torch.Tensor,
        fake_tracks: torch.Tensor,
        enable_retargeting: bool = False,
    ):
        """
        Args:
            real_tracks, fake_tracks(torch.Tensor): Predicted tracks, shape in [T, N, 2].
            enable_retargeting (bool): Whether to enable retargeting. Default: False.
        """

        if enable_retargeting:
            fake_tracks_np = fake_tracks.cpu().numpy()
            real_tracks_np = real_tracks.cpu().numpy()
            # follow mimic-motion https://github.com/Tencent/MimicMotion/blob/ce20af19cbba7c8a9ffb8c573817d43e75dda41f/mimicmotion/dwpose/preprocess.py#L9
            # 1. fit fake y to real via linear regression
            ay, by = np.polyfit(fake_tracks_np[0, :, 1], real_tracks_np[0, :, 1], 1)

            # 2. fit fake x to real via simple linear scaling
            ax = (real_tracks_np[0, :, 0].max() - real_tracks_np[0, :, 0].min()) / (
                fake_tracks_np[0, :, 0].max() - fake_tracks_np[0, :, 0].min()
            )
            bx = np.mean(real_tracks_np[0, :, 0] - ax * fake_tracks_np[0, :, 0])
            a = torch.Tensor([ax, ay]).to(fake_tracks)
            b = torch.Tensor([bx, by]).to(fake_tracks)

            # 3. apply the transformation to fake_tracks
            fake_tracks = fake_tracks * a + b

        real_norm_scale = torch.stack(
            [
                real_tracks[0, :, 0].max() - real_tracks[0, :, 0].min(),
                real_tracks[0, :, 1].max() - real_tracks[0, :, 1].min(),
            ]
        )
        fake_norm_scale = torch.stack(
            [
                fake_tracks[0, :, 0].max() - fake_tracks[0, :, 0].min(),
                fake_tracks[0, :, 1].max() - fake_tracks[0, :, 1].min(),
            ]
        )

        real_tracks_norm = real_tracks / real_norm_scale
        fake_tracks_norm = fake_tracks / fake_norm_scale

        tracks_diff_l2 = torch.norm(real_tracks_norm - fake_tracks_norm, p=2, dim=-1)
        diff_distance = tracks_diff_l2.sum(dim=0).mean()

        return diff_distance

    @torch.no_grad()
    def feed_one_sample(self, sample: SAMPLE_TYPE, mode: str):
        """Feed one sample.
        Args:
            sample (torch.Tensor | dict): If sample is tensor, sample should be
                [F, C, H, W], order in RGB, range in (0, 1). Otherwise, is
        """
        # NOTE: the input for tracker should be [0, 255].

        if mode == "fake":
            fake_sample = sample * 255  # [0, 255]
            first_frame = (
                fake_sample[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )  # [H, W, C]
            query = self._get_query(first_frame)

            pred_tracking, pred_visibility = self._preprocess_one_clip(
                fake_sample.cuda(),
                query,
            )
            self.fake_track_list.append(pred_tracking)

            vis_dict = {}
            if self.enable_vis:
                fake_tracking_vis = self.visualier.visualize(
                    fake_sample[None],  # [1, F, C, H, W]
                    tracks=pred_tracking,  # [1, F, N, 2]
                    visibility=pred_visibility,  # [1, F, N]
                    save_video=False,
                )[0]  # [F, C, H, W]
                fake_tracking_np_list = [
                    frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    for frame in fake_tracking_vis
                ]
                vis_dict = {"tracking_fake": fake_tracking_np_list}
            return vis_dict, {}

        elif mode == "real":
            driving_sample = np.stack(sample["driving_video"], axis=0)  # [f, h, w, c]
            driving_sample = (
                torch.from_numpy(driving_sample)
                .permute(0, 3, 1, 2)
                .to(dtype=torch.float32, device="cuda")
            )  # [F, C, H, W]
            query = self._get_query(sample["driving_video"][0])  # [1, N, 3]

            pred_tracking, pred_visibility = self._preprocess_one_clip(
                driving_sample,
                query,
            )

            assert len(self.fake_track_list) == 1, (
                "When call feed_one_sample with mode `real`, "
                "`PointTracking.fake_track_list` should only contain one element. "
                "Please check your code!"
            )
            fake_pred_tracking = self.fake_track_list.pop()
            fake_pred_tracking = fake_pred_tracking[0]  # [T, N, 2]
            real_pred_tracking = pred_tracking[0]  # [T, N, 2]

            tracking_distance = self._calc_score(
                real_pred_tracking,
                fake_pred_tracking,
                enable_retargeting=True,
            )
            self.distance_list.append(tracking_distance[None])

            vis_dict = {}
            if self.enable_vis:
                real_tracking_vis = self.visualier.visualize(
                    driving_sample[None],  # [1, F, C, H, W]
                    tracks=pred_tracking,  # [1, F, N, 2]
                    visibility=pred_visibility,  # [1, F, N]
                    save_video=False,
                )[0]  # [F, C, H, W]
                real_tracking_np_list = [
                    frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    for frame in real_tracking_vis
                ]
                vis_dict = {"tracking_real": real_tracking_np_list}

            return vis_dict, {"point_tracking": tracking_distance.item()}
        else:
            raise ValueError(f"Do not support mode {mode}.")
