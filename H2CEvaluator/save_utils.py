import os.path as osp
from typing import List, Optional

import cv2
import numpy as np
import torch


def load_mp4_to_torch(file_name: str):
    """
    Load a local mp4 file to torch.Tensor in range [0, 1] and [F, C, H, W].
    """
    if not osp.exists(file_name):
        return None

    try:
        cap = cv2.VideoCapture(file_name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        video = torch.from_numpy(np.array(frames))  # [F, H, W, C]
        video = video.permute(0, 3, 1, 2) / 255.0  # [F, C, H, W]
    except Exception as e:
        print(e)
        video = None

    return video


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
        assert (
            save_name is not None
        ), '"save_name" must be passed if "save_as_frames" is False.'
        resumed_video = load_mp4_to_torch(save_name)

        if resumed_video is not None:
            save_suffix = save_name.split(".")[-1]
            sample_combine_name = save_name.replace(
                f".{save_suffix}", f"_comb.{save_suffix}"
            )
            resumed_combine_video = load_mp4_to_torch(sample_combine_name)
            if resumed_combine_video is not None:
                w = resumed_video.shape[2]
                resumed_cond = resumed_combine_video[:, :, :, w : w * 2]

                if resumed_cond.shape != resumed_video.shape:
                    print(
                        f"Condition shape {resumed_cond.shape} does not "
                        f"match video shape {resumed_video.shape}. Ignore."
                    )
                    resumed_cond = None
            else:
                resumed_cond = None

    if n_frames is not None:
        if resumed_video is not None and resumed_video.shape[0] != n_frames:
            if resumed_video.shape[0] < n_frames:
                resumed_video = resumed_cond = None
                print(f"{save_name} is found, but shorter than desired length. Ignore.")
            else:
                resumed_video = resumed_video[:n_frames]
                if resumed_cond is not None:
                    resumed_cond = resumed_cond[:n_frames]
                print(
                    f"{save_name} is found, but longer than desired length. Truncate."
                )

    if resumed_video is None:
        print(f"Did not find {save_name}")
    else:
        resumed_video = resumed_video.contiguous()
        if resumed_cond is not None:
            resumed_cond = resumed_cond.contiguous()

    return resumed_video, resumed_cond
