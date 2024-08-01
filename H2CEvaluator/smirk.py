from typing import Dict, Union, List
from PIL import Image

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.transform import estimate_transform, warp

from .SMIRK.FLAME.FLAME import FLAME
from .SMIRK.renderer.renderer import Renderer
from .SMIRK.smirk_encoder import SmirkEncoder

SAMPLE_TYPE = Union[torch.Tensor, dict]


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, DST_PTS)

    return tform


class SMIRK:
    """
    Metric for face parameters extracted from SMIRK.

    Args:
        crop_face (bool):
        render_orig (bool):
        enable_vis (bool): Whether enable visualization.
    """

    def __init__(
        self,
        model_path: str = "./models/SMIRK_em1.pt",
        flame_model_path: str = "./models/FLAME2020/generic_model.pkl",
        flame_lmk_embedding_path: str = "./models/landmark_embedding.npy",
        flame_l_eyelid_path: str = "./models/l_eyelid.npy",
        flame_r_eyelid_path: str = "./models/r_eyelid.npy",
        flame_mask_path: str = "./models/FLAME_masks.pkl",
        head_template_path: str = "./models/head_template.obj",
        mediapipe_landmark_embedding: str = "./models/mediapipe_landmark_embedding.npz",
        mediapipe_detector_path: str = "./models/face_landmarker.task",
        enable_expression: bool = True,
        enable_head_pose: bool = True,
        crop_face: bool = False,
        render_orig: bool = True,
        enable_vis: bool = True,
    ):
        smirk_encoder = SmirkEncoder().cuda()
        checkpoint = torch.load(model_path)
        checkpoint_encoder = {
            k.replace("smirk_encoder.", ""): v
            for k, v in checkpoint.items()
            if "smirk_encoder" in k
        }  # checkpoint includes both smirk_encoder and smirk_generator
        smirk_encoder.load_state_dict(checkpoint_encoder)
        smirk_encoder.eval()

        flame = FLAME(
            flame_model_path=flame_model_path,
            flame_lmk_embedding_path=flame_lmk_embedding_path,
            l_eyelid_path=flame_l_eyelid_path,
            r_eyelid_path=flame_r_eyelid_path,
            mediapipe_landmark_embedding=mediapipe_landmark_embedding,
        ).cuda()

        renderer = Renderer(
            obj_filename=head_template_path, flame_mask=flame_mask_path
        ).cuda()

        base_options = python.BaseOptions(model_asset_path=mediapipe_detector_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.encoder = smirk_encoder
        self.flame = flame
        self.renderer = renderer

        self.enable_expression = enable_expression
        self.enable_head_pose = enable_head_pose

        self.real_expression_list = []
        self.fake_expression_list = []
        self.real_head_pose_list = []
        self.fake_head_pose_list = []

        self.crop_face = crop_face
        self.render_orig = render_orig

        self.enable_vis = enable_vis

    def prepare(self, *args, **kwargs):
        """Do not need prepare. Do nothing."""
        pass

    def _extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Get landmark via mediapipe
        modified from https://github.com/georgeretsi/smirk/blob/main/utils/mediapipe_utils.py  # noqa

        Args:
            image (image): [H, W, C], order in RGB, range in (0, 255), uint8 type.
        """
        height, width = image.shape[0], image.shape[1]

        image_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
        )

        detection_result = self.detector.detect(image_mp)

        if len(detection_result.face_landmarks) == 0:
            print("No face detected")
            return None

        face_landmarks = detection_result.face_landmarks[0]

        face_landmarks_numpy = np.zeros((478, 3))

        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [
                landmark.x * width,
                landmark.y * height,
                landmark.z,
            ]

        return face_landmarks_numpy

    def _extract_one_frame(
        self, frame: Union[torch.Tensor, Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract face feature and head pose feature for **one frame**.

        This function is modified from https://github.com/georgeretsi/smirk/blob/main/demo.py  #noqa

        Args:
            frame (torch.Tensor): [C, H, W], order in RGB, range in [0, 1].

        Returns:
            A dict contains rerendered face and face parameters.
        """
        if isinstance(frame, torch.Tensor):
            frame_np = (frame.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        elif isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            raise TypeError(
                f"Only support torch.Tensor or Image.Image, but receive {type(frame)}."
            )

        orig_height, orig_width = frame_np.shape[0], frame_np.shape[1]
        landmarks = self._extract_landmarks(frame_np)

        if self.crop_face:
            landmarks = landmarks[..., :2]

            tform = crop_face(frame_np, landmarks, scale=1.4, image_size=224)

            cropped_frame = warp(
                frame_np,
                tform.inverse,
                output_shape=(224, 224),
                preserve_range=True,
            ).astype(np.uint8)

            cropped_landmarks = np.dot(
                tform.params,
                np.hstack([landmarks, np.ones([landmarks.shape[0], 1])]).T,
            ).T
            cropped_landmarks = cropped_landmarks[:, :2]
        else:
            cropped_frame = frame_np
            cropped_landmarks = landmarks

        cropped_frame = cv2.resize(cropped_frame, (224, 224))
        cropped_frame = (
            torch.tensor(cropped_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).cuda()

        outputs = self.encoder(cropped_frame)

        flame_output = self.flame.forward(outputs)
        renderer_output = self.renderer.forward(
            flame_output["vertices"],
            outputs["cam"],
            landmarks_fan=flame_output["landmarks_fan"],
            landmarks_mp=flame_output["landmarks_mp"],
        )

        rendered_img = renderer_output["rendered_img"]  # [1, 3, 224, 224]

        if self.render_orig:
            if self.crop_face:
                rendered_img_numpy = (
                    rendered_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    * 255.0
                ).astype(np.uint8)
                rendered_img_orig = warp(
                    rendered_img_numpy,
                    tform,
                    output_shape=(orig_height, orig_width),
                    preserve_range=True,
                ).astype(np.uint8)
                # back to pytorch to concatenate with full_image
                rendered_img_orig = (
                    torch.Tensor(rendered_img_orig)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
            else:
                rendered_img_orig = F.interpolate(
                    rendered_img,
                    (orig_height, orig_width),
                    mode="bilinear",
                ).cpu()
            output_dict = {"rerender_frame": rendered_img_orig}
        else:
            output_dict = {"rerender_frame": rendered_img}

        # keys: 'pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params'
        output_dict.update(outputs)
        # keys: 'rendered_img', 'transformed_vertices', 'landmarks_fan', 'landmarks_mp
        output_dict.update(renderer_output)

        return output_dict

    def run_evaluation(self):
        # TODO:
        return {}

    @torch.no_grad()
    def feed_one_sample(self, sample: SAMPLE_TYPE, mode: str):
        """Feed one sample.

        Args:
            sample (torch.Tensor or Dict): If sample is torch.Tensor, tensor
                is shape in [F, C, H, W], order in RGB, range in (0, 1).
        """

        expression_key = "transformed_vertices"  # TODO: maybe use other keys
        headpose_key = "pose_params"
        vis_key = "rerender_frame"

        if mode == "fake":
            fake_dict_list = []
            for frame in sample:
                fake_dict_ = self._extract_one_frame(frame)
                fake_dict_list.append(fake_dict_)

            keys = fake_dict_list[0].keys()
            fake_dict = {
                k: torch.cat([d[k] for d in fake_dict_list], dim=0) for k in keys
            }
            if self.enable_expression:
                self.fake_expression_list.append(fake_dict[expression_key])
            if self.enable_head_pose:
                self.fake_head_pose_list.append(fake_dict[headpose_key])

            # return item (maybe items) for visualization
            if self.enable_vis:
                tensor_to_vis = fake_dict[vis_key]
                pil_to_vis = [
                    Image.fromarray(
                        (ten * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    )
                    for ten in tensor_to_vis
                ]
                return {f"{vis_key}_fake": pil_to_vis}
            else:
                return {}

        elif mode == "real":
            real_dict_list = []
            real_sample: List[Image.Image] = sample["pose_images"]
            for frame in real_sample:
                real_dict_ = self._extract_one_frame(frame)
                real_dict_list.append(real_dict_)

            keys = real_dict_list[0].keys()
            real_dict = {
                k: torch.cat([d[k] for d in real_dict_list], dim=0) for k in keys
            }

            if self.enable_expression:
                self.real_expression_list.append(real_dict[expression_key])
            if self.enable_head_pose:
                self.real_head_pose_list.append(real_dict[headpose_key])

            # return item (maybe items) for visualization
            if self.enable_vis:
                tensor_to_vis = real_dict[vis_key]
                pil_to_vis = [
                    Image.fromarray(
                        (ten * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    )
                    for ten in tensor_to_vis
                ]
                return {f"{vis_key}_real": pil_to_vis}
            else:
                return {}
        else:
            raise ValueError(f"Do not support mode {mode}.")
