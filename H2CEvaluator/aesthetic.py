import clip
import torch
import torch.nn as nn
from accelerate import PartialState

from .dist_utils import gather_all_tensors
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Aesthetic:
    def __init__(self, model_path: str = "./models/sac+logos+ava1-l14-linearMSE.pth"):
        self.model = MLP(768)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.cuda().eval()

        self.clip, _ = clip.load("ViT-L/14", device="cuda")
        self.preprocessor = _transform(self.clip.visual.input_resolution)

        self.score_list = []

    def prepare(self, *args, **kwargs):
        """Do not need prepare. Do nothing."""
        return

    def run_evaluation(self):
        score_list = torch.cat(self.score_list)
        if PartialState().use_distributed:
            score = gather_all_tensors(score_list)
            score = torch.cat(score)
        else:
            score = score_list

        score = torch.mean(score).item()
        self.score_list.clear()
        result_dict = {"aesthetic_score": score}
        return result_dict

    @staticmethod
    def _normalized(a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    @torch.no_grad()
    def feed_one_sample(self, sample, mode: str):
        """Feed one sample.

        Args:
            sample (torch.Tensor): [F, C, H, W], order in RGB, range in (0, 1).
        """
        if mode == "fake":
            fake_sample = sample  # [0, 1]
            fake_sample = fake_sample.cuda()
            fake_sample = self.preprocessor(fake_sample)

            img_feature = self.clip.encode_image(fake_sample).cpu().numpy()
            im_emb_arr = self._normalized(img_feature)
            im_emb = torch.from_numpy(im_emb_arr).cuda().type(torch.cuda.FloatTensor)

            score = self.model(im_emb)
            self.score_list.append(score)

        elif mode == "real":
            pass

        else:
            raise ValueError(f"Do not support mode {mode}.")

        # no intermedia results for visualization, return empty dict
        return {}
