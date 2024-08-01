import torch
import torchvision
from accelerate import PartialState

from .dist_utils import gather_all_tensors
from .hyperIQA.models import HyperNet, TargetNet


class HyperIQA:
    def __init__(
        self,
        model_path="./models/koniq_pretrained.pkl",
        n_randn_crop: int = 10,
    ):
        model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        model_hyper.train(False)

        model_hyper.load_state_dict((torch.load(model_path)))
        self.model = model_hyper

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=224),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.transforms = transforms

        self.n_randn_crop = n_randn_crop

        self.iqa_scores = []

    def prepare(self, *args, **kwargs):
        """Do not need prepare. Do nothing."""
        return

    def run_evaluation(self):
        if len(self.iqa_scores) > 1:
            iqa_scores = torch.cat(self.iqa_scores)
        else:
            iqa_scores = self.iqa_scores[0]

        if PartialState().use_distributed:
            iqa_scores = gather_all_tensors(iqa_scores)
            iqa_scores = torch.cat(iqa_scores)

        score = iqa_scores.mean().item()

        result_dict = {"iqa_score": score}

        self.iqa_scores.clear()

        return result_dict

    @torch.no_grad()
    def feed_one_sample(self, sample: torch.Tensor, mode: str):
        """Feed one sample.

        Args:
            sample (torch.Tensor): [F, C, H, W], order in RGB, range in (0, 1).
        """
        if mode == "fake":
            pred_scores = []

            for _ in range(self.n_randn_crop):
                sample_inp = self.transforms(sample.clone())
                paras = self.model(sample_inp.cuda())
                model_target = TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False
                pred = model_target(paras["target_in_vec"])
                pred_scores.append(pred)

            score = torch.mean(torch.cat(pred_scores), dim=0, keepdim=True)
            self.iqa_scores.append(score)

            # no intermedia results for visualization, return empty dict
            return {}

        elif mode == "real":
            # do not need real samples, return
            return {}

        else:
            raise ValueError(f"Do not support mode {mode}.")
