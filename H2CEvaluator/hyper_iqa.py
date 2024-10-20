import os.path as osp
import torch
import torchvision

from .dist_utils import gather_tensor_list
from .hyperIQA.models import HyperNet, TargetNet

from .metric_utils import FileHashItem, MetricModelItems, DEFAULT_CACHE_DIR


class HyperIQA:
    metric_items = MetricModelItems(
        file_list=[
            FileHashItem(
                "koniq_pretrained.pkl",
                sha256="ff9277bcc68ecc10e77d88b6d0a32825ec3c85562095542734ec6212eaaf6d81",
            ),
        ],
        remote_subfolder="hyperiqa",
    )

    def __init__(
        self,
        model_dir: str = osp.join(DEFAULT_CACHE_DIR, "hyperiqa"),
        n_randn_crop: int = 10,
    ):
        self.metric_items.prepare_model(model_dir)
        model_path = f"{model_dir}/koniq_pretrained.pkl"
        model_hyper = HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
        model_hyper.train(False)

        model_hyper.load_state_dict((torch.load(model_path, map_location="cpu")))
        self.model = model_hyper.cuda()

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

    def run_evaluation(self):
        iqa_scores = gather_tensor_list(self.iqa_scores)
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
            return {}, {"hyperIQA": score.item()}

        elif mode == "real":
            # do not need real samples, return
            return {}, {}

        else:
            raise ValueError(f"Do not support mode {mode}.")
