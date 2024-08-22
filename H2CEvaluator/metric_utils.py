import hashlib
import os
import os.path as osp
import shutil
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from huggingface_hub import hf_hub_download

DEFAULT_MODEL_REPO = "Leoxing/H2CEvaluator"

DEFAULT_CACHE_DIR = osp.abspath(osp.expanduser("~/.cache/H2CEvaluator"))

os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

SAMPLE_TYPE = Union[torch.Tensor, dict]


def get_dataset_meta(dataset):
    dataset_identity = [
        "fps",
        "width",
        "height",
        "max_frames",
        "interval",
        "default_fps",
        "samples",
    ]
    dataset_kwargs = {k: dataset.__dict__.get(k, None) for k in dataset_identity}
    md5 = hashlib.md5(str(dataset_kwargs).encode()).hexdigest()
    dataset_kwargs["md5"] = md5
    return dataset_kwargs, md5


def get_hash(file):
    """
    Return sha256 hash of file
    """
    sha256_hash = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@dataclass
class FileHashItem:
    file: str
    sha256: Optional[str] = None

    def check_file(self, local_dir: str) -> bool:
        """Check whether file is exist."""
        local_file_path = osp.join(local_dir, self.file)
        if not osp.exists(local_file_path) or not osp.isfile(local_file_path):
            return False
        else:
            return self.check_hash(local_file_path)

    def check_hash(self, local_path: str) -> bool:
        if self.sha256 is None:
            return True
        local_file_hash = get_hash(local_path)
        return local_file_hash == self.sha256

    @property
    def folder(self):
        return osp.dirname(self.file)

    @property
    def basename(self):
        return osp.basename(self.file)


@dataclass
class MetricModelItems:
    file_list: List[FileHashItem]

    remte_repo: str = DEFAULT_MODEL_REPO
    remote_subfolder: str = ""

    def prepare_model(self, model_dir: str) -> str:
        model_dir = model_dir or DEFAULT_CACHE_DIR

        local_basefolder = osp.dirname(model_dir)
        local_subfolder = osp.basename(osp.normpath(model_dir))

        for file in self.file_list:
            if not file.check_file(model_dir):
                subfolder = osp.join(self.remote_subfolder, file.folder)
                subfolder = subfolder[:-1] if subfolder.endswith("/") else subfolder
                download_path = hf_hub_download(
                    self.remte_repo,
                    file.basename,
                    subfolder=subfolder,
                    local_dir=local_basefolder,
                )
                if local_subfolder != subfolder:
                    tar_folder = osp.normpath(
                        osp.join(
                            local_basefolder,
                            local_subfolder,
                            file.folder,
                        )
                    )
                    tar_path = osp.join(tar_folder, file.basename)

                    os.makedirs(tar_folder, exist_ok=True)
                    shutil.move(
                        download_path,
                        tar_path,
                    )

        return model_dir
