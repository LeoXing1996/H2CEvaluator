# Useful scripts

## Preprocess Inception feature for `FID` and `FVD`: `preprocess_inception.py`

This script preprocesses the Inception features for `FID` and `FVD` calculation. It extract the Inception features of the given dataset and save them in a `.pt` file.
The name of the cache file will be calculated based on the meta info of the dataset:

```python
import os.path as osp
import hashlib

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

_, md5 = get_dataset_meta(dataset)
fvd_cache_path = osp.join(cache_path, f"fvd_real_cache_{md5}.pt")
fid_cache_path = osp.join(cache_path, f"fid_real_cache_{md5}.pt")
```

And by default, the calculated inception feature will be saved to `~/.cache/H2CEvaluator` (i.e., `DEFAULT_CACHE_DIR`).
If you want to change the cache directory, you can use `--cache-dir` argument.

If you want to use a custom dataset, please make sure your dataset contains above attributes.

Usage:
```bash
preprocess-inception --config scripts/default_config.yaml \
    # path for the dataset
    --reference-image-root datasets/portrait_animation_benchmark/reference/images \
    --driving-video-root datasets/portrait_animation_benchmark/driving/videos \
    # extract fvd
    --fvd \
    # extract fid
    --fid
```

If you only want to test this script and do not want to preprocess the entire dataset, you can use `--eval-mode` argument:
```bash
preprocess-inception scripts/preprocess.py --config scripts/default_config.yaml \
    # path for the dataset
    --reference-image-root datasets/portrait_animation_benchmark/reference/images \
    --driving-video-root datasets/portrait_animation_benchmark/driving/videos \
    # extract fvd
    --fvd \
    # extract fid
    --fid \
    # only process the first 3 samples, use `--eval-mode fast` for 15 samples
    --eval-mode debug \
```
