# Evaluator for Human2Character😘

A powerful evaluator for Human2Character project. Support various metrics and distributed evaluation.

Supported metrics:
* [HyperIQA](https://github.com/SSL92/hyperIQA)
* [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).
* [SMIRK-Face and head pose](https://github.com/georgeretsi/smirk)
* [SSIM](https://github.com/VainF/pytorch-msssim)
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
* [Aesthetic Score](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* [FID](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/frechet_inception_distance.py)
* [FVD](https://github.com/wilson1yan/VideoGPT/blob/master/scripts/compute_fvd.py)
* [Point Tracking](https://github.com/facebookresearch/co-tracker)

## Install

### 0. Clone the repo and install

Clone the repo and submodules with the following command.

```bash
git clone git@github.com:LeoXing1996/H2CEvaluator.git

# install
pip install -e .

# install pre-commit hook if you want to contribute!
pip install pre-commit
pre-commit install
```

We do not provide official `requirements.txt` now, you should install all dependencies by youself ^_^.


### 1. Basic Usage

To use [H2CEvaluator](./H2CEvaluator/evaluator.py), you need to prepare `dataset`, `pipeline`, and configs for metrics you want to evaluate. Here is a use case,

```python
# build pipeline and dataset
pipe = Pose2VideoPipeline(
    vae=vae,
    image_encoder=image_enc,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    pose_guider=pose_guider,
    scheduler=scheduler,
).to('cuda')
dataset = PatchTestDataset(config)

# default inference kwargs for pipeline
pipeline_kwargs = {
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "height": args.H,
    "width": args.W,
}
evaluator = Evaluator(
    dataset,
    pipeline_kwargs,
    metric_list=metric_config,
    save_image=True,
    save_as_frames=True,
    save_path=save_dir,
)
print("Start Evaluation.....")
result = evaluator.run_eval(pipe)

print(result)
# {'iqa_score': 55.85905075073242, 'arcface_score': 0.5672686100006104}
```

#### Dataset requirements

In current version, the output of `dataset.__getitem__` should be as follow:

```python
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
```

#### Pipeline requirements

The output of `pipeline` should contain `videos` attribute. And `videos` should be a `torch.Tensor` which shape is `[1, F, C, H, W]` and value range should be `[0, 1]`.

### 2. Config and preparation for each metric

You do not need to prepare models for each metric. Models for each metric will be loaded automatically to `DEFAULT_CACHE_DIR` (i.e., `~/.cache/H2CEvaluator`).

#### 2.1 [HyperIQA](./H2CEvaluator/hyper_iqa.py)

```yaml
metrics:
  - type: 'HyperIQA'
    n_randn_crop: 10  # number of random crop for each sample
```

#### 2.2 [ArcFace](./H2CEvaluator/arcface.py)

```yaml
metrics:
  - type: "arcface"
    model_type: 'r100'
```

#### 2.3 [SMIRK](./H2CEvaluator/smirk.py)

```yaml
metrics:
  - type: "SMIRK"
    enable_expression: true  # evaluate expression metric
    enable_head_pose: true  # evaluate head pose metric
    enable_vis: true  # save reconstructed face
```

#### 2.4 [SSIM](./H2CEvaluator/ssim.py)

```yaml
metrics:
  - type: 'SSIM'
```

#### 2.5 [LPIPS](./H2CEvaluator/lpips.py)

```yaml
metrics:
  - type: 'LPIPS'
```

#### 2.6 [Aesthetic Score](./H2CEvaluator/aesthetic.py)

```yaml
metrics:
  - type: 'Aesthetic'
```

#### 2.7 [FID](./H2CEvaluator/fid.py) and [FVD](./H2CEvaluator/fvd.py)

You need to prepare the inception feature for real data via `scripts/prepreocess_inception.py`.
Please refer to [this document](./scripts/README.md) for more information.

```yaml
metrics:
  - type: "FID"
    cache_path: ~/.cache/H2CEvaluator/fid_real_cache_17edbf9bf460120eb820adc439279af7.pt  # path to inception feature
  - type: "FVD"
    cache_path: ~/.cache/H2CEvaluator/fvd_real_cache_17edbf9bf460120eb820adc439279af7.pt  # path to inception feature
```

#### 2.8 [Point Tracking](./H2CEvaluator/point_tracking.py)

```yaml
metrics:
  - type: PointTracking
```
