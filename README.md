# Evaluator for Human2Character😘

A powerful evaluator for Human2Character project. Support various metrics and distributed evaluation.

Supported metrics:
* [HyperIQA](https://github.com/SSL92/hyperIQA)
* [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).
* [SSIM](https://github.com/VainF/pytorch-msssim)
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
* [Aesthetic Score](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* [FID](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/frechet_inception_distance.py)
* [FVD](https://github.com/wilson1yan/VideoGPT/blob/master/scripts/compute_fvd.py)
* [SMIRK-Face and head pose](https://github.com/georgeretsi/smirk)

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
output_dict = {
    "src_image": self.src_list[index],
    "pose_images": self.patch_list[index],
    # NOTE: pack ref_image as list for collate_fn
    "ref_image": [self.ref_image_list[index]],
    "patch_width": self.patch_width,
    "patch_height": self.patch_height,
    "video_length": self.video_length_list[index],
    "save_name_list": self.save_name_list[index],
}
```

#### Pipeline requirements

The output of `pipeline` should contain `videos` attribute. And `videos` should be a `torch.Tensor` which shape is `[1, F, C, H, W]` and value range should be `[0, 1]`.

### 2. Config and preparation for each metric

#### 2.1 [HyperIQA](./H2CEvaluator/hyper_iqa.py)

You should download [`koniq_pretrained.pkl`](https://drive.google.com/file/d/1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE/view). And the config should be like this:

```yaml
metrics:
  - type: 'HyperIQA'
    model_path: './models/koniq_pretrained.pkl'
    n_randn_crop: 10  # number of random crop for each sample
```

#### 2.2 [ArcFace](./H2CEvaluator/arcface.py)

You should download `glint360k_cosface_r100_fp16_0.1` from this [url](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC), and unzip it, and then put `backbone.pth` under models folder and rename it to `glint360k_cosface_r100_fp16_0.1.pth`. And the config should be like this:

```yaml
metrics:
  - type: "arcface"
    model_type: 'r100'
    model_path: './models/glint360k_cosface_r100_fp16_0.1.pth'
```

#### 2.3 [SMIRK](./H2CEvaluator/smirk.py)

* Download `SMIRK_em1.pt` from this [url](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view), and put it to your model directory.
* Download `FLAME2020.zip` from this [url](https://flame.is.tue.mpg.de/download.php), unzip it to your model directory.
* Download `landmark_embedding.npy` from this [url](https://github.com/georgeretsi/smirk/blob/main/assets/landmark_embedding.npy) and put it to your model directory.
* Download `head_template.obj` from this [url](https://github.com/georgeretsi/smirk/blob/main/assets/head_template.obj) and put it to your model directory.
* Download `face_landmarker.task` from this [url](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) and put it to your model directory.
* Download [`l_eyelid.npy`](https://github.com/georgeretsi/smirk/blob/main/assets/l_eyelid.npy) and [`r_eyelib.npy`](https://github.com/georgeretsi/smirk/blob/main/assets/r_eyelid.npy) and put them to your model directory.
* Download [`mediapipe_landmark_embedding.npz`](https://github.com/georgeretsi/smirk/blob/main/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz) and put it to your model directory.
* Downlaad [`FLAME_masks.pkl`](https://github.com/georgeretsi/smirk/blob/main/assets/FLAME_masks/FLAME_masks.pkl) and put it to your model directory.

And the config should be like this:

```yaml
metrics:
  - type: "SMIRK"

    model_path: "./models/SMIRK_em1.pt"
    flame_model_path:  "./models/FLAME2020/generic_model.pkl"
    flame_lmk_embedding_path: "./models/landmark_embedding.npy"
    flame_l_eyelid_path: "./models/l_eyelid.npy"
    flame_r_eyelid_path: "./models/r_eyelid.npy"
    flame_mask_path: "./models/FLAME_masks.pkl"
    head_template_path: "./models/head_template.obj"
    mediapipe_landmark_embedding: "./models/mediapipe_landmark_embedding.npz"
    mediapipe_detector_path: "./models/face_landmarker.task"

    enable_expression: true  # evaluate expression metric
    enable_head_pose: true  # evaluate head pose metric

    enable_vis: true  # save reconstructed face
```

#### 2.4 [SSIM](./H2CEvaluator/ssim.py)

Do not need prepare any model, just config like this:

```yaml
metrics:
  - type: 'SSIM'
```

#### 2.5 [LPIPS](./H2CEvaluator/lpips.py)

Do not need prepare any model, just config like this:

```yaml
metrics:
  - type: 'LPIPS'
```

#### 2.6 [Aesthetic Score](./H2CEvaluator/aesthetic.py)

Download [`sac+logos+ava1-l14-linearMSE.pth`](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) and put it to your model directory.

And the config should be like this:

```yaml
metrics:
  - type: 'Aesthetic'
    model_path: './models/sac+logos+ava1-l14-linearMSE.pth'
```

#### 2.7 [FID](./H2CEvaluator/fid.py) and [FVD](./H2CEvaluator/fvd.py)

```yaml
  - type: "FID"
    model_path: models/inception-2015-12-05.pt
    cache_path: ~/.cache/H2CEvaluator/fid_real_cache_17edbf9bf460120eb820adc439279af7.pt
  - type: "FVD"
    model_path: models/i3d_torchscript.pt
    cache_path: ~/.cache/H2CEvaluator/fvd_real_cache_17edbf9bf460120eb820adc439279af7.pt
```
