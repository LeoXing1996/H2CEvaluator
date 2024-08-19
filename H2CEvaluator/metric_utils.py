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
