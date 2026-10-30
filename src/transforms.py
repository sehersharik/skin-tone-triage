"""Image transforms. Training-time augmentation is only ever added to the
training transform; validation and test always use `eval_transform`."""
from torchvision import transforms as T


def eval_transform(images_cfg):
    size = images_cfg["input_size"]
    return T.Compose([
        T.Resize(size),          # short side -> 224, aspect ratio kept
        T.CenterCrop(size),      # 224 x 224
        T.ToTensor(),
        T.Normalize(images_cfg["mean"], images_cfg["std"]),
    ])


def train_transform(images_cfg):
    if images_cfg["augmentation"] == "none":
        return eval_transform(images_cfg)
    raise ValueError(f"Unknown augmentation: {images_cfg['augmentation']}")
