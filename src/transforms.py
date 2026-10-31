"""Image transforms.

Two separate ideas:
- Preprocessing (`images.preprocessing`) is deterministic and applied to every
  image the model sees, in training and in evaluation, so train and test images
  look alike: `none`, `shades_of_gray` colour constancy, or `clahe` contrast
  enhancement.
- Augmentation (`images.augmentation`) is random and only ever added to the
  training transform; validation and test always use `eval_transform`.
"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T


class ShadesOfGray:
    """Shades of Gray colour constancy (Finlayson & Trezzi, 2004).

    Estimates the colour of the light as the Minkowski p-norm mean of each
    channel, then rescales the channels so that light becomes neutral grey.
    p = 1 is "grey world" (plain mean), p = infinity is "max-RGB"; p = 6 is the
    usual choice for skin images. This removes colour casts from different
    phones and lighting; it does not change the image's overall brightness much.
    """

    def __init__(self, power=6):
        self.power = power

    def __call__(self, image):
        rgb = np.asarray(image, dtype=np.float64)
        illuminant = np.power(np.mean(np.power(rgb, self.power), axis=(0, 1)), 1 / self.power)
        illuminant = illuminant / np.linalg.norm(illuminant)          # unit length
        corrected = rgb / (illuminant * np.sqrt(3))                    # neutral light -> (1,1,1)
        return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalisation on the lightness channel.

    The image is split into tiles and each tile's histogram is equalised, with a
    clip limit so noise isn't amplified. Working on L of the LAB colour space
    changes contrast without shifting the colours.
    """

    def __init__(self, clip_limit=2.0, tile_grid=8):
        self.clip_limit, self.tile_grid = clip_limit, tile_grid

    def __call__(self, image):
        lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=(self.tile_grid, self.tile_grid))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def preprocessing(images_cfg):
    """The deterministic preprocessing step, or [] for none."""
    name = images_cfg.get("preprocessing", "none")
    if name == "none":
        return []
    if name == "shades_of_gray":
        return [ShadesOfGray(images_cfg.get("sog_power", 6))]
    if name == "clahe":
        return [CLAHE(images_cfg.get("clahe_clip_limit", 2.0), images_cfg.get("clahe_tile_grid", 8))]
    raise ValueError(f"Unknown preprocessing: {name}")


def to_tensor(images_cfg):
    return [T.ToTensor(), T.Normalize(images_cfg["mean"], images_cfg["std"])]


def eval_transform(images_cfg):
    size = images_cfg["input_size"]
    return T.Compose(preprocessing(images_cfg) + [
        T.Resize(size),          # short side -> 224, aspect ratio kept
        T.CenterCrop(size),      # 224 x 224
    ] + to_tensor(images_cfg))


def train_transform(images_cfg):
    augmentation = images_cfg["augmentation"]
    if augmentation == "none":
        return eval_transform(images_cfg)
    if augmentation == "standard":
        size = images_cfg["input_size"]
        return T.Compose(preprocessing(images_cfg) + [
            T.RandomRotation(20),                                  # +-20 degrees
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),           # crop 60-100% of the area
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # mild: small enough not to change the apparent skin tone much
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ] + to_tensor(images_cfg))
    raise ValueError(f"Unknown augmentation: {augmentation}")
