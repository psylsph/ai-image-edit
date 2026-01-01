"""Image upscaling using Real-ESRGAN."""

from pathlib import Path
from urllib import request

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from models.model_downloader import get_device

_DEVICE = get_device()
_CACHE_DIR = Path.home() / ".cache" / "ai-image-edit" / "realesrgan"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_2X = None
_MODEL_4X = None


def _pixel_unshuffle(x, scale):
    """Pixel unshuffle (inverse of pixel shuffle)."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * scale * scale, h // scale, w // scale)
    return x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass through residual dense blocks."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        """Forward pass through 3 residual dense blocks with residual scaling."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def _make_layer(block, n_layers, **kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class RRDBNetX2Plus(nn.Module):
    """Real-ESRGAN x2plus model with pixel_unshuffle."""

    def __init__(self, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(12, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass with pixel unshuffle for 2x upscaling."""
        feat = _pixel_unshuffle(x, 2)
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDBNetX4Plus(nn.Module):
    """Real-ESRGAN x4plus model without pixel_unshuffle."""

    def __init__(self, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass without pixel unshuffle for 4x upscaling."""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def _download_model(scale):
    """Download Real-ESRGAN model."""
    x2_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    x4_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    if scale == 2:
        filename = "RealESRGAN_x2plus.pth"
        url = x2_url
    else:
        filename = "RealESRGAN_x4plus.pth"
        url = x4_url

    model_path = _CACHE_DIR / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        request.urlretrieve(url, str(model_path))
        print(f"Model saved to {model_path}")
    return str(model_path)


def _load_model(scale):
    """Load Real-ESRGAN model."""
    global _MODEL_2X, _MODEL_4X

    if scale == 2:
        if _MODEL_2X is not None:
            return _MODEL_2X

        try:
            model_path = _download_model(2)
            _MODEL_2X = RRDBNetX2Plus(num_feat=64, num_block=23, num_grow_ch=32)

            loadnet = torch.load(model_path, map_location=_DEVICE)
            if 'params_ema' in loadnet:
                _MODEL_2X.load_state_dict(loadnet['params_ema'], strict=False)
            else:
                _MODEL_2X.load_state_dict(loadnet, strict=False)

            _MODEL_2X = _MODEL_2X.to(_DEVICE)
            _MODEL_2X.eval()
            print("Real-ESRGAN x2plus model loaded successfully")
            return _MODEL_2X

        except (OSError, RuntimeError, ImportError) as exc:
            print(f"Warning: Real-ESRGAN x2plus model not available ({exc})")
            return None

    else:
        if _MODEL_4X is not None:
            return _MODEL_4X

        try:
            model_path = _download_model(4)
            _MODEL_4X = RRDBNetX4Plus(num_feat=64, num_block=23, num_grow_ch=32)

            loadnet = torch.load(model_path, map_location=_DEVICE)
            if 'params_ema' in loadnet:
                _MODEL_4X.load_state_dict(loadnet['params_ema'], strict=False)
            else:
                _MODEL_4X.load_state_dict(loadnet, strict=False)

            _MODEL_4X = _MODEL_4X.to(_DEVICE)
            _MODEL_4X.eval()
            print("Real-ESRGAN x4plus model loaded successfully")
            return _MODEL_4X

        except (OSError, RuntimeError, ImportError) as exc:
            print(f"Warning: Real-ESRGAN x4plus model not available ({exc})")
            return None


def _prepare_input(image_rgb, scale):
    """Prepare input array with proper padding and normalization."""
    pad_h = (scale - image_rgb.height % scale) % scale
    pad_w = (scale - image_rgb.width % scale) % scale

    image_array = np.array(image_rgb).astype(np.float32) / 255.0

    if pad_h > 0 or pad_w > 0:
        image_array = np.pad(
            image_array,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )

    return image_array, pad_h, pad_w


def upscale_image(image: Image.Image, scale: int = 2) -> tuple[Image.Image, Image.Image]:
    """Upscale image using Real-ESRGAN.

    Args:
        image: Input PIL Image
        scale: Upscaling factor (2 or 4)

    Returns:
        Tuple of (upscaled_image, debug_image)
    """
    if scale not in [2, 4]:
        raise ValueError("Scale must be 2 or 4")

    debug_image = image.copy()

    original_mode = image.mode
    alpha_channel = None
    if original_mode == 'RGBA':
        image_rgb = image.convert('RGB')
        alpha_channel = image.split()[-1]
    elif original_mode == 'L':
        image_rgb = image.convert('RGB')
    else:
        image_rgb = image

    model = _load_model(scale)

    if model is None:
        new_size = (image_rgb.width * scale, image_rgb.height * scale)
        upscaled = image_rgb.resize(new_size, Image.Resampling.BICUBIC)
    else:
        image_array, pad_h, pad_w = _prepare_input(image_rgb, scale)

        input_tensor = (
            torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0).to(_DEVICE)
        )

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_array = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        out_h = image_rgb.height * scale - pad_h * scale
        output_array = output_array[:out_h, :image_rgb.width * scale - pad_w * scale]
        output_array = (output_array * 255.0).clip(0, 255).astype(np.uint8)
        upscaled = Image.fromarray(output_array, mode='RGB')

    if original_mode == 'RGBA':
        alpha_resized = alpha_channel.resize(upscaled.size, Image.Resampling.LANCZOS)
        upscaled = upscaled.convert('RGBA')
        upscaled.putalpha(alpha_resized)

    return upscaled, debug_image


if __name__ == "__main__":
    print("Testing upscaling...")
    img = Image.new('RGB', (64, 64), color='red')
    result, debug = upscale_image(img, scale=2)
    print(f"Input size: {img.size}")
    print(f"Output size: {result.size}")
    print("Upscaling test passed!")
