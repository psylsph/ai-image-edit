"""Tests for the upscale module using real images."""
import pytest
from pathlib import Path
from PIL import Image

from app.pipeline.upscale import upscale_image


TEST_IMAGE_PATH = Path(__file__).parent / "image" / "office.png"
CAT_JPG_PATH = Path(__file__).parent / "image" / "cat.jpg"
CAT_HEIF_PATH = Path(__file__).parent / "image" / "cat.heif"


@pytest.fixture
def sample_image():
    """Load the test image."""
    return Image.open(TEST_IMAGE_PATH)


@pytest.fixture
def cat_jpg_image():
    """Load the cat.jpg test image."""
    return Image.open(CAT_JPG_PATH)


@pytest.fixture
def cat_heif_image():
    """Load the cat.heif test image."""
    return Image.open(CAT_HEIF_PATH)


@pytest.fixture
def small_sample_image():
    """Load and resize the test image for faster tests."""
    img = Image.open(TEST_IMAGE_PATH)
    img.thumbnail((128, 128))
    return img


class TestUpscaleImage:
    """Tests for upscale_image function with real images."""

    def test_returns_tuple(self, small_sample_image):
        """Test that function returns a tuple of two images."""
        result = upscale_image(small_sample_image, 2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_pil_images(self, small_sample_image):
        """Test that returned images are PIL Images."""
        result_image, debug = upscale_image(small_sample_image, 2)
        assert isinstance(result_image, Image.Image)
        assert isinstance(debug, Image.Image)

    def test_preserves_mode(self, small_sample_image):
        """Test that image mode is preserved."""
        rgb = small_sample_image.convert('RGB')
        rgba = small_sample_image.convert('RGBA')
        gray = small_sample_image.convert('L')

        result_rgb, _ = upscale_image(rgb, 2)
        result_rgba, _ = upscale_image(rgba, 2)
        result_gray, _ = upscale_image(gray, 2)

        assert result_rgb.mode == 'RGB'
        assert result_rgba.mode == 'RGBA'
        assert result_gray.mode == 'RGB'

    def test_grayscale_upscaling(self, small_sample_image):
        """Test upscaling grayscale image."""
        gray = small_sample_image.convert('L')
        result, _ = upscale_image(gray, 2)
        expected = (gray.size[0] * 2, gray.size[1] * 2)
        assert result.size == expected
        assert result.mode == 'RGB'

    def test_rgba_upscaling(self, small_sample_image):
        """Test upscaling RGBA image."""
        rgba = small_sample_image.convert('RGBA')
        result, _ = upscale_image(rgba, 4)
        expected = (rgba.size[0] * 4, rgba.size[1] * 4)
        assert result.size == expected
        assert result.mode == 'RGBA'

    def test_with_office_png(self, sample_image):
        """Test upscaling with office.png test image."""
        result_image, debug = upscale_image(sample_image, 2)
        assert result_image.size[0] == sample_image.size[0] * 2
        assert result_image.size[1] == sample_image.size[1] * 2

    @pytest.mark.skip(reason="Slow on CPU - cat.jpg is 2189x3890")
    def test_with_cat_jpg(self, cat_jpg_image):
        """Test upscaling with cat.jpg test image."""
        result_image, debug = upscale_image(cat_jpg_image, 2)
        assert result_image.size[0] == cat_jpg_image.size[0] * 2
        assert result_image.size[1] == cat_jpg_image.size[1] * 2

    @pytest.mark.skip(reason="Slow on CPU - cat.heif is 2189x3890, 4x upscale takes too long")
    def test_with_cat_heif(self, cat_heif_image):
        """Test upscaling with cat.heif test image."""
        result_image, debug = upscale_image(cat_heif_image, 4)
        assert result_image.size[0] == cat_heif_image.size[0] * 4
        assert result_image.size[1] == cat_heif_image.size[1] * 4

    def test_scale_factor_2(self, small_sample_image):
        """Test upscaling with scale factor 2."""
        result_image, _ = upscale_image(small_sample_image, 2)
        expected_size = (small_sample_image.size[0] * 2, small_sample_image.size[1] * 2)
        assert result_image.size == expected_size

    def test_scale_factor_4(self, small_sample_image):
        """Test upscaling with scale factor 4."""
        result_image, _ = upscale_image(small_sample_image, 4)
        expected_size = (small_sample_image.size[0] * 4, small_sample_image.size[1] * 4)
        assert result_image.size == expected_size
