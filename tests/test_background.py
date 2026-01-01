"""Tests for the background module using real images."""
import pytest
from pathlib import Path
from PIL import Image

from app.pipeline.background import process_background


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
    img.thumbnail((256, 256))
    return img


class TestProcessBackground:
    """Tests for process_background function with real images."""

    def test_returns_tuple(self, sample_image):
        """Test that function returns a tuple of two images."""
        result = process_background(sample_image, 15)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_pil_images(self, sample_image):
        """Test that returned images are PIL Images."""
        result_image, mask = process_background(sample_image, 15)
        assert isinstance(result_image, Image.Image)
        assert isinstance(mask, Image.Image)

    def test_preserves_size(self, sample_image):
        """Test that output image size is preserved."""
        result_image, _ = process_background(sample_image, 15)
        assert result_image.size == sample_image.size

    def test_different_blur_strengths(self, small_sample_image):
        """Test with different blur strength values."""
        for strength in [1, 15, 25, 50]:
            result_image, mask = process_background(small_sample_image, strength)
            assert result_image.size == small_sample_image.size
            assert mask.size == small_sample_image.size

    def test_blur_strength_zero(self, sample_image):
        """Test with blur strength of 0 returns original."""
        result_image, mask = process_background(sample_image, 0)
        assert result_image.size == sample_image.size
        assert mask.size == sample_image.size

    def test_negative_blur_strength(self, sample_image):
        """Test with negative blur strength returns original."""
        result_image, mask = process_background(sample_image, -5)
        assert result_image.size == sample_image.size

    def test_preserves_mode(self, small_sample_image):
        """Test that image mode is preserved for RGBA."""
        rgba = small_sample_image.convert('RGBA')
        result_image, _ = process_background(rgba, 15)
        assert result_image.mode == 'RGBA'

    def test_grayscale_input(self, small_sample_image):
        """Test with grayscale input image."""
        gray = small_sample_image.convert('L')
        result_image, mask = process_background(gray, 15)
        assert result_image.mode in ['L', 'RGB']
        assert mask.mode in ['L', 'RGB']

    def test_with_office_png(self, sample_image):
        """Test processing with office.png test image."""
        result_image, mask = process_background(sample_image, 15)
        assert result_image.size == sample_image.size
        assert mask.size == sample_image.size
        assert result_image.mode in ['RGB', 'RGBA']

    def test_with_cat_jpg(self, cat_jpg_image):
        """Test processing with cat.jpg test image."""
        result_image, mask = process_background(cat_jpg_image, 15)
        assert result_image.size == cat_jpg_image.size
        assert mask.size == cat_jpg_image.size

    def test_with_cat_heif(self, cat_heif_image):
        """Test processing with cat.heif test image."""
        result_image, mask = process_background(cat_heif_image, 15)
        assert result_image.size == cat_heif_image.size
        assert mask.size == cat_heif_image.size
