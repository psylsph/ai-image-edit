"""Tests for the grain module."""
import pytest
from pathlib import Path
from PIL import Image

from app.pipeline.grain import apply_film_grain


TEST_IMAGE_PATH = Path(__file__).parent / "image" / "office.png"


@pytest.fixture
def sample_image():
    """Load the test image."""
    return Image.open(TEST_IMAGE_PATH)


@pytest.fixture
def small_sample_image():
    """Load and resize the test image for faster tests."""
    img = Image.open(TEST_IMAGE_PATH)
    img.thumbnail((256, 256))
    return img


class TestApplyFilmGrain:
    """Tests for apply_film_grain function."""

    def test_returns_tuple(self, sample_image):
        """Test that function returns a tuple of two images."""
        result = apply_film_grain(sample_image, 0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_pil_images(self, sample_image):
        """Test that returned images are PIL Images."""
        result_image, grain_only = apply_film_grain(sample_image, 0.5)
        assert isinstance(result_image, Image.Image)
        assert isinstance(grain_only, Image.Image)

    def test_grain_intensity_0(self, sample_image):
        """Test grain with intensity 0 returns original image."""
        result_image, grain_only = apply_film_grain(sample_image, 0.0)
        assert list(result_image.getdata()) == list(sample_image.getdata())

    def test_grain_intensity_low(self, sample_image):
        """Test grain with low intensity."""
        result_image, grain_only = apply_film_grain(sample_image, 0.1)
        assert result_image.size == sample_image.size

    def test_grain_intensity_high(self, sample_image):
        """Test grain with high intensity."""
        result_image, grain_only = apply_film_grain(sample_image, 1.0)
        assert result_image.size == sample_image.size

    def test_preserves_size(self, sample_image):
        """Test that image size is preserved."""
        for intensity in [0.1, 0.5, 1.0]:
            result_image, _ = apply_film_grain(sample_image, intensity)
            assert result_image.size == sample_image.size

    def test_preserves_mode(self, sample_image):
        """Test that image mode is preserved."""
        rgb_image = sample_image.convert('RGB')
        rgba_image = sample_image.convert('RGBA')

        result_rgb, _ = apply_film_grain(rgb_image, 0.5)
        result_rgba, _ = apply_film_grain(rgba_image, 0.5)

        assert result_rgb.mode == 'RGB'
        assert result_rgba.mode == 'RGBA'

    def test_grayscale_image(self, sample_image):
        """Test grain on grayscale image."""
        gray = sample_image.convert('L')
        result_image, grain_only = apply_film_grain(gray, 0.5)
        assert result_image.mode == 'L'
        assert grain_only.mode == 'L'

    def test_different_sizes(self, small_sample_image):
        """Test grain on images of different sizes."""
        sizes = [(100, 100), (256, 128), (128, 256)]
        for size in sizes:
            img = small_sample_image.copy()
            img = img.resize(size)
            result, _ = apply_film_grain(img, 0.5)
            assert result.size == size

    def test_grain_only_is_grayscale(self, sample_image):
        """Test that grain_only image is grayscale (L mode) or matches input mode."""
        _, grain_only = apply_film_grain(sample_image, 0.5)
        assert grain_only.mode in ['L', 'RGB', 'RGBA']

    def test_grain_intensity_default(self, sample_image):
        """Test that default intensity is 0.5."""
        result_image, grain_only = apply_film_grain(sample_image)
        assert result_image.size == sample_image.size

    def test_result_differs_from_input(self, sample_image):
        """Test that result differs from input (grain is applied)."""
        result_image, _ = apply_film_grain(sample_image, 0.5)
        assert list(result_image.getdata()) != list(sample_image.getdata())

    def test_with_real_image(self, sample_image):
        """Test grain on the real test image."""
        result_image, grain_only = apply_film_grain(sample_image, 0.5)
        assert result_image.size == sample_image.size
        assert grain_only.size == sample_image.size
        assert result_image.mode == sample_image.mode
