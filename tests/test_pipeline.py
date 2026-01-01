"""Tests for the pipeline module."""
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from app.pipeline import preprocess_image, process_pipeline, process_image_simple


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


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_no_resize_needed_small_image(self):
        """Test that small images are not resized."""
        img = Image.new('RGB', (500, 400), color='blue')
        result = preprocess_image(img, max_dimension=1536)
        assert result.size == img.size

    def test_resize_landscape_image(self):
        """Test resizing landscape orientation image."""
        img = Image.new('RGB', (800, 600), color='red')
        result = preprocess_image(img, max_dimension=500)
        assert max(result.size) == 500
        assert result.size[0] == 500
        assert result.size[1] == 375

    def test_resize_portrait_image(self):
        """Test resizing portrait orientation image."""
        portrait = Image.new('RGB', (400, 800), color='yellow')
        result = preprocess_image(portrait, max_dimension=500)
        assert max(result.size) == 500
        assert result.size[1] == 500
        assert result.size[0] == 250

    def test_resize_square_image(self):
        """Test resizing square image."""
        square = Image.new('RGB', (1024, 1024), color='green')
        result = preprocess_image(square, max_dimension=512)
        assert max(result.size) == 512
        assert result.size == (512, 512)

    def test_maintains_aspect_ratio(self):
        """Test that aspect ratio is preserved."""
        wide = Image.new('RGB', (1600, 900), color='cyan')
        result = preprocess_image(wide, max_dimension=800)
        assert result.size[0] == 800
        assert result.size[1] == 450
        assert result.size[0] / result.size[1] == wide.size[0] / wide.size[1]

    def test_default_max_dimension(self, small_sample_image):
        """Test default max_dimension of 1536."""
        result = preprocess_image(small_sample_image)
        assert max(result.size) == 1536 or max(result.size) == max(small_sample_image.size)

    def test_preserves_image_mode(self):
        """Test that image mode is preserved."""
        rgba = Image.new('RGBA', (800, 600), color=(255, 0, 0, 128))
        result = preprocess_image(rgba, max_dimension=500)
        assert result.mode == 'RGBA'

    def test_with_real_image(self, sample_image):
        """Test preprocessing with real image."""
        result = preprocess_image(sample_image, max_dimension=512)
        assert max(result.size) == 512
        assert result.mode == sample_image.mode


class TestProcessPipeline:
    """Tests for process_pipeline function."""

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_all_stages_enabled(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test pipeline with all stages enabled."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=True,
            blur_strength=15,
            enable_grain=True,
            grain_intensity=0.5,
            enable_upscale=True,
            upscale_factor=2,
            debug=True
        )

        assert 'final' in result
        mock_bg.assert_called_once()
        mock_grain.assert_called_once()
        mock_upscale.assert_called_once()

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_background_blur_disabled(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test pipeline with background blur disabled."""
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=False,
            enable_grain=True,
            grain_intensity=0.5,
            enable_upscale=True,
            upscale_factor=2,
            debug=True
        )

        mock_bg.assert_not_called()
        mock_grain.assert_called_once()

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_grain_disabled(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test pipeline with grain disabled."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=True,
            blur_strength=15,
            enable_grain=False,
            enable_upscale=True,
            upscale_factor=2,
            debug=True
        )

        mock_grain.assert_not_called()

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_upscale_disabled(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test pipeline with upscaling disabled."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=True,
            blur_strength=15,
            enable_grain=True,
            grain_intensity=0.5,
            enable_upscale=False,
            debug=True
        )

        mock_upscale.assert_not_called()

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_debug_mode_false(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test pipeline with debug=False excludes intermediate results."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=True,
            enable_grain=True,
            enable_upscale=True,
            debug=False
        )

        assert 'final' in result
        assert 'preprocessed' not in result
        assert 'bg_result' not in result

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_returns_dict(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test that pipeline returns a dictionary."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(sample_image)
        assert isinstance(result, dict)
        assert 'final' in result

    @patch('app.pipeline.process_background')
    @patch('app.pipeline.apply_film_grain')
    @patch('app.pipeline.upscale_image')
    def test_all_debug_keys_present(self, mock_upscale, mock_grain, mock_bg, sample_image):
        """Test that all debug keys are present when debug=True."""
        mock_bg.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_grain.return_value = (sample_image, Image.new('L', sample_image.size))
        mock_upscale.return_value = (sample_image, Image.new('L', sample_image.size))

        result = process_pipeline(
            sample_image,
            enable_background_blur=True,
            enable_grain=True,
            enable_upscale=True,
            debug=True
        )

        expected_keys = ['final', 'preprocessed', 'bg_result', 'bg_mask',
                         'grain_result', 'grain_only', 'upscale_result']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestProcessImageSimple:
    """Tests for process_image_simple function."""

    @patch('app.pipeline.process_pipeline')
    def test_returns_final_image_only(self, mock_pipeline, sample_image):
        """Test that simple interface returns only the final image."""
        mock_pipeline.return_value = {
            'final': sample_image,
            'preprocessed': sample_image,
            'bg_result': sample_image
        }

        result = process_image_simple(sample_image)
        assert result == sample_image
        mock_pipeline.assert_called_once()

    @patch('app.pipeline.process_pipeline')
    def test_debug_false_by_default(self, mock_pipeline, sample_image):
        """Test that debug is set to False in simple interface."""
        mock_pipeline.return_value = {'final': sample_image}

        process_image_simple(sample_image)
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs.get('debug') is False

    @patch('app.pipeline.process_pipeline')
    def test_passes_all_parameters(self, mock_pipeline, sample_image):
        """Test that all parameters are passed through."""
        mock_pipeline.return_value = {'final': sample_image}

        process_image_simple(
            sample_image,
            enable_background_blur=True,
            blur_strength=20,
            enable_grain=False,
            grain_intensity=0.3,
            enable_upscale=True,
            upscale_factor=4
        )

        mock_pipeline.assert_called_once()
