#!/usr/bin/env python3
"""Unit tests for ID Photo Booth application."""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Import the module to test
import id_photo_booth


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that valid config passes validation."""
        valid_config = {
            "output": {"width": 300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
            "display": {"width": 540, "height": 720, "fullscreen": True},
            "camera": {"device_index": 0, "width": 1280, "height": 720, "fps": 30, "max_failed_frames": 10},
            "ui": {"title": "Staff ID Photo", "subtitle": "Position your head within the outline"},
            "background_options": {"rembg": True, "bria": True, "birefnet": True},
            "birefnet": {"processing_width": 288, "processing_height": 384}
        }
        # Should not raise any exception
        id_photo_booth.validate_config(valid_config)

    def test_missing_section(self):
        """Test that missing config section raises ValueError."""
        invalid_config = {
            "output": {"width": 300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
            # Missing display section
        }
        with self.assertRaises(ValueError) as context:
            id_photo_booth.validate_config(invalid_config)
        self.assertIn("Missing required config section", str(context.exception))

    def test_invalid_output_dimensions(self):
        """Test that invalid output dimensions raise ValueError."""
        invalid_config = {
            "output": {"width": -300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
            "display": {"width": 540, "height": 720, "fullscreen": True},
            "camera": {"device_index": 0, "width": 1280, "height": 720, "fps": 30, "max_failed_frames": 10},
            "ui": {"title": "Staff ID Photo", "subtitle": "Position your head within the outline"},
            "background_options": {"rembg": True, "bria": True, "birefnet": True},
            "birefnet": {"processing_width": 288, "processing_height": 384}
        }
        with self.assertRaises(ValueError) as context:
            id_photo_booth.validate_config(invalid_config)
        self.assertIn("Output dimensions must be positive", str(context.exception))

    def test_invalid_jpeg_quality(self):
        """Test that invalid JPEG quality raises ValueError."""
        invalid_config = {
            "output": {"width": 300, "height": 400, "jpeg_quality": 150, "directory": "id_photos"},
            "display": {"width": 540, "height": 720, "fullscreen": True},
            "camera": {"device_index": 0, "width": 1280, "height": 720, "fps": 30, "max_failed_frames": 10},
            "ui": {"title": "Staff ID Photo", "subtitle": "Position your head within the outline"},
            "background_options": {"rembg": True, "bria": True, "birefnet": True},
            "birefnet": {"processing_width": 288, "processing_height": 384}
        }
        with self.assertRaises(ValueError) as context:
            id_photo_booth.validate_config(invalid_config)
        self.assertIn("JPEG quality must be between 1 and 100", str(context.exception))


class TestLoadConfig(unittest.TestCase):
    """Test configuration loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_file = id_photo_booth.__file__
        # Mock the __file__ attribute
        id_photo_booth.__file__ = str(Path(self.test_dir) / "id_photo_booth.py")

    def tearDown(self):
        """Clean up test fixtures."""
        id_photo_booth.__file__ = self.original_file
        shutil.rmtree(self.test_dir)

    def test_load_defaults_when_no_config(self):
        """Test that defaults are loaded when config file doesn't exist."""
        config = id_photo_booth.load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("output", config)
        self.assertIn("display", config)
        self.assertIn("camera", config)

    def test_load_valid_config_file(self):
        """Test loading a valid config file."""
        valid_config = {
            "output": {"width": 300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
            "display": {"width": 540, "height": 720, "fullscreen": False},
            "camera": {"device_index": 0, "width": 1280, "height": 720, "fps": 30, "max_failed_frames": 10},
            "ui": {"title": "Test Photo", "subtitle": "Test subtitle"},
            "background_options": {"rembg": True, "bria": False, "birefnet": True},
            "birefnet": {"processing_width": 288, "processing_height": 384}
        }
        config_path = Path(self.test_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(valid_config, f)

        config = id_photo_booth.load_config()
        self.assertEqual(config["ui"]["title"], "Test Photo")
        self.assertEqual(config["display"]["fullscreen"], False)


class TestCameraStream(unittest.TestCase):
    """Test CameraStream class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "camera": {
                "device_index": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            }
        }

    @patch('id_photo_booth.cv2.VideoCapture')
    def test_camera_stream_initialization(self, mock_video_capture):
        """Test camera stream initialization."""
        stream = id_photo_booth.CameraStream(self.config)
        self.assertIsNotNone(stream)
        self.assertEqual(stream.running, False)
        self.assertIsNone(stream.frame)

    @patch('id_photo_booth.cv2.VideoCapture')
    def test_camera_stream_start_success(self, mock_video_capture):
        """Test successful camera stream start."""
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = True
        mock_video_capture.return_value = mock_camera

        stream = id_photo_booth.CameraStream(self.config)
        result = stream.start()

        self.assertTrue(result)
        self.assertTrue(stream.running)
        mock_camera.set.assert_called()

    @patch('id_photo_booth.cv2.VideoCapture')
    def test_camera_stream_start_failure(self, mock_video_capture):
        """Test camera stream start failure."""
        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = False
        mock_video_capture.return_value = mock_camera

        stream = id_photo_booth.CameraStream(self.config)
        result = stream.start()

        self.assertFalse(result)


class TestConstants(unittest.TestCase):
    """Test that constants are defined correctly."""

    def test_aspect_ratio(self):
        """Test that aspect ratio is 3:4."""
        self.assertAlmostEqual(id_photo_booth.ASPECT_RATIO, 0.75, places=2)

    def test_file_names(self):
        """Test that file name constants are strings."""
        self.assertIsInstance(id_photo_booth.HEAD_OUTLINE_FILE, str)
        self.assertIsInstance(id_photo_booth.SHUTTER_SOUND_FILE, str)
        self.assertIsInstance(id_photo_booth.CONFIG_FILE, str)


if __name__ == '__main__':
    unittest.main()
