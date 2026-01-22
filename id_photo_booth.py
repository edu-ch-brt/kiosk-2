#!/usr/bin/env python3
"""Staff ID Photo Booth Application."""

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import messagebox, ttk  # Added ttk for Progressbar
import threading
import io
import re
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple

# Constants
ASPECT_RATIO = 3 / 4  # Portrait 3:4 aspect ratio
VIDEO_UPDATE_INTERVAL_MS = 66  # ~15 FPS (optimized from 30 FPS for lower CPU usage)
LOADING_CHECK_INTERVAL_MS = 100  # Check model loading progress
FRAME_SLEEP_SECONDS = 0.01  # Camera stream sleep interval (increased for lower CPU)
PREVIEW_WINDOW_WIDTH = 400
PREVIEW_WINDOW_HEIGHT = 600
PREVIEW_IMAGE_WIDTH = 300
PREVIEW_IMAGE_HEIGHT = 400
HEAD_OUTLINE_FILE = "head_outline.png"
SHUTTER_SOUND_FILE = "shutter_sound.wav"
CONFIG_FILE = "config.json"
LOG_FILE = "id_photo_booth.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import rembg if available
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# Optional: pygame for sound
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# For BiRefNet: torch and transformers
try:
    import torch
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    BIREFNET_AVAILABLE = True
except ImportError:
    BIREFNET_AVAILABLE = False
    logger.warning("BiRefNet-Portrait not available. Only rembg will be used.")
    logger.info("Install with: pip install torch torchvision transformers")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate config structure and values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["output", "display", "camera", "ui", "background_options", "birefnet"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate output section
    try:
        output_width = config["output"]["width"]
        output_height = config["output"]["height"]
        jpeg_quality = config["output"]["jpeg_quality"]
        output_directory = config["output"]["directory"]
    except KeyError as e:
        raise ValueError(f"Missing required field in 'output' section: {e.args[0]}") from e
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Output dimensions must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")

    # Validate display section
    try:
        display_width = config["display"]["width"]
        display_height = config["display"]["height"]
        display_fullscreen = config["display"]["fullscreen"]
    except KeyError as e:
        raise ValueError(f"Missing required field in 'display' section: {e.args[0]}") from e
    if display_width <= 0 or display_height <= 0:
        raise ValueError("Display dimensions must be positive")

    # Validate camera section
    try:
        camera_device_index = config["camera"]["device_index"]
        camera_width = config["camera"]["width"]
        camera_height = config["camera"]["height"]
        camera_fps = config["camera"]["fps"]
        # Optional fields with defaults
        camera_preview_width = config["camera"].get("preview_width", camera_width)
        camera_preview_height = config["camera"].get("preview_height", camera_height)
    except KeyError as e:
        raise ValueError(f"Missing required field in 'camera' section: {e.args[0]}") from e
    if camera_device_index < 0:
        raise ValueError("Camera device index cannot be negative")
    if camera_width <= 0 or camera_height <= 0:
        raise ValueError("Camera dimensions must be positive")
    if camera_preview_width <= 0 or camera_preview_height <= 0:
        raise ValueError("Camera preview dimensions must be positive")
    if camera_fps <= 0:
        raise ValueError("Camera FPS must be positive")

    # Validate UI section
    try:
        _ = config["ui"]["title"]
        _ = config["ui"]["subtitle"]
    except KeyError as e:
        raise ValueError(f"Missing required field in 'ui' section: {e.args[0]}") from e

    # Validate background_options section
    try:
        _ = config["background_options"]["rembg"]
        _ = config["background_options"]["bria"]
        _ = config["background_options"]["birefnet"]
    except KeyError as e:
        raise ValueError(f"Missing required field in 'background_options' section: {e.args[0]}") from e

    # Validate BiRefNet section
    try:
        birefnet_processing_width = config["birefnet"]["processing_width"]
        birefnet_processing_height = config["birefnet"]["processing_height"]
        # Optional field with default
        birefnet_use_gpu = config["birefnet"].get("use_gpu", True)
    except KeyError as e:
        raise ValueError(f"Missing required field in 'birefnet' section: {e.args[0]}") from e
    if birefnet_processing_width <= 0 or birefnet_processing_height <= 0:
        raise ValueError("BiRefNet processing dimensions must be positive")

    logger.info("Config validation passed")


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file or return defaults.

    Returns:
        Dictionary containing application configuration with sections for
        output, display, camera, UI, background options, and BiRefNet settings.
    """
    config_path = Path(__file__).parent / CONFIG_FILE
    defaults: Dict[str, Any] = {
        "output": {"width": 300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
        "display": {"width": 540, "height": 720, "fullscreen": True},
        "camera": {
            "device_index": 0,
            "width": 1280,
            "height": 720,
            "fps": 15,  # Optimized: Reduced from 30 to 15 FPS for lower CPU usage
            "max_failed_frames": 10,
            "preview_width": 640,  # Lower resolution for preview to reduce CPU load
            "preview_height": 480
        },
        "ui": {"title": "Staff ID Photo", "subtitle": "Position your head within the outline"},
        "background_options": {"rembg": True, "bria": True, "birefnet": True},
        "birefnet": {
            "processing_width": 192,  # Optimized: Reduced from 288 for faster processing
            "processing_height": 256,  # Optimized: Reduced from 384 for faster processing
            "use_gpu": True  # Enable GPU acceleration if available
        }
    }
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                validate_config(config)
                return config
        except ValueError as e:
            logger.error(f"Config validation failed: {e}. Using defaults.")
            return defaults
        except Exception as e:
            logger.warning(f"Invalid config.json: {e}. Using defaults.")
            return defaults
    logger.info("No config.json found, using defaults")
    return defaults

class CameraStream:
    """Threaded camera stream handler for continuous frame capture.

    This class manages camera access in a separate thread to prevent blocking
    the main UI thread. It continuously reads frames from the camera and makes
    them available through a thread-safe read() method.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the camera stream.

        Args:
            config: Application configuration dictionary containing camera settings.
        """
        self.camera: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.lock: threading.Lock = threading.Lock()
        self.failed_frame_count: int = 0
        self.config: Dict[str, Any] = config

    def start(self) -> bool:
        """Start camera capture and background streaming thread.

        Returns:
            True if camera started successfully, False otherwise.
        """
        try:
            self.camera = cv2.VideoCapture(self.config["camera"]["device_index"])
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera at index {self.config['camera']['device_index']}")
                return False
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
            self.camera.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
            self.running = True
            threading.Thread(target=self._stream, daemon=True).start()
            logger.info(f"Camera started successfully at index {self.config['camera']['device_index']}")
            return True
        except Exception as e:
            logger.error(f"Exception starting camera: {e}")
            return False

    def _stream(self) -> None:
        """Read frames in background loop."""
        while self.running:
            ret, frame = self.camera.read()
            with self.lock:
                if ret:
                    self.frame = frame
                    self.failed_frame_count = 0
                else:
                    self.failed_frame_count += 1
            time.sleep(FRAME_SLEEP_SECONDS)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest camera frame in a thread-safe manner.

        Returns:
            Tuple of (success, frame) where success is True if a frame is available,
            and frame is the numpy array containing the image data or None.
        """
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        """Stop stream and release camera resources."""
        self.running = False
        if self.camera:
            try:
                self.camera.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")

class IDPhotoBooth:
    """Main photo booth application managing UI, camera, and background removal.

    This class coordinates all aspects of the photo booth including camera
    capture, UI display, background removal model loading, and photo saving.
    """
    def __init__(self) -> None:
        """Initialize the photo booth application and set up all components."""
        self.config: Dict[str, Any] = load_config()
        self.root: tk.Tk = tk.Tk()
        self.root.title(self.config["ui"]["title"])
        display_w: int = self.config["display"]["width"]
        display_h: int = self.config["display"]["height"]
        self.root.geometry(f"{display_w}x{display_h}")
        if self.config["display"]["fullscreen"]:
            self.root.attributes('-fullscreen', True)
        self.camera_stream: CameraStream = CameraStream(self.config)
        self.running: bool = False
        self.video_after_id: Optional[str] = None
        self.head_outline: Optional[Image.Image] = self._load_head_outline(display_w, display_h)
        self.shutter_sound: Optional[Any] = self._load_shutter_sound()
        self.rembg_session: Optional[Any] = None
        self.bria_session: Optional[Any] = None
        self.birefnet_model: Optional[Any] = None
        self.birefnet_transform: Optional[Any] = None  # Cache transform for performance
        self.device: str = "cpu"  # Will be set to cuda if available
        self.output_dir: Path = Path(self.config["output"]["directory"])
        self.output_dir.mkdir(exist_ok=True)
        # Use preview resolution if specified, otherwise use full camera resolution
        self.preview_width: int = self.config["camera"].get("preview_width", self.config["camera"]["width"])
        self.preview_height: int = self.config["camera"].get("preview_height", self.config["camera"]["height"])
        self._setup_ui()
        self.root.bind('<space>', lambda e: self.capture_photo())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        self.model_loading_thread: Optional[threading.Thread] = None  # Track loading thread

    def _load_head_outline(self, w: int, h: int) -> Optional[Image.Image]:
        """Load head outline guide image and prepare for overlay.

        Converts white background to transparent and resizes to display dimensions.

        Args:
            w: Target width in pixels.
            h: Target height in pixels.

        Returns:
            PIL Image with transparency or None if file not found.
        """
        path = Path(__file__).parent / HEAD_OUTLINE_FILE
        if path.exists():
            img = Image.open(path).convert('RGB')
            # Convert to grayscale, invert for alpha (black outline opaque, white transparent)
            gray = ImageOps.grayscale(img)
            alpha = Image.eval(gray, lambda x: 255 - x)
            img.putalpha(alpha)
            return img.resize((w, h), Image.LANCZOS)
        logger.warning("head_outline.png not found.")
        return None

    def _load_shutter_sound(self) -> Optional[Any]:
        """Load optional shutter sound if pygame available."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                path = Path(__file__).parent / SHUTTER_SOUND_FILE
                if path.exists():
                    return pygame.mixer.Sound(str(path))
            except Exception as e:
                logger.warning(f"Sound load failed: {e}")
        return None

    def _setup_ui(self) -> None:
        """Setup main UI elements: labels, canvas, status, and hidden loading frame."""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(self.main_frame, text=self.config["ui"]["title"], font=('Helvetica', 24, 'bold')).pack(pady=10)
        tk.Label(self.main_frame, text=self.config["ui"]["subtitle"], font=('Helvetica', 12)).pack(pady=5)

        # Canvas container for centering
        canvas_container = tk.Frame(self.main_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        # Canvas with fixed size to maintain aspect ratio
        self.canvas = tk.Canvas(
            canvas_container,
            width=self.config["display"]["width"],
            height=self.config["display"]["height"],
            highlightthickness=0
        )
        self.canvas.pack(anchor=tk.CENTER, expand=True)

        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(self.main_frame, textvariable=self.status_var, font=('Helvetica', 12)).pack(pady=10)

        # Add hidden loading frame with progress bar
        self.loading_frame = tk.Frame(self.main_frame)
        self.progress_bar = ttk.Progressbar(self.loading_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(pady=10)
        tk.Label(self.loading_frame, text="Loading background tools...").pack()
        self.loading_frame.pack_forget()  # Hide initially

    def start_camera(self) -> bool:
        """Start camera stream and initialize video preview loop.

        Also triggers asynchronous loading of background removal models.

        Returns:
            True if camera started successfully, False otherwise.
        """
        if self.camera_stream.start():
            self.running = True
            self._update_video()
            self._load_models_async()
            return True
        self.status_var.set("Error: Camera not found")
        return False

    def _load_models_async(self) -> None:
        """Load enabled background removal models in a background thread."""
        # Show progress bar if any models to load
        if REMBG_AVAILABLE and (self.config["background_options"]["rembg"] or self.config["background_options"]["bria"]) or (BIREFNET_AVAILABLE and self.config["background_options"]["birefnet"]):
            self.loading_frame.pack(pady=10)
            self.progress_bar.start()
            self.status_var.set("Loading tools...")
        
        def load():
            if REMBG_AVAILABLE and self.config["background_options"]["rembg"]:
                try:
                    logger.info("Loading rembg u2net model...")
                    self.rembg_session = new_session(model_name="u2net")
                    logger.info("rembg u2net model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load rembg u2net model: {e}")

            if REMBG_AVAILABLE and self.config["background_options"]["bria"]:
                try:
                    logger.info("Loading BRIA model...")
                    self.bria_session = new_session(model_name="bria-rmbg")
                    logger.info("BRIA model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load BRIA model: {e}")

            if BIREFNET_AVAILABLE and self.config["background_options"]["birefnet"]:
                try:
                    logger.info("Loading BiRefNet-portrait model...")
                    self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                        "ZhengPeng7/BiRefNet-portrait", trust_remote_code=True
                    )
                    self.birefnet_model.eval()

                    # GPU optimization: Move model to GPU if available and configured
                    if self.config["birefnet"].get("use_gpu", True) and torch.cuda.is_available():
                        self.device = "cuda"
                        self.birefnet_model = self.birefnet_model.to(self.device)
                        logger.info("BiRefNet-portrait model moved to GPU")
                    else:
                        self.device = "cpu"
                        logger.info("BiRefNet-portrait model using CPU")

                    # Pre-cache the transform for performance
                    image_size = (
                        self.config["birefnet"]["processing_height"],
                        self.config["birefnet"]["processing_width"]
                    )
                    self.birefnet_transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                    logger.info("BiRefNet-portrait model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load BiRefNet-portrait model: {e}")
        
        self.model_loading_thread = threading.Thread(target=load, daemon=True)
        self.model_loading_thread.start()
        
        # Check periodically if loading done
        self._check_loading_complete()

    def _check_loading_complete(self) -> None:
        """Check if model loading thread finished; hide bar and update status."""
        if self.model_loading_thread.is_alive():
            self.root.after(LOADING_CHECK_INTERVAL_MS, self._check_loading_complete)
        else:
            self.progress_bar.stop()
            self.loading_frame.pack_forget()
            self.status_var.set("Ready - Position yourself")

    def _update_video(self) -> None:
        """Update canvas with latest frame, cropped to 3:4 and overlaid with outline."""
        if not self.running:
            return
        ret, frame = self.camera_stream.read()
        if ret:
            # Optimization: Resize to preview resolution first to reduce processing
            # Use INTER_AREA for downscaling - fastest and best quality for shrinking
            if frame.shape[1] > self.preview_width or frame.shape[0] > self.preview_height:
                frame = cv2.resize(frame, (self.preview_width, self.preview_height),
                                 interpolation=cv2.INTER_AREA)

            # Crop frame to 3:4 aspect ratio by centering the shorter dimension
            h, w = frame.shape[:2]
            target_ratio = ASPECT_RATIO
            current_ratio = w / h
            if current_ratio > target_ratio:
                crop_w = int(h * target_ratio)
                start_x = (w - crop_w) // 2
                frame = frame[:, start_x:start_x + crop_w]
            else:
                crop_h = int(w / target_ratio)
                start_y = (h - crop_h) // 2
                frame = frame[start_y:start_y + crop_h, :]

            # Resize to display size and convert for PIL compositing
            # Use INTER_AREA for downscaling, INTER_LINEAR for upscaling
            interpolation = cv2.INTER_AREA if (frame.shape[1] > self.config["display"]["width"]) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (self.config["display"]["width"], self.config["display"]["height"]),
                             interpolation=interpolation)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame)
            if self.head_outline:
                img = Image.alpha_composite(img.convert('RGBA'), self.head_outline)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.video_after_id = self.root.after(VIDEO_UPDATE_INTERVAL_MS, self._update_video)  # Target ~15 FPS

    def capture_photo(self) -> None:
        """Capture current camera frame and initiate processing.

        Triggered by spacebar press. Plays shutter sound if available and
        processes the captured frame in a background thread.
        """
        if not self.running:
            return
        ret, frame = self.camera_stream.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture.")
            return
        if self.shutter_sound:
            self.shutter_sound.play()
        self.status_var.set("Processing...")
        threading.Thread(target=self._process_capture, args=(frame,), daemon=True).start()

    def _process_capture(self, frame: np.ndarray) -> None:
        """Process captured frame: crop/resize to output size, generate background removal versions."""
        # Crop to 3:4 and resize to final output dimensions
        h, w = frame.shape[:2]
        target_ratio = ASPECT_RATIO
        current_ratio = w / h
        if current_ratio > target_ratio:
            crop_w = int(h * target_ratio)
            start_x = (w - crop_w) // 2
            frame = frame[:, start_x:start_x + crop_w]
        else:
            crop_h = int(w / target_ratio)
            start_y = (h - crop_h) // 2
            frame = frame[start_y:start_y + crop_h, :]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
            (self.config["output"]["width"], self.config["output"]["height"]), Image.LANCZOS
        )
        # Generate versions: original always, others if enabled and sessions/models loaded
        version_original = pil_img
        version_rembg = None
        version_bria = None
        version_birefnet = None
        if REMBG_AVAILABLE and self.rembg_session:
            try:
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                result = remove(img_bytes.read(), session=self.rembg_session)
                result_img = Image.open(io.BytesIO(result)).convert('RGBA')
                white_bg = Image.new('RGBA', result_img.size, (255, 255, 255, 255))
                version_rembg = Image.alpha_composite(white_bg, result_img).convert('RGB')
            except Exception as e:
                logger.error(f"rembg failed: {e}")
        if REMBG_AVAILABLE and self.bria_session:
            try:
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                result = remove(img_bytes.read(), session=self.bria_session)
                result_img = Image.open(io.BytesIO(result)).convert('RGBA')
                white_bg = Image.new('RGBA', result_img.size, (255, 255, 255, 255))
                version_bria = Image.alpha_composite(white_bg, result_img).convert('RGB')
            except Exception as e:
                logger.error(f"BRIA failed: {e}")
        if BIREFNET_AVAILABLE and self.birefnet_model and self.birefnet_transform:
            try:
                # Preprocess: Use cached transform
                input_tensor = self.birefnet_transform(pil_img).unsqueeze(0)

                # Move to GPU if available
                if self.device == "cuda":
                    input_tensor = input_tensor.to(self.device)

                # Inference with optimizations
                with torch.no_grad():
                    # Use half precision on GPU for faster inference (if supported)
                    if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
                        with torch.cuda.amp.autocast():
                            preds = self.birefnet_model(input_tensor)[-1].sigmoid()
                    else:
                        preds = self.birefnet_model(input_tensor)[-1].sigmoid()

                # Move back to CPU for post-processing
                if self.device == "cuda":
                    preds = preds.cpu()

                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                # Use faster BILINEAR instead of LANCZOS for mask resizing
                mask = pred_pil.resize(pil_img.size, Image.BILINEAR)

                # Postprocess: Apply mask to image on white bg
                result_img = Image.new('RGBA', pil_img.size)
                result_img.paste(pil_img, mask=mask.convert('L'))
                white_bg = Image.new('RGBA', result_img.size, (255, 255, 255, 255))
                version_birefnet = Image.alpha_composite(white_bg, result_img).convert('RGB')
            except Exception as e:
                logger.error(f"BiRefNet failed: {e}")
        # Schedule preview on main thread
        self.root.after(0, lambda: self._show_preview(version_original, version_rembg, version_bria, version_birefnet))

    def _show_preview(self, v_original: Image.Image, v_rembg: Optional[Image.Image],
                      v_bria: Optional[Image.Image], v_birefnet: Optional[Image.Image]) -> None:
        """Show modal preview window with radio options, image display, filename entry, and buttons."""
        self.status_var.set("Previewing...")
        preview = tk.Toplevel(self.root)
        preview.title("Photo Preview")
        # Center window on screen for better usability
        w, h = PREVIEW_WINDOW_WIDTH, PREVIEW_WINDOW_HEIGHT
        sx, sy = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        preview.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        preview.transient(self.root)  # Tie to main window
        preview.grab_set()  # Make modal
        tk.Label(preview, text="Select Background Removal", font=('Helvetica', 14)).pack(pady=10)
        option_var = tk.StringVar(value='rembg' if v_rembg else 'original')
        img_label = tk.Label(preview)
        img_label.pack(pady=10)

        def update_img():
            """Update displayed image based on selected radio option."""
            selected = {'original': v_original, 'rembg': v_rembg, 'bria': v_bria, 'birefnet': v_birefnet}.get(option_var.get())
            if selected:
                photo = ImageTk.PhotoImage(selected.resize((PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT), Image.LANCZOS))
                img_label.config(image=photo)
                img_label.image = photo  # Retain reference to prevent GC

        tk.Radiobutton(preview, text="Original (no removal)", variable=option_var, value='original', command=update_img).pack(anchor='w')
        if self.config["background_options"]["rembg"]:
            rembg_radio = tk.Radiobutton(preview, text="Basic rembg", variable=option_var, value='rembg', command=update_img)
            rembg_radio.pack(anchor='w')
            if not v_rembg:
                rembg_radio.config(state=tk.DISABLED)
        if self.config["background_options"]["bria"]:
            bria_radio = tk.Radiobutton(preview, text="Accurate BRIA", variable=option_var, value='bria', command=update_img)
            bria_radio.pack(anchor='w')
            if not v_bria:
                bria_radio.config(state=tk.DISABLED)
        if self.config["background_options"]["birefnet"]:
            birefnet_radio = tk.Radiobutton(preview, text="BiRefNet-Portrait", variable=option_var, value='birefnet', command=update_img)
            birefnet_radio.pack(anchor='w')
            if not v_birefnet:
                birefnet_radio.config(state=tk.DISABLED)
        update_img()  # Initial image display
        tk.Label(preview, text="Filename:").pack(pady=5)
        filename_var = tk.StringVar()
        entry = tk.Entry(preview, textvariable=filename_var)
        entry.pack()
        entry.focus()  # Auto-focus for quick entry

        def retake():
            """Close preview and reset status for new capture."""
            preview.destroy()
            self.status_var.set("Ready for retake")

        def save():
            """Save selected version with sanitized filename, handle overwrites."""
            name = re.sub(r'[<>:"/\\|?*]', '', filename_var.get().strip())
            if not name:
                messagebox.showerror("Error", "Enter filename.")
                return
            selected = {'original': v_original, 'rembg': v_rembg, 'bria': v_bria, 'birefnet': v_birefnet}.get(option_var.get())
            if not selected:
                messagebox.showerror("Error", "No version.")
                return
            path = self.output_dir / f"{name}.jpg"
            if path.exists():
                if not messagebox.askyesno("Overwrite?", "File exists. Overwrite?"):
                    return
            try:
                selected.save(path, 'JPEG', quality=self.config["output"]["jpeg_quality"])
                logger.info(f"Saved photo: {path}")
                preview.destroy()
                self.status_var.set(f"Saved: {name}.jpg")
            except Exception as e:
                logger.error(f"Failed to save photo {path}: {e}")
                messagebox.showerror("Error", f"Failed to save photo: {e}")

        tk.Button(preview, text="Retake", command=retake).pack(side='left', padx=10, pady=10)
        tk.Button(preview, text="Save", command=save).pack(side='right', padx=10, pady=10)
        entry.bind('<Return>', lambda e: save())  # Enter key saves

    def quit_app(self) -> None:
        """Graceful shutdown: stop loops, release resources."""
        self.running = False
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
        self.camera_stream.stop()
        self.root.destroy()

    def run(self) -> None:
        """Run the app: start camera if possible, enter mainloop."""
        if self.start_camera():
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            self.root.mainloop()
        else:
            self.root.destroy()

if __name__ == "__main__":
    app = IDPhotoBooth()
    app.run()