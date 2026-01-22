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
from pathlib import Path
import time

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
    print("Warning: BiRefNet-Portrait not available. Only rembg will be used.")
    print("Install with: pip install torch torchvision transformers")


def load_config():
    """Load config from json or defaults."""
    config_path = Path(__file__).parent / "config.json"
    defaults = {
        "output": {"width": 300, "height": 400, "jpeg_quality": 95, "directory": "id_photos"},
        "display": {"width": 540, "height": 720, "fullscreen": True},
        "camera": {"device_index": 0, "width": 1280, "height": 720, "fps": 30, "max_failed_frames": 10},
        "ui": {"title": "Staff ID Photo", "subtitle": "Position your head within the outline"},
        "background_options": {"rembg": True, "bria": True, "birefnet": True},
        "birefnet": {"processing_width": 288, "processing_height": 384}
    }
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Invalid config.json: {e}. Using defaults.")
    return defaults

class CameraStream:
    """Threaded camera stream handler."""
    def __init__(self, config):
        self.camera = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.failed_frame_count = 0
        self.config = config

    def start(self):
        """Start camera and thread."""
        self.camera = cv2.VideoCapture(self.config["camera"]["device_index"])
        if not self.camera.isOpened():
            return False
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
        self.camera.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
        self.running = True
        threading.Thread(target=self._stream, daemon=True).start()
        return True

    def _stream(self):
        """Read frames in background loop."""
        while self.running:
            ret, frame = self.camera.read()
            with self.lock:
                if ret:
                    self.frame = frame
                    self.failed_frame_count = 0
                else:
                    self.failed_frame_count += 1
            time.sleep(0.001)

    def read(self):
        """Get latest frame thread-safely."""
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stop stream and release camera resources."""
        self.running = False
        if self.camera:
            self.camera.release()

class IDPhotoBooth:
    """Main photo booth app class."""
    def __init__(self):
        self.config = load_config()
        self.root = tk.Tk()
        self.root.title(self.config["ui"]["title"])
        display_w, display_h = self.config["display"]["width"], self.config["display"]["height"]
        self.root.geometry(f"{display_w}x{display_h}")
        if self.config["display"]["fullscreen"]:
            self.root.attributes('-fullscreen', True)
        self.camera_stream = CameraStream(self.config)
        self.running = False
        self.video_after_id = None
        self.head_outline = self._load_head_outline(display_w, display_h)
        self.shutter_sound = self._load_shutter_sound()
        self.rembg_session = None
        self.bria_session = None
        self.birefnet_model = None
        self.output_dir = Path(self.config["output"]["directory"])
        self.output_dir.mkdir(exist_ok=True)
        self._setup_ui()
        self.root.bind('<space>', lambda e: self.capture_photo())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        self.model_loading_thread = None  # Track loading thread

    def _load_head_outline(self, w, h):
        """Load outline image and convert white background to transparent alpha."""
        path = Path(__file__).parent / "head_outline.png"
        if path.exists():
            img = Image.open(path).convert('RGB')
            # Convert to grayscale, invert for alpha (black outline opaque, white transparent)
            gray = ImageOps.grayscale(img)
            alpha = Image.eval(gray, lambda x: 255 - x)
            img.putalpha(alpha)
            return img.resize((w, h), Image.LANCZOS)
        print("Warning: head_outline.png not found.")
        return None

    def _load_shutter_sound(self):
        """Load optional shutter sound if pygame available."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                path = Path(__file__).parent / "shutter_sound.wav"
                if path.exists():
                    return pygame.mixer.Sound(str(path))
            except Exception as e:
                print(f"Warning: Sound load failed: {e}")
        return None

    def _setup_ui(self):
        """Setup main UI elements: labels, canvas, status, and hidden loading frame."""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.main_frame, text=self.config["ui"]["title"], font=('Helvetica', 24, 'bold')).pack(pady=10)
        tk.Label(self.main_frame, text=self.config["ui"]["subtitle"], font=('Helvetica', 12)).pack(pady=5)
        self.canvas = tk.Canvas(self.main_frame, width=self.config["display"]["width"], height=self.config["display"]["height"])
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(self.main_frame, textvariable=self.status_var, font=('Helvetica', 12)).pack(pady=10)
        
        # Add hidden loading frame with progress bar
        self.loading_frame = tk.Frame(self.main_frame)
        self.progress_bar = ttk.Progressbar(self.loading_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(pady=10)
        tk.Label(self.loading_frame, text="Loading background tools...").pack()
        self.loading_frame.pack_forget()  # Hide initially

    def start_camera(self):
        """Start camera stream and begin video update loop."""
        if self.camera_stream.start():
            self.running = True
            self._update_video()
            self._load_models_async()
            return True
        self.status_var.set("Error: Camera not found")
        return False

    def _load_models_async(self):
        """Load enabled background removal models in a background thread."""
        # Show progress bar if any models to load
        if REMBG_AVAILABLE and (self.config["background_options"]["rembg"] or self.config["background_options"]["bria"]) or (BIREFNET_AVAILABLE and self.config["background_options"]["birefnet"]):
            self.loading_frame.pack(pady=10)
            self.progress_bar.start()
            self.status_var.set("Loading tools...")
        
        def load():
            if REMBG_AVAILABLE and self.config["background_options"]["rembg"]:
                self.rembg_session = new_session(model_name="u2net")
            if REMBG_AVAILABLE and self.config["background_options"]["bria"]:
                self.bria_session = new_session(model_name="bria-rmbg")
            if BIREFNET_AVAILABLE and self.config["background_options"]["birefnet"]:
                self.birefnet_model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet-portrait", trust_remote_code=True)
                self.birefnet_model.eval()
        
        self.model_loading_thread = threading.Thread(target=load, daemon=True)
        self.model_loading_thread.start()
        
        # Check periodically if loading done
        self._check_loading_complete()

    def _check_loading_complete(self):
        """Check if model loading thread finished; hide bar and update status."""
        if self.model_loading_thread.is_alive():
            self.root.after(100, self._check_loading_complete)
        else:
            self.progress_bar.stop()
            self.loading_frame.pack_forget()
            self.status_var.set("Ready - Position yourself")

    def _update_video(self):
        """Update canvas with latest frame, cropped to 3:4 and overlaid with outline."""
        if not self.running:
            return
        ret, frame = self.camera_stream.read()
        if ret:
            # Crop frame to 3:4 aspect ratio by centering the shorter dimension
            h, w = frame.shape[:2]
            target_ratio = 3/4
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
            frame = cv2.resize(frame, (self.config["display"]["width"], self.config["display"]["height"]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame)
            if self.head_outline:
                img = Image.alpha_composite(img.convert('RGBA'), self.head_outline)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.video_after_id = self.root.after(33, self._update_video)  # Target ~30 FPS

    def capture_photo(self):
        """Capture current frame on space key press, play sound if available."""
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

    def _process_capture(self, frame):
        """Process captured frame: crop/resize to output size, generate background removal versions."""
        # Crop to 3:4 and resize to final output dimensions
        h, w = frame.shape[:2]
        target_ratio = 3/4
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
                print(f"rembg failed: {e}")
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
                print(f"BRIA failed: {e}")
        if BIREFNET_AVAILABLE and self.birefnet_model:
            try:
                # Preprocess: Resize to model input size, normalize
                image_size = (self.config["birefnet"]["processing_height"], self.config["birefnet"]["processing_width"])  # Portrait orientation
                transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(pil_img).unsqueeze(0)
                # Inference
                with torch.no_grad():
                    preds = self.birefnet_model(input_tensor)[-1].sigmoid()
                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(pil_img.size)
                # Postprocess: Apply mask to image on white bg
                result_img = Image.new('RGBA', pil_img.size)
                result_img.paste(pil_img, mask=mask.convert('L'))
                white_bg = Image.new('RGBA', result_img.size, (255, 255, 255, 255))
                version_birefnet = Image.alpha_composite(white_bg, result_img).convert('RGB')
            except Exception as e:
                print(f"BiRefNet failed: {e}")
        # Schedule preview on main thread
        self.root.after(0, lambda: self._show_preview(version_original, version_rembg, version_bria, version_birefnet))

    def _show_preview(self, v_original, v_rembg, v_bria, v_birefnet):
        """Show modal preview window with radio options, image display, filename entry, and buttons."""
        self.status_var.set("Previewing...")
        preview = tk.Toplevel(self.root)
        preview.title("Photo Preview")
        # Center window on screen for better usability
        w, h = 400, 600
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
                photo = ImageTk.PhotoImage(selected.resize((300, 400), Image.LANCZOS))
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
            selected.save(path, 'JPEG', quality=self.config["output"]["jpeg_quality"])
            preview.destroy()
            self.status_var.set(f"Saved: {name}.jpg")

        tk.Button(preview, text="Retake", command=retake).pack(side='left', padx=10, pady=10)
        tk.Button(preview, text="Save", command=save).pack(side='right', padx=10, pady=10)
        entry.bind('<Return>', lambda e: save())  # Enter key saves

    def quit_app(self):
        """Graceful shutdown: stop loops, release resources."""
        self.running = False
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
        self.camera_stream.stop()
        self.root.destroy()

    def run(self):
        """Run the app: start camera if possible, enter mainloop."""
        if self.start_camera():
            self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
            self.root.mainloop()
        else:
            self.root.destroy()

if __name__ == "__main__":
    app = IDPhotoBooth()
    app.run()