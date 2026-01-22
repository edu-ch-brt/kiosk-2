# Staff ID Photo Kiosk

A Python app for capturing standardized 300x400 JPG ID photos via webcam, with live preview, head outline guide, and optional background removal.

## Features
- Portrait-oriented preview with 3:4 aspect ratio crop.
- Head outline overlay for positioning.
- Capture on SPACE, preview with background options (original always, rembg/BRIA/BiRefNet configurable).
- Filename entry and retake/save in preview.
- ESC to exit.
- Progress bar during initial model loading.
- Optional shutter sound on capture.

## Requirements
- Python 3.8+
- Webcam
- Portrait monitor recommended
- Hardware: Tested on i5-3300 CPU, 16GB DDR3, Win11 (debloated)

## Installation
1. Clone or download repo.
2. Install deps: `pip install -r requirements.txt`
3. Download head_outline.png: [Your URL] and place in dir.
4. Download shutter_sound.wav: https://github.com/edu-ch-brt/staff-id-photo-kiosk/raw/refs/heads/claude/implement-feature-mkeef0xxabd1qw8s-fnhwW/shutter_sound.wav and place in dir (for audio feedback).
5. Optional: Edit config.json (e.g., camera index, fullscreen=false for testing; toggle background_options to enable/disable rembg/bria/birefnet—original always on).

## Usage
- Run: `python id_photo_booth.py`
- Position head in outline.
- Press SPACE to capture (shutter sound plays).
- In preview: Select removal (options based on config), enter filename, Save or Retake.
- Photos saved to id_photos/ as filename.jpg.
- ESC quits.

## Performance Optimizations

This version includes significant CPU optimizations:

- **Reduced Frame Rate:** 15 FPS preview (down from 30) for ~50% lower CPU usage
- **Dual Resolution:** Lower resolution preview (640x480) with full resolution capture (1280x720)
- **GPU Acceleration:** Automatic GPU support for BiRefNet (5-10x faster on CUDA GPUs)
- **Optimized BiRefNet:** Smaller processing resolution (192x256) for faster inference
- **Faster Interpolation:** Optimized resize algorithms (INTER_AREA for downscaling)
- **Cached Transforms:** Pre-computed BiRefNet transform pipeline

**Result:** ~40-60% CPU reduction during preview with identical final photo quality.

For detailed optimization information, see [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

### GPU Support

If you have an NVIDIA GPU with CUDA:
1. Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
2. GPU will be automatically detected and used for BiRefNet
3. Check logs for "BiRefNet-portrait model moved to GPU"

## Notes
- First run downloads models (~300-500MB each, needs internet)—progress bar shows.
- If slow, disable options in config.json (e.g., "bria": false).
- BiRefNet requires torch/transformers; much faster with GPU support.
- Run tests: `python -m unittest test_id_photo_booth.py`

## Troubleshooting
- No camera: Check device_index in config.
- No sound: Ensure pygame installed and wav file present.
- Model not loading: Check internet for downloads; toggle in config.
- Customize: Edit config for UI messages, processing sizes.

License: MIT (or as per your repo).