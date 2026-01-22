# Copilot Instructions for kiosk-2

## Repository overview
- **Purpose:** Python/Tkinter kiosk app for capturing standardized 300x400 JPG staff ID photos from a webcam, with live preview, head outline overlay, optional background removal (rembg/BRIA/BiRefNet), and shutter sound.
- **Repo size:** Single Python entry point plus assets/config (~8 files in repo root).
- **Tech stack:** Python 3.8+ (validated with 3.12.3), Tkinter GUI, OpenCV, Pillow, NumPy, rembg, torch/torchvision/transformers for BiRefNet, pygame for shutter sound.
- **Entry point:** `id_photo_booth.py`.

## Build / bootstrap / run / test / lint
> Follow these instructions before searching elsewhere; only explore if something is missing or incorrect.

### Bootstrap (required)
```bash
cd /home/runner/work/kiosk-2/kiosk-2
python -m pip install -r requirements.txt
```
- **Validated:** Works in this environment; downloads large dependencies (torch ~900MB + CUDA wheels). Expect several minutes and high bandwidth.
- **Note:** This installs GPU-enabled torch wheels. If disk/network is constrained, consider disabling `birefnet`/`bria` in `config.json` for runtime use, but dependencies are still required by `requirements.txt`.

### Run (GUI app)
```bash
python id_photo_booth.py
```
- **Preconditions:** `head_outline.png` and `shutter_sound.wav` must be present in repo root; a webcam must be attached; `tkinter` must be available on the system.
- **Validated failure in this environment:** `ModuleNotFoundError: No module named 'tkinter'` when running on the CI runner.
  - **Workaround:** install OS package (e.g., `python3-tk`) or run on a desktop environment that includes Tkinter.

### Tests
```bash
python -m unittest discover
```
- **Validated:** Returns `Ran 0 tests` with exit code 5 (no tests present). There are no test files in the repo.

### Lint / format
- No linting tools or configs present (no `pyproject.toml`, `setup.cfg`, `tox.ini`, or `.flake8`).

## Project layout & architecture
- **Root files:**
  - `id_photo_booth.py` — main Tkinter application; also contains config loading and background removal logic.
  - `config.json` — runtime configuration (camera/device, UI size, background removal toggles).
  - `requirements.txt` — Python dependencies.
  - `head_outline.png`, `head_outline_720x540.png` — overlay assets.
  - `shutter_sound.wav` — optional shutter sound.
  - `readme.md` — usage instructions and troubleshooting.
- **Key runtime paths:**
  - Output photos saved to `id_photos/` (created at runtime).
  - Config is loaded from `config.json` in repo root; defaults are in `load_config()` inside `id_photo_booth.py`.
- **Main flow (id_photo_booth.py):**
  - `IDPhotoBooth.run()` → `start_camera()` → `_update_video()` loop.
  - Capture flow: `capture_photo()` → `_process_capture()` (crop/resize + optional background removal) → `_show_preview()` for save/retake.

## CI / validation
- GitHub Actions workflows are minimal and primarily run the Copilot agent workflow. No dedicated build/test workflow is defined for the app itself.
- To validate changes locally: run `python -m unittest discover` (expect no tests) and ensure `python id_photo_booth.py` launches in a GUI-capable environment with Tkinter installed.

## README summary (for quick reference)
- Run app: `python id_photo_booth.py`.
- Configure `config.json` to change camera index, fullscreen, background removal options.
- Models download on first run; expect 300–500MB per model.

## Trust this document
These instructions were validated in the sandbox. Only search the repo if you need details not covered here or if the instructions are outdated.
