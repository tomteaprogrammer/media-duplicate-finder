#!/usr/bin/env python3
"""
Media Duplicate & Similar Finder GUI

Windows-friendly desktop tool for finding:
  - Exact duplicates of photos/videos (SHA-256)
  - Similar photos, including resized/compressed/cropped versions
  - Similar videos, including different resolutions such as 1080p vs 4K

Safety model:
  - Never deletes automatically
  - Shows grouped results for review
  - Moves only files you check into a review folder
  - Preserves the original folder hierarchy inside the review folder
  - Removes empty source folders after moving, when enabled
  - Standalone empty-folder cleanup tool with preview/removal manifest
  - Writes a CSV manifest for every move
  - Optional protected/master folders are keep-only by default
  - Save/load scan-result caches without unsafe pickle files

GPU/CUDA:
  - Uses PyTorch + torchvision embeddings when installed
  - Uses CUDA automatically when torch.cuda.is_available() is true
  - Checks/preloads ResNet50 model weights so the first scan does not surprise-download
  - Falls back to perceptual hashes if PyTorch is not installed
  - Results view uses a draggable splitter, larger rows, horizontal scrolling, and collapse/expand controls
  - Move action preserves folder hierarchy instead of flattening files into group folders
  - v1.6 improved recall but was too loose for some libraries
  - v1.7 changes similar matching to conservative consensus matching to reduce false positives
  - v1.8 makes cleanup trust-first: AI is off by default and possible crops are separate unchecked suggestions
  - v1.9 keeps NVIDIA/CUDA mode separate from AI/deep matching. Selecting NVIDIA no longer auto-checks AI.
  - v2.0 moves results into their own full-height tab with better vertical/horizontal scrolling.
  - v2.1 adds a hard keeper guard: the chosen KEEP file in each group can never be moved, even if checked manually.
  - v2.2 adds standalone empty-folder preview/removal.
  - v2.3 adds result confidence filtering, exact-only move, undo/restore from move manifest, and safer keeper scoring.
  - v2.4 keeps similar photo/video finding, but makes similar/crop/video matches review-first and unchecked by default.
  - v2.5 splits exact duplicates and similar media into separate Results subtabs.
  - v2.6 adds a Similar-tab preference button that checks lower-quality copies while skipping crop suggestions.

Performance / reliability:
  - Recursive subfolder scans
  - Parallel exact hashing
  - Disk-space guard
  - Windows long-path helpers

Run:
  py media_duplicate_finder.py

Optional first-time CUDA install from command prompt:
  py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import gzip
import json
import hashlib
import importlib.util
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
import string
import warnings
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------
# Check required libraries
# -----------------------------

REQUIRED_IMPORTS = {
    "PySide6": "PySide6",
    "PIL": "Pillow",
    "pillow_heif": "pillow-heif",
    "imagehash": "ImageHash",
    "cv2": "opencv-python",
    "numpy": "numpy",
}


def _missing_packages() -> List[str]:
    missing: List[str] = []
    for import_name, pip_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    return missing


def _check_required_packages() -> None:
    """Stop with a clear message when a required package is missing. Nothing is installed automatically."""
    missing = _missing_packages()
    if not missing:
        return
    message = (
        "Some required Python packages are missing:\n  " + ", ".join(missing) + "\n\n"
        "Open a command prompt in this program's folder and run:\n"
        "  pip install -r requirements.txt\n\n"
        "Then start the program again."
    )
    print(message, file=sys.stderr)
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Missing packages", message)
        root.destroy()
    except Exception:
        pass
    raise SystemExit(1)


_check_required_packages()

# Quiet noisy FFmpeg/OpenCV decoder warnings such as:
# [h264] mmco: unref short failure
# Those usually mean a quirky/corrupt frame, not a failed scan.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # type: ignore
import imagehash  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ImageOps, UnidentifiedImageError  # type: ignore
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass
from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes.*",
    category=UserWarning,
    module="PIL.Image",
)
try:
    # In most OpenCV builds 0 is silent. If unavailable, this harmlessly does nothing.
    cv2.setLogLevel(0)
except Exception:
    pass

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torchvision  # type: ignore
    from torchvision import models, transforms  # type: ignore
except Exception:
    torch = None
    nn = None
    torchvision = None
    models = None
    transforms = None


APP_NAME = "Media Duplicate & Similar Finder"
APP_VERSION = "2.6.0"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".mpg", ".mpeg", ".webm", ".3gp",
}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

IGNORED_DIR_NAMES = {
    "$recycle.bin", "system volume information", "windows", "program files", "program files (x86)",
    "programdata", "appdata", ".git", "node_modules", "__pycache__",
}


# -----------------------------
# Data classes
# -----------------------------

@dataclasses.dataclass
class ScanSettings:
    roots: List[str]
    protected_roots: List[str] = dataclasses.field(default_factory=list)
    include_all_files_for_exact: bool = False
    scan_exact: bool = True
    scan_similar_images: bool = True
    scan_similar_videos: bool = True
    recursive: bool = True
    use_deep_embeddings: bool = False
    include_possible_crops: bool = True
    check_similar_by_default: bool = False
    acceleration_mode: str = "auto"  # auto, cuda_required, hash_only
    skip_system_dirs: bool = True
    min_file_size_bytes: int = 1
    image_phash_threshold: int = 8
    image_crop_threshold: int = 10
    image_embedding_threshold: float = 0.94
    video_frame_phash_threshold: float = 10.0
    video_embedding_threshold: float = 0.92
    video_sample_frames: int = 12
    image_batch_size: int = 32
    max_files: int = 0  # 0 means no cap
    min_free_space_gb: float = 5.0
    hash_workers: int = 0  # 0 means auto


@dataclasses.dataclass
class FileRecord:
    path: str
    size: int
    ext: str
    media_type: str  # image, video, other
    protected: bool = False
    sha256: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    phash: Optional[Any] = None
    ahash: Optional[Any] = None
    dhash: Optional[Any] = None
    whash: Optional[Any] = None
    crop_hash: Optional[Any] = None
    embedding: Optional[np.ndarray] = None
    video_phashes: Optional[List[Any]] = None


@dataclasses.dataclass
class ResultGroup:
    group_id: int
    group_type: str
    score: str
    files: List[str]
    keep_path: str
    notes: str = ""
    move_checked_by_default: bool = True


# -----------------------------
# Utility functions
# -----------------------------


def human_bytes(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num} B"

def prepare_path(path: str) -> str:
    """Normalize paths and add the Windows long-path prefix for filesystem calls."""
    norm = os.path.abspath(os.path.normpath(path))
    if platform.system().lower() == "windows" and len(norm) >= 240 and not norm.startswith("\\\\?\\"):
        if norm.startswith("\\\\"):
            return "\\\\?\\UNC\\" + norm.lstrip("\\")
        return "\\\\?\\" + norm
    return norm


def display_path(path: str) -> str:
    """Remove Windows long-path prefixes for GUI display."""
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def same_file_path(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.abspath(display_path(a))) == os.path.normcase(os.path.abspath(display_path(b)))
    except Exception:
        return a == b


def is_subpath(path: str, parent: str) -> bool:
    if not parent:
        return False
    try:
        path_abs = os.path.normcase(os.path.abspath(display_path(path)))
        parent_abs = os.path.normcase(os.path.abspath(display_path(parent)))
        return os.path.commonpath([parent_abs, path_abs]) == parent_abs
    except Exception:
        return False


def reveal_in_file_manager(path: str) -> None:
    path = display_path(path)
    try:
        if platform.system().lower() == "windows":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif platform.system().lower() == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception:
        pass


def get_free_space_gb(folder: str) -> float:
    try:
        target = display_path(folder)
        while target and not os.path.exists(target):
            parent = os.path.dirname(target)
            if parent == target:
                break
            target = parent
        if not target:
            return 9999.0
        total, used, free = shutil.disk_usage(target)
        return free / (1024 ** 3)
    except Exception:
        return 9999.0


def check_disk_guard(roots: Sequence[str], minimum_gb: float) -> None:
    if minimum_gb <= 0:
        return
    checked: set[str] = set()
    for root in roots:
        try:
            anchor = Path(display_path(root)).anchor or display_path(root)
        except Exception:
            anchor = display_path(root)
        key = os.path.normcase(anchor)
        if key in checked:
            continue
        checked.add(key)
        free_gb = get_free_space_gb(root)
        if free_gb < minimum_gb:
            raise RuntimeError(
                f"Disk guard stopped the scan: {display_path(root)} has only {free_gb:.2f} GB free "
                f"which is below the {minimum_gb:.2f} GB limit."
            )


def run_command_capture(cmd: Sequence[str], timeout: int = 15) -> Tuple[int, str]:
    """Run a command and return combined stdout/stderr without opening a console window."""
    try:
        creationflags = 0
        if platform.system().lower() == "windows":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            creationflags=creationflags,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except FileNotFoundError:
        return 127, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out: {' '.join(cmd)}"
    except Exception as exc:
        return 1, str(exc)


def nvidia_smi_summary() -> Tuple[bool, str]:
    """Return whether NVIDIA tooling is visible and a human-readable summary."""
    rc, out = run_command_capture([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    if rc == 0 and out:
        lines = []
        for idx, line in enumerate(out.splitlines(), start=1):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                lines.append(f"GPU {idx}: {parts[0]} | Driver {parts[1]} | VRAM {parts[2]}")
            else:
                lines.append(f"GPU {idx}: {line.strip()}")
        return True, "\n".join(lines)
    return False, out or "nvidia-smi did not return GPU information."


def resnet50_weights_cache_path() -> Optional[Path]:
    """Return the expected torchvision ResNet50 checkpoint path without downloading it."""
    if torch is None or models is None:
        return None
    try:
        weights = models.ResNet50_Weights.DEFAULT
        url = getattr(weights, "url", "") or ""
        filename = Path(url).name or "resnet50-11ad3fa6.pth"
        return Path(torch.hub.get_dir()) / "checkpoints" / filename
    except Exception:
        return None


def resnet50_weights_cached() -> bool:
    path = resnet50_weights_cache_path()
    return bool(path and path.exists() and path.stat().st_size > 0)


def resnet50_weights_summary() -> str:
    path = resnet50_weights_cache_path()
    if path is None:
        return "ResNet50 model weights: cannot check because PyTorch/torchvision is not loaded."
    if path.exists():
        try:
            size_mb = path.stat().st_size / (1024 ** 2)
            return f"ResNet50 model weights: cached ({size_mb:.1f} MB) at {path}"
        except Exception:
            return f"ResNet50 model weights: cached at {path}"
    return (
        "ResNet50 model weights: not cached yet. "
        "The first deep scan or the Preload ResNet50 button will download them to "
        f"{path}"
    )


def pytorch_status_summary() -> Tuple[bool, bool, str]:
    """Return (torch_installed, cuda_available, summary)."""
    if torch is None:
        return False, False, "PyTorch/torchvision is not installed or could not import."
    lines = [f"PyTorch: {getattr(torch, '__version__', 'unknown')}"]
    if torchvision is not None:
        lines.append(f"torchvision: {getattr(torchvision, '__version__', 'unknown')}")
    try:
        lines.append(f"PyTorch CUDA build: {getattr(torch.version, 'cuda', None) or 'CPU-only build'}")
        cuda_ok = bool(torch.cuda.is_available())
        lines.append(f"torch.cuda.is_available(): {cuda_ok}")
        if cuda_ok:
            count = torch.cuda.device_count()
            lines.append(f"CUDA devices visible to PyTorch: {count}")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                lines.append(f"Device {i}: {torch.cuda.get_device_name(i)} | {vram_gb:.1f} GB VRAM")
        lines.append(resnet50_weights_summary())
        return True, cuda_ok, "\n".join(lines)
    except Exception as exc:
        lines.append(f"CUDA check error: {exc}")
        return True, False, "\n".join(lines)


def build_system_check_report() -> Tuple[str, bool, bool]:
    """Build a startup/check dialog report. Returns (report, nvidia_visible, torch_cuda_ok)."""
    nvidia_visible, nvidia_text = nvidia_smi_summary()
    torch_installed, torch_cuda_ok, torch_text = pytorch_status_summary()
    lines = [
        f"{APP_NAME} {APP_VERSION} system check",
        "",
        f"Python executable: {sys.executable}",
        f"Python version: {sys.version.split()[0]}",
        f"Operating system: {platform.platform()}",
        "",
        "NVIDIA driver / nvidia-smi:",
        nvidia_text,
        "",
        "PyTorch / CUDA:",
        torch_text,
        "",
    ]
    weights_ok = resnet50_weights_cached() if torch_installed else False
    if torch_cuda_ok and weights_ok:
        lines.append("Status: GPU acceleration is ready and ResNet50 model weights are already cached.")
    elif torch_cuda_ok and not weights_ok:
        lines.append("Status: GPU acceleration is ready, but ResNet50 model weights are not cached yet. Click Preload ResNet50 before the first deep scan to avoid a surprise download.")
    elif nvidia_visible and torch_installed:
        lines.append("Status: NVIDIA GPU is visible, but PyTorch cannot use CUDA. Install the CUDA PyTorch wheel or use CPU/laptop mode.")
    elif nvidia_visible and not torch_installed:
        lines.append("Status: NVIDIA GPU is visible, but PyTorch is missing. Install CUDA PyTorch or use CPU/laptop mode.")
    else:
        lines.append("Status: No NVIDIA CUDA setup detected. CPU/laptop mode will still scan exact duplicates and perceptual hash similarity.")
    return "\n".join(lines), nvidia_visible, torch_cuda_ok


def sha256_file(path: str, chunk_size: int = 1024 * 1024 * 4) -> str:
    h = hashlib.sha256()
    with open(prepare_path(path), "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_probably_hidden_or_system_dir(path: str) -> bool:
    name = Path(path).name.lower()
    return name in IGNORED_DIR_NAMES


def safe_stat(path: str) -> Optional[os.stat_result]:
    try:
        return os.stat(prepare_path(path))
    except OSError:
        return None


def file_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "other"


def iter_files(settings: ScanSettings, roots: Optional[Sequence[str]] = None) -> Iterable[str]:
    seen: set[str] = set()
    count = 0
    scan_roots = list(roots if roots is not None else settings.roots)
    for root in scan_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file():
            candidates = [str(root_path)]
        else:
            candidates = []
            if settings.recursive:
                for dirpath, dirnames, filenames in os.walk(root):
                    if settings.skip_system_dirs:
                        dirnames[:] = [d for d in dirnames if not is_probably_hidden_or_system_dir(os.path.join(dirpath, d))]
                    for filename in filenames:
                        candidates.append(os.path.join(dirpath, filename))
            else:
                try:
                    for child in root_path.iterdir():
                        if child.is_file():
                            candidates.append(str(child))
                except OSError:
                    continue
        for p in candidates:
            norm = prepare_path(p)
            seen_key = os.path.normcase(display_path(norm))
            if seen_key in seen:
                continue
            seen.add(seen_key)
            st = safe_stat(norm)
            if not st or st.st_size < settings.min_file_size_bytes:
                continue
            mt = file_media_type(norm)
            if mt == "other" and not settings.include_all_files_for_exact:
                continue
            yield norm
            count += 1
            if settings.max_files and count >= settings.max_files:
                return


def image_dimensions(path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.width, im.height
    except Exception:
        return None, None


def video_dimensions_duration(path: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    cap = cv2.VideoCapture(prepare_path(path))
    try:
        if not cap.isOpened():
            return None, None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration = float(frames / fps) if fps > 0 and frames > 0 else None
        return width, height, duration
    finally:
        cap.release()


def open_image_rgb(path: str, max_side: Optional[int] = None) -> Image.Image:
    """Open an image as RGB while handling EXIF orientation and transparent palettes cleanly."""
    with Image.open(prepare_path(path)) as source:
        im = ImageOps.exif_transpose(source)

        # Pillow warns when converting palette images with byte transparency directly to RGB.
        # Convert those to RGBA first, then composite onto white so hashing/embeddings are stable.
        has_palette_byte_transparency = (
            im.mode == "P" and isinstance(im.info.get("transparency"), (bytes, bytearray))
        )
        if im.mode in {"RGBA", "LA"} or has_palette_byte_transparency:
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")

        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return im.copy()


def compute_image_hashes(path: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Return several complementary image fingerprints.

    pHash is good for resized/recompressed copies. aHash/dHash/wHash catch cases where
    one perceptual hash misses. crop_resistant_hash helps with cropped screenshots/photos.
    """
    try:
        with open_image_rgb(path, max_side=1600) as im:
            ph = ah = dh = wh = crop_h = None
            try:
                ph = imagehash.phash(im, hash_size=8)
            except Exception:
                pass
            try:
                ah = imagehash.average_hash(im, hash_size=8)
            except Exception:
                pass
            try:
                dh = imagehash.dhash(im, hash_size=8)
            except Exception:
                pass
            try:
                wh = imagehash.whash(im, hash_size=8)
            except Exception:
                pass
            try:
                crop_h = imagehash.crop_resistant_hash(im)
            except Exception:
                crop_h = None
            return ph, ah, dh, wh, crop_h
    except (OSError, UnidentifiedImageError, ValueError):
        return None, None, None, None, None


def sample_video_frames(path: str, sample_count: int) -> List[Image.Image]:
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(prepare_path(path))
    try:
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # Fallback: read first N frames every 1 second if frame count is unknown.
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            positions = [int(i * fps) for i in range(sample_count)]
        else:
            # Avoid first/last black frames.
            positions = np.linspace(max(0, total * 0.08), max(0, total * 0.92), sample_count).astype(int).tolist()
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).convert("RGB")
            if max(pil.size) > 720:
                pil.thumbnail((720, 720), Image.Resampling.LANCZOS)
            frames.append(pil)
    finally:
        cap.release()
    return frames


def compute_video_phashes(path: str, sample_count: int) -> List[Any]:
    hashes: List[Any] = []
    for im in sample_video_frames(path, sample_count):
        try:
            hashes.append(imagehash.phash(im, hash_size=8))
        except Exception:
            pass
    return hashes


def mean_video_hash_distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Compare sampled video frame hashes with a small time-shift tolerance.

    The same video exported at a different frame rate, bitrate, or resolution often has
    sample points that are off by one or two frames. This shifted comparison makes
    1080p/4K and re-encoded copies match more reliably than strict index-by-index matching.
    """
    if not a or not b:
        return 999.0
    best = 999.0
    max_shift = 2 if min(len(a), len(b)) >= 6 else 1
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            aa = a[shift:]
            bb = b[:len(aa)]
        else:
            bb = b[-shift:]
            aa = a[:len(bb)]
        n = min(len(aa), len(bb))
        if n <= 0:
            continue
        dist = float(sum((aa[i] - bb[i]) for i in range(n)) / n)
        if dist < best:
            best = dist
    return best


def crop_hash_distance(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        # imagehash ImageMultiHash supports hash_diff(). Lower is more similar.
        return float(a.hash_diff(b))
    except Exception:
        try:
            return float(a - b)
        except Exception:
            return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def choose_keeper(records_by_path: Dict[str, FileRecord], files: Sequence[str]) -> str:
    def score(path: str) -> Tuple[int, int, int, int, float, str]:
        rec = records_by_path.get(path)
        st = safe_stat(path)
        mtime = st.st_mtime if st else 0.0
        area = 0
        duration_ms = 0
        size = st.st_size if st else 0
        protected = 1 if rec and rec.protected else 0
        if rec and rec.width and rec.height:
            area = int(rec.width * rec.height)
        if rec and rec.duration:
            duration_ms = int(rec.duration * 1000)
        # Safer keeper preference:
        #   1. Protected/master folder copy when present
        #   2. Higher resolution
        #   3. Longer video duration when relevant
        #   4. Larger file size
        #   5. Newer modified time as a fallback
        return (protected, area, duration_ms, size, mtime, path)

    return max(files, key=score)


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    for i in range(1, 10000):
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique destination for {dest}")


def _same_device(src: str, dest_dir: str) -> bool:
    try:
        return os.stat(src).st_dev == os.stat(dest_dir).st_dev
    except OSError:
        return False


def verified_move(src: str, dest: str) -> Tuple[bool, str]:
    """Move src to dest and check the result before the source is removed.

    Same drive: the file is renamed in place, then its size is checked.
    Different drive: the file is copied, the copy's size and SHA-256 are compared with the
    source, and only then is the source removed. On any mismatch the source is kept and the
    bad copy is removed. Returns (ok, message). dest must not exist yet.
    """
    if os.path.exists(dest):
        return False, "Destination already exists; not overwriting"
    src_size = os.path.getsize(src)
    dest_dir = os.path.dirname(dest)
    if _same_device(src, dest_dir):
        os.rename(src, dest)
        dest_size = os.path.getsize(dest) if os.path.exists(dest) else -1
        if dest_size != src_size:
            return False, f"Size check failed after rename ({dest_size} != {src_size} bytes)"
        return True, "Moved (same drive, size verified)"

    src_hash = sha256_file(src)
    shutil.copy2(src, dest)
    try:
        dest_size = os.path.getsize(dest)
        dest_hash = sha256_file(dest) if dest_size == src_size else ""
    except OSError as exc:
        dest_size, dest_hash = -1, ""
        error = str(exc)
    else:
        error = ""
    if dest_size != src_size or dest_hash != src_hash:
        try:
            os.remove(dest)
        except OSError:
            pass
        detail = error or f"size {dest_size} vs {src_size} bytes, hash match: {dest_hash == src_hash}"
        return False, f"Copy did not match the original ({detail}); original kept"
    os.remove(src)
    return True, "Moved (copied to another drive, size and SHA-256 verified)"


def safe_folder_label(text: str, max_len: int = 80) -> str:
    """Create a Windows-safe folder name for labels such as scan-root names."""
    value = display_path(str(text)).strip() or "root"
    replacements = {"<": "_", ">": "_", ":": "_", '"': "_", "/": "_", "\\": "_", "|": "_", "?": "_", "*": "_"}
    for bad, good in replacements.items():
        value = value.replace(bad, good)
    value = "_".join(value.split())
    value = value.strip(" ._") or "root"
    return value[:max_len]


def path_parts_from_rel(rel_path: str) -> List[str]:
    """Split a relative path into parts without allowing '..' to escape the review folder."""
    normalized = str(rel_path).replace("\\", "/")
    parts = []
    for part in normalized.split("/"):
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    return parts


def matching_scan_root(path: str, cleanup_roots: Sequence[str]) -> Optional[str]:
    """Return the deepest cleanup root that contains path."""
    path_display = display_path(path)
    matches: List[str] = []
    for root in cleanup_roots:
        if not root:
            continue
        root_display = display_path(root)
        if same_file_path(path_display, root_display) or is_subpath(path_display, root_display):
            matches.append(root_display)
    if not matches:
        return None
    return max(matches, key=lambda r: len(os.path.abspath(r)))


def preserved_destination_for_source(
    source: str,
    dest_root: Path,
    cleanup_roots: Sequence[str],
    preserve_hierarchy: bool = True,
    group_id: int = 0,
    group_type: str = "group",
) -> Tuple[Path, str, str]:
    """Build a destination path. Returns (destination, source_root, relative_path)."""
    src_display = display_path(source)
    if not preserve_hierarchy:
        group_folder = dest_root / f"group_{group_id:04d}_{safe_folder_label(group_type.lower().replace(' ', '_'))}"
        return group_folder / Path(src_display).name, "", Path(src_display).name

    match_root = matching_scan_root(src_display, cleanup_roots)
    if match_root:
        try:
            rel = os.path.relpath(src_display, match_root)
        except Exception:
            rel = Path(src_display).name
        rel_parts = path_parts_from_rel(rel) or [Path(src_display).name]
        # With one root, keep the review folder clean. With multiple roots, add a root label
        # so two folders/drives with the same relative paths do not collide.
        if len([r for r in cleanup_roots if r]) > 1:
            root_label = safe_folder_label(match_root)
            dest = dest_root / root_label
        else:
            dest = dest_root
        for part in rel_parts:
            dest = dest / part
        return dest, match_root, os.path.join(*rel_parts) if rel_parts else Path(src_display).name

    # Fallback for cached scans when the original cleanup roots are not loaded.
    drive, tail = os.path.splitdrive(src_display)
    root_label = safe_folder_label(drive or "absolute_path")
    rel_parts = path_parts_from_rel(tail.strip("\\/")) or [Path(src_display).name]
    dest = dest_root / "absolute_paths" / root_label
    for part in rel_parts:
        dest = dest / part
    return dest, "", os.path.join(*rel_parts) if rel_parts else Path(src_display).name


def remove_empty_parent_dirs(start_dir: str, stop_roots: Sequence[str]) -> List[str]:
    """Remove empty folders from start_dir upward, stopping before cleanup roots."""
    removed: List[str] = []
    if not stop_roots:
        return removed
    stop_norms = {os.path.normcase(os.path.abspath(display_path(r))) for r in stop_roots if r}
    current = os.path.abspath(display_path(start_dir))
    while current:
        current_norm = os.path.normcase(current)
        if current_norm in stop_norms:
            break
        if not any(is_subpath(current, root) for root in stop_roots if root):
            break
        try:
            os.rmdir(prepare_path(current))
        except OSError:
            break
        removed.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return removed


class BKTree:
    """Small BK-tree for approximate nearest-neighbor lookup with integer distances."""
    def __init__(self, distance_func):
        self.distance_func = distance_func
        self.nodes: List[Tuple[Any, str, Dict[int, int]]] = []

    def add(self, key: Any, item: str) -> None:
        if key is None:
            return
        if not self.nodes:
            self.nodes.append((key, item, {}))
            return
        idx = 0
        while True:
            node_key, _node_item, children = self.nodes[idx]
            try:
                d = int(round(float(self.distance_func(key, node_key))))
            except Exception:
                return
            if d in children:
                idx = children[d]
            else:
                children[d] = len(self.nodes)
                self.nodes.append((key, item, {}))
                return

    def query(self, key: Any, max_distance: float) -> List[Tuple[str, float]]:
        if key is None or not self.nodes:
            return []
        max_d = float(max_distance)
        out: List[Tuple[str, float]] = []
        stack = [0]
        while stack:
            idx = stack.pop()
            node_key, node_item, children = self.nodes[idx]
            try:
                d = float(self.distance_func(key, node_key))
            except Exception:
                continue
            if d <= max_d:
                out.append((node_item, d))
            low = math.floor(d - max_d)
            high = math.ceil(d + max_d)
            for child_dist, child_idx in children.items():
                if low <= child_dist <= high:
                    stack.append(child_idx)
        return out


def imagehash_distance(a: Any, b: Any) -> float:
    return float(a - b)


def _hash_dist(rec_a: FileRecord, rec_b: FileRecord, attr_name: str) -> Optional[float]:
    a = getattr(rec_a, attr_name, None)
    b = getattr(rec_b, attr_name, None)
    if a is None or b is None:
        return None
    try:
        return float(a - b)
    except Exception:
        return None


def _aspect_ratio(rec: FileRecord) -> Optional[float]:
    if rec.width and rec.height and rec.width > 0 and rec.height > 0:
        return float(rec.width) / float(rec.height)
    return None


def _aspect_close(rec_a: FileRecord, rec_b: FileRecord, tolerance: float = 0.15) -> bool:
    ar_a = _aspect_ratio(rec_a)
    ar_b = _aspect_ratio(rec_b)
    if ar_a is None or ar_b is None:
        return True
    if ar_a <= 0 or ar_b <= 0:
        return True
    return abs(math.log(ar_a / ar_b)) <= tolerance


def photo_pair_is_strict_duplicate(rec_a: FileRecord, rec_b: FileRecord, settings: ScanSettings) -> Optional[str]:
    """Trust-first verifier for same-photo resized/compressed copies.

    This intentionally avoids AI-only matches and avoids crop-only matches. A pair is accepted
    only when pHash is extremely close or several independent perceptual hashes agree while
    the aspect ratios are close. This is designed for cleanup, not broad visual discovery.
    """
    ph = _hash_dist(rec_a, rec_b, "phash")
    ah = _hash_dist(rec_a, rec_b, "ahash")
    dh = _hash_dist(rec_a, rec_b, "dhash")
    wh = _hash_dist(rec_a, rec_b, "whash")

    ph_t = float(settings.image_phash_threshold)
    aux_t = max(3.0, ph_t - 1.0)
    aspect_ok = _aspect_close(rec_a, rec_b, tolerance=0.10)
    if not aspect_ok:
        return None

    # Very close pHash is the safest resized/compressed-photo signal.
    if ph is not None and ph <= min(4.0, ph_t):
        return f"very close pHash={ph:.0f}"

    close_names: List[str] = []
    for name, dist, limit in (
        ("pHash", ph, ph_t),
        ("dHash", dh, aux_t),
        ("wHash", wh, aux_t),
        ("aHash", ah, max(2.0, aux_t - 1.0)),
    ):
        if dist is not None and dist <= limit:
            close_names.append(f"{name}={dist:.0f}")

    # Require at least 3 agreeing hashes for normal cleanup. Two hashes was too loose for
    # personal photo libraries where many photos share the same subject/background.
    if len(close_names) >= 3:
        return " + ".join(close_names[:4])

    return None


def photo_pair_is_possible_crop_candidate(rec_a: FileRecord, rec_b: FileRecord, settings: ScanSettings) -> Optional[str]:
    """Separate, unchecked review suggestion for possible crops/edits.

    These are not treated as cleanup-safe duplicates. They are shown only when the user turns
    on possible crop suggestions and they are left unchecked by default.
    """
    crop = crop_hash_distance(rec_a.crop_hash, rec_b.crop_hash)
    if crop is None:
        return None
    ph = _hash_dist(rec_a, rec_b, "phash")
    dh = _hash_dist(rec_a, rec_b, "dhash")
    wh = _hash_dist(rec_a, rec_b, "whash")
    ah = _hash_dist(rec_a, rec_b, "ahash")
    sim: Optional[float] = None
    if rec_a.embedding is not None and rec_b.embedding is not None:
        sim = cosine_similarity(rec_a.embedding, rec_b.embedding)

    crop_t = float(settings.image_crop_threshold)
    if crop > crop_t:
        return None

    # Crop suggestions need confirmation from another signal, but are still review-only.
    confirmations: List[str] = [f"crop={crop:.1f}"]
    for name, dist, limit in (
        ("pHash", ph, settings.image_phash_threshold + 2),
        ("dHash", dh, settings.image_phash_threshold + 2),
        ("wHash", wh, settings.image_phash_threshold + 2),
        ("aHash", ah, settings.image_phash_threshold + 1),
    ):
        if dist is not None and dist <= float(limit):
            confirmations.append(f"{name}={dist:.0f}")
            break
    if len(confirmations) > 1:
        return " + ".join(confirmations)
    if sim is not None and sim >= max(0.96, float(settings.image_embedding_threshold)):
        return f"crop={crop:.1f} + embedding={sim:.3f}"
    return None

def union_photo_hash_bktree_validated(
    paths: Sequence[str],
    records: Dict[str, FileRecord],
    attr_name: str,
    threshold: float,
    uf: "UnionFind",
    settings: ScanSettings,
    progress_cb=None,
    label: str = "hash",
) -> int:
    """Use a hash BK-tree for candidate generation, then validate with consensus."""
    tree = BKTree(imagehash_distance)
    unions = 0
    valid = [p for p in paths if getattr(records[p], attr_name, None) is not None]
    for idx, path in enumerate(valid):
        key = getattr(records[path], attr_name, None)
        if key is not None:
            for other, _dist in tree.query(key, threshold):
                if photo_pair_is_strict_duplicate(records[path], records[other], settings):
                    uf.union(path, other)
                    unions += 1
            tree.add(key, path)
        if progress_cb and idx % 500 == 0:
            progress_cb(idx + 1, len(valid), f"Grouping photos by verified {label}: {idx + 1}/{len(valid)}")
    return unions


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> List[List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for item in list(self.parent.keys()):
            grouped[self.find(item)].append(item)
        return [g for g in grouped.values() if len(g) > 1]


# -----------------------------
# Deep embedding model
# -----------------------------

class EmbeddingModel:
    def __init__(self, device_preference: str = "auto") -> None:
        if torch is None or models is None or transforms is None or nn is None:
            raise RuntimeError("PyTorch/torchvision are not installed.")
        if device_preference == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was required, but torch.cuda.is_available() is False.")
            self.device = torch.device("cuda")
        elif device_preference == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_preference)
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        self.preprocess = weights.transforms()
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
        self.name = f"ResNet50/ImageNet on {self.device}"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.zeros((0, 2048), dtype=np.float32)
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                batch = torch.stack([self.preprocess(im) for im in batch_images]).to(self.device)
                feats = self.model(batch).flatten(1)
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                outs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outs)

    def embed_image_paths(self, paths: Sequence[str], batch_size: int = 32, progress_cb=None) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        batch_paths: List[str] = []
        batch_images: List[Image.Image] = []
        for idx, path in enumerate(paths):
            try:
                im = open_image_rgb(path, max_side=1200)
            except Exception:
                if progress_cb:
                    progress_cb(idx + 1, len(paths), f"Skipping unreadable image: {Path(path).name}")
                continue
            batch_paths.append(path)
            batch_images.append(im)
            if len(batch_images) >= batch_size:
                emb = self.embed_images(batch_images, batch_size=batch_size)
                for p, e in zip(batch_paths, emb):
                    result[p] = e
                batch_paths.clear()
                batch_images.clear()
            if progress_cb:
                progress_cb(idx + 1, len(paths), f"Embedding images: {idx + 1}/{len(paths)}")
        if batch_images:
            emb = self.embed_images(batch_images, batch_size=batch_size)
            for p, e in zip(batch_paths, emb):
                result[p] = e
        return result

    def embed_video_path(self, path: str, sample_count: int, batch_size: int = 32) -> Optional[np.ndarray]:
        frames = sample_video_frames(path, sample_count)
        if not frames:
            return None
        emb = self.embed_images(frames, batch_size=batch_size)
        if emb.shape[0] == 0:
            return None
        avg = emb.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-12:
            avg = avg / norm
        return avg.astype(np.float32)


# -----------------------------
# Result filtering and cache helpers
# -----------------------------

def split_protected(records: Dict[str, FileRecord], files: Sequence[str]) -> Tuple[List[str], List[str]]:
    protected_files = [p for p in files if records.get(p) and records[p].protected]
    cleanup_files = [p for p in files if records.get(p) and not records[p].protected]
    return protected_files, cleanup_files


def make_result_group(
    records: Dict[str, FileRecord],
    files: Sequence[str],
    group_id: int,
    base_type: str,
    score: str,
    notes: str,
    protected_mode: bool,
    move_checked_by_default: bool = True,
) -> Optional[ResultGroup]:
    files = sorted(set(files))
    if len(files) < 2:
        return None
    if protected_mode:
        protected_files, cleanup_files = split_protected(records, files)
        if protected_files and cleanup_files:
            keep = choose_keeper(records, protected_files)
            return ResultGroup(
                group_id=group_id,
                group_type=f"{base_type} - matches protected",
                score=score,
                files=sorted(protected_files + cleanup_files),
                keep_path=keep,
                notes=notes + " Protected folder files are marked keep-only by default.",
                move_checked_by_default=move_checked_by_default,
            )
        if len(cleanup_files) > 1:
            keep = choose_keeper(records, cleanup_files)
            return ResultGroup(
                group_id=group_id,
                group_type=f"{base_type} - cleanup internal",
                score=score,
                files=sorted(cleanup_files),
                keep_path=keep,
                notes=notes,
                move_checked_by_default=move_checked_by_default,
            )
        return None
    keep = choose_keeper(records, files)
    return ResultGroup(group_id=group_id, group_type=base_type, score=score, files=files, keep_path=keep, notes=notes, move_checked_by_default=move_checked_by_default)


def record_to_cache_dict(rec: FileRecord) -> Dict[str, Any]:
    return {
        "path": rec.path,
        "size": rec.size,
        "ext": rec.ext,
        "media_type": rec.media_type,
        "protected": rec.protected,
        "sha256": rec.sha256,
        "width": rec.width,
        "height": rec.height,
        "duration": rec.duration,
    }


def record_from_cache_dict(data: Dict[str, Any]) -> FileRecord:
    return FileRecord(
        path=str(data.get("path", "")),
        size=int(data.get("size") or 0),
        ext=str(data.get("ext", "")),
        media_type=str(data.get("media_type", "other")),
        protected=bool(data.get("protected", False)),
        sha256=data.get("sha256"),
        width=data.get("width"),
        height=data.get("height"),
        duration=data.get("duration"),
    )


def group_to_cache_dict(group: ResultGroup) -> Dict[str, Any]:
    return dataclasses.asdict(group)


def group_from_cache_dict(data: Dict[str, Any]) -> ResultGroup:
    return ResultGroup(
        group_id=int(data.get("group_id") or 0),
        group_type=str(data.get("group_type", "Group")),
        score=str(data.get("score", "")),
        files=[str(x) for x in data.get("files", [])],
        keep_path=str(data.get("keep_path", "")),
        notes=str(data.get("notes", "")),
        move_checked_by_default=bool(data.get("move_checked_by_default", True)),
    )

# -----------------------------
# Scanner worker
# -----------------------------

class ScannerWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(list, dict, str)
    failed = QtCore.Signal(str)

    def __init__(self, settings: ScanSettings):
        super().__init__()
        self.settings = settings
        self._cancel = False

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel = True

    def _check_cancel(self) -> None:
        if self._cancel:
            raise RuntimeError("Scan canceled by user.")

    @QtCore.Slot()
    def run(self) -> None:
        try:
            results, records, device_info = self._run_scan()
            self.finished.emit(results, records, device_info)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_scan(self) -> Tuple[List[ResultGroup], Dict[str, FileRecord], str]:
        settings = self.settings
        self.log.emit(f"{APP_NAME} {APP_VERSION}")
        self.log.emit(f"Python: {sys.version.split()[0]} on {platform.platform()}")

        device_info = "Deep embeddings disabled or unavailable."
        embedder: Optional[EmbeddingModel] = None
        if settings.acceleration_mode == "hash_only":
            device_info = "Laptop/CPU-safe mode: PyTorch/CUDA deep embeddings skipped."
            self.log.emit(device_info)
        elif settings.use_deep_embeddings:
            try:
                device_preference = "cuda" if settings.acceleration_mode == "cuda_required" else "auto"
                embedder = EmbeddingModel(device_preference=device_preference)
                device_info = f"Deep embeddings: {embedder.name}"
                self.log.emit(device_info)
                if torch is not None and torch.cuda.is_available():
                    self.log.emit(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
                elif settings.acceleration_mode == "auto":
                    self.log.emit("PyTorch is available but CUDA is not; CPU embeddings may be slow on laptops.")
            except Exception as exc:
                if settings.acceleration_mode == "cuda_required":
                    raise RuntimeError(f"NVIDIA/CUDA mode was required but is not ready: {exc}")
                self.log.emit(f"Deep embeddings unavailable; using hash-only fallback. Reason: {exc}")

        self.progress.emit(0, 0, "Collecting files...")
        all_roots = list(settings.roots) + list(settings.protected_roots)
        check_disk_guard(all_roots, settings.min_free_space_gb)
        cleanup_paths = list(iter_files(settings, settings.roots))
        self._check_cancel()
        protected_paths = list(iter_files(settings, settings.protected_roots)) if settings.protected_roots else []
        self._check_cancel()
        protected_keys = {os.path.normcase(display_path(p)) for p in protected_paths}
        ordered_paths: List[str] = []
        seen_path_keys: set[str] = set()
        for p in cleanup_paths + protected_paths:
            key = os.path.normcase(display_path(p))
            if key not in seen_path_keys:
                seen_path_keys.add(key)
                ordered_paths.append(p)
        paths = ordered_paths
        self.log.emit(f"Found {len(cleanup_paths):,} cleanup candidate files and {len(protected_paths):,} protected candidate files.")
        if not paths:
            return [], {}, device_info

        records: Dict[str, FileRecord] = {}
        for i, path in enumerate(paths):
            if i % 500 == 0:
                check_disk_guard(all_roots, settings.min_free_space_gb)
            st = safe_stat(path)
            if not st:
                continue
            mt = file_media_type(path)
            is_protected = os.path.normcase(display_path(path)) in protected_keys
            rec = FileRecord(path=path, size=st.st_size, ext=Path(display_path(path)).suffix.lower(), media_type=mt, protected=is_protected)
            records[path] = rec
            if mt == "image":
                rec.width, rec.height = image_dimensions(path)
            elif mt == "video":
                rec.width, rec.height, rec.duration = video_dimensions_duration(path)
            if i % 200 == 0:
                self.progress.emit(i + 1, len(paths), f"Reading metadata: {i + 1}/{len(paths)}")
        self._check_cancel()

        result_groups: List[ResultGroup] = []
        group_counter = 1

        # Exact duplicates by SHA-256, optimized by file size first.
        if settings.scan_exact:
            self.progress.emit(0, 0, "Finding exact duplicates...")
            size_groups: Dict[int, List[str]] = defaultdict(list)
            for path, rec in records.items():
                if rec.media_type == "other" and not settings.include_all_files_for_exact:
                    continue
                size_groups[rec.size].append(path)

            needs_hash = [p for same_size in size_groups.values() if len(same_size) > 1 for p in same_size]
            max_workers = settings.hash_workers or min(8, (os.cpu_count() or 4) + 2)
            def _hash_one(p: str) -> Tuple[str, Optional[str], Optional[str]]:
                try:
                    return p, sha256_file(p), None
                except OSError as exc:
                    return p, None, str(exc)
            if needs_hash:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_hash_one, p) for p in needs_hash]
                    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        self._check_cancel()
                        if idx % 250 == 0:
                            check_disk_guard(all_roots, settings.min_free_space_gb)
                        path, sha, err = future.result()
                        if sha:
                            records[path].sha256 = sha
                        elif err:
                            self.log.emit(f"Could not hash {path}: {err}")
                        self.progress.emit(idx + 1, len(needs_hash), f"Hashing exact duplicates: {idx + 1}/{len(needs_hash)}")

            sha_groups: Dict[str, List[str]] = defaultdict(list)
            for path in needs_hash:
                sha = records[path].sha256
                if sha:
                    sha_groups[sha].append(path)
            exact_start_counter = group_counter
            for sha, files in sha_groups.items():
                group = make_result_group(
                    records,
                    files,
                    group_counter,
                    "Exact duplicate",
                    f"SHA-256 {sha[:12]}…",
                    "Byte-for-byte identical files.",
                    protected_mode=bool(settings.protected_roots),
                )
                if group:
                    result_groups.append(group)
                    group_counter += 1
            self.log.emit(f"Exact duplicate groups: {group_counter - exact_start_counter}")

        # Similar images.
        image_paths = [p for p, r in records.items() if r.media_type == "image"]
        if settings.scan_similar_images and image_paths:
            self.progress.emit(0, len(image_paths), "Computing photo hashes...")
            for idx, path in enumerate(image_paths):
                self._check_cancel()
                ph, ah, dh, wh, ch = compute_image_hashes(path)
                records[path].phash = ph
                records[path].ahash = ah
                records[path].dhash = dh
                records[path].whash = wh
                records[path].crop_hash = ch
                self.progress.emit(idx + 1, len(image_paths), f"Photo hashes: {idx + 1}/{len(image_paths)}")

            if embedder is not None:
                self.progress.emit(0, len(image_paths), "Computing photo deep embeddings...")
                embeddings = embedder.embed_image_paths(
                    image_paths,
                    batch_size=settings.image_batch_size,
                    progress_cb=lambda a, b, m: self.progress.emit(a, b, m),
                )
                for p, e in embeddings.items():
                    records[p].embedding = e

            self.progress.emit(0, 0, "Grouping similar photos...")
            uf = UnionFind()
            image_paths_valid = [p for p in image_paths if records[p].phash is not None or records[p].ahash is not None or records[p].dhash is not None or records[p].whash is not None or records[p].crop_hash is not None or records[p].embedding is not None]
            for p in image_paths_valid:
                uf.find(p)

            # Conservative consensus grouping:
            #   1) BK-trees generate candidate pairs from multiple perceptual hashes
            #   2) photo_pair_is_strict_duplicate validates candidates with at least one very strong
            #      signal or two independent matching signals
            #   3) possible crops are collected separately as review-only suggestions; deep
            #      embeddings can confirm a crop suggestion but never group photos by themselves
            phash_unions = union_photo_hash_bktree_validated(
                image_paths_valid, records, "phash", settings.image_phash_threshold, uf, settings,
                progress_cb=lambda a, b, m: self.progress.emit(a, b, m), label="pHash"
            )
            ahash_unions = union_photo_hash_bktree_validated(
                image_paths_valid, records, "ahash", max(3, settings.image_phash_threshold - 1), uf, settings,
                progress_cb=lambda a, b, m: self.progress.emit(a, b, m), label="aHash"
            )
            dhash_unions = union_photo_hash_bktree_validated(
                image_paths_valid, records, "dhash", max(4, settings.image_phash_threshold - 1), uf, settings,
                progress_cb=lambda a, b, m: self.progress.emit(a, b, m), label="dHash"
            )
            whash_unions = union_photo_hash_bktree_validated(
                image_paths_valid, records, "whash", max(4, settings.image_phash_threshold - 1), uf, settings,
                progress_cb=lambda a, b, m: self.progress.emit(a, b, m), label="wHash"
            )

            possible_crop_unions = 0
            possible_crop_uf = UnionFind()
            if settings.include_possible_crops:
                crop_tree = BKTree(lambda a, b: crop_hash_distance(a, b) if crop_hash_distance(a, b) is not None else 9999.0)
                crop_paths = [p for p in image_paths_valid if records[p].crop_hash is not None]
                for p in crop_paths:
                    possible_crop_uf.find(p)
                for idx, path in enumerate(crop_paths):
                    self._check_cancel()
                    ch = records[path].crop_hash
                    if ch is not None:
                        for other, _dist in crop_tree.query(ch, settings.image_crop_threshold):
                            # Do not duplicate a strict group. Possible crop results are review-only.
                            if uf.find(path) != uf.find(other) and photo_pair_is_possible_crop_candidate(records[path], records[other], settings):
                                possible_crop_uf.union(path, other)
                                possible_crop_unions += 1
                        crop_tree.add(ch, path)
                    if idx % 300 == 0:
                        self.progress.emit(idx + 1, len(crop_paths), f"Finding possible crop suggestions: {idx + 1}/{len(crop_paths)}")

            if any(records[p].embedding is not None for p in image_paths_valid):
                self.log.emit("Photo embeddings loaded: used only as extra confirmation; AI-only photo matches are disabled by default.")
            self.log.emit(
                "Photo links after trust-first verification: "
                f"pHash={phash_unions}, aHash={ahash_unions}, dHash={dhash_unions}, "
                f"wHash={whash_unions}, possible_crop_suggestions={possible_crop_unions}"
            )

            for files in uf.groups():
                group = make_result_group(
                    records,
                    files,
                    group_counter,
                    "Similar photo",
                    "strict resized/compressed match",
                    "Likely same photo exported at a different size or compression. AI-only and crop-only matches are not used for cleanup.",
                    protected_mode=bool(settings.protected_roots),
                    move_checked_by_default=settings.check_similar_by_default,
                )
                if group:
                    result_groups.append(group)
                    group_counter += 1

            if settings.include_possible_crops:
                for files in possible_crop_uf.groups():
                    group = make_result_group(
                        records,
                        files,
                        group_counter,
                        "Possible cropped/edited photo",
                        "review-only crop suggestion",
                        "Possible crop/edit relationship. Files in this group are intentionally unchecked by default.",
                        protected_mode=bool(settings.protected_roots),
                        move_checked_by_default=False,
                    )
                    if group:
                        result_groups.append(group)
                        group_counter += 1
            self.log.emit("Similar photo grouping complete.")

        # Similar videos.
        video_paths = [p for p, r in records.items() if r.media_type == "video"]
        if settings.scan_similar_videos and video_paths:
            self.progress.emit(0, len(video_paths), "Computing video frame hashes...")
            for idx, path in enumerate(video_paths):
                self._check_cancel()
                records[path].video_phashes = compute_video_phashes(path, settings.video_sample_frames)
                self.progress.emit(idx + 1, len(video_paths), f"Video frame hashes: {idx + 1}/{len(video_paths)}")

            if embedder is not None:
                self.progress.emit(0, len(video_paths), "Computing video deep embeddings...")
                for idx, path in enumerate(video_paths):
                    self._check_cancel()
                    try:
                        records[path].embedding = embedder.embed_video_path(
                            path,
                            sample_count=settings.video_sample_frames,
                            batch_size=settings.image_batch_size,
                        )
                    except Exception as exc:
                        self.log.emit(f"Could not embed video {path}: {exc}")
                    self.progress.emit(idx + 1, len(video_paths), f"Video embeddings: {idx + 1}/{len(video_paths)}")

            self.progress.emit(0, 0, "Grouping similar videos...")
            ufv = UnionFind()
            for p in video_paths:
                ufv.find(p)
            total_pairs = len(video_paths) * (len(video_paths) - 1) // 2
            done = 0
            video_hash_unions = 0
            video_embedding_unions = 0
            for i in range(len(video_paths)):
                self._check_cancel()
                a = video_paths[i]
                ra = records[a]
                for j in range(i + 1, len(video_paths)):
                    b = video_paths[j]
                    rb = records[b]
                    # Avoid comparing obviously different durations unless duration metadata is unavailable.
                    if ra.duration and rb.duration:
                        ratio = abs(ra.duration - rb.duration) / max(ra.duration, rb.duration)
                        if ratio > 0.08:
                            done += 1
                            continue
                    matched = False
                    frame_dist: Optional[float] = None
                    if ra.video_phashes and rb.video_phashes:
                        frame_dist = mean_video_hash_distance(ra.video_phashes, rb.video_phashes)
                        if frame_dist <= settings.video_frame_phash_threshold:
                            matched = True
                            video_hash_unions += 1
                    # Video embeddings are used only as confirmation unless extremely high.
                    # This avoids grouping different videos with similar scenes or subjects.
                    if not matched and ra.embedding is not None and rb.embedding is not None:
                        sim = cosine_similarity(ra.embedding, rb.embedding)
                        if sim >= max(0.975, settings.video_embedding_threshold + 0.04):
                            matched = True
                            video_embedding_unions += 1
                        elif frame_dist is not None and frame_dist <= settings.video_frame_phash_threshold + 3 and sim >= settings.video_embedding_threshold:
                            matched = True
                            video_embedding_unions += 1
                    if matched:
                        ufv.union(a, b)
                    done += 1
                    if done % 1000 == 0:
                        self.progress.emit(done, max(total_pairs, 1), f"Comparing videos: {done:,}/{total_pairs:,}")

            for files in ufv.groups():
                group = make_result_group(
                    records,
                    files,
                    group_counter,
                    "Similar video",
                    "sampled-frame match",
                    "Likely same or near-same video, possibly different resolution/bitrate. Review before moving.",
                    protected_mode=bool(settings.protected_roots),
                    move_checked_by_default=settings.check_similar_by_default,
                )
                if group:
                    result_groups.append(group)
                    group_counter += 1
            self.log.emit(f"Video similarity links: frame_hash={video_hash_unions}, embedding={video_embedding_unions}")
            self.log.emit("Similar video grouping complete.")

        self.progress.emit(1, 1, "Scan complete.")
        return result_groups, records, device_info


# -----------------------------
# Move worker
# -----------------------------

class MoveWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(str, int)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        moves: List[Tuple[int, str, str]],
        dest_root: str,
        cleanup_roots: Optional[Sequence[str]] = None,
        preserve_hierarchy: bool = True,
        prune_empty_dirs: bool = True,
    ):
        super().__init__()
        self.moves = moves
        self.dest_root = Path(dest_root)
        self.cleanup_roots = [display_path(r) for r in (cleanup_roots or []) if r]
        self.preserve_hierarchy = preserve_hierarchy
        self.prune_empty_dirs = prune_empty_dirs
        self.failed_count = 0

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.dest_root.mkdir(parents=True, exist_ok=True)
            manifest_path = self.dest_root / f"move_manifest_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "action",
                    "group_id",
                    "group_type",
                    "source_root",
                    "relative_path",
                    "source",
                    "destination",
                ])
                for idx, (group_id, group_type, source) in enumerate(self.moves):
                    source_display = display_path(source)
                    src_fs = Path(prepare_path(source_display))
                    if not src_fs.exists():
                        self.progress.emit(idx + 1, len(self.moves), f"Missing, skipped: {source_display}")
                        writer.writerow([
                            _dt.datetime.now().isoformat(timespec="seconds"),
                            "missing_skipped",
                            group_id,
                            group_type,
                            "",
                            "",
                            source_display,
                            "",
                        ])
                        continue

                    raw_dest, source_root, rel_path = preserved_destination_for_source(
                        source_display,
                        self.dest_root,
                        self.cleanup_roots,
                        preserve_hierarchy=self.preserve_hierarchy,
                        group_id=group_id,
                        group_type=group_type,
                    )
                    dest = unique_destination(raw_dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    ok, verify_message = verified_move(str(src_fs), prepare_path(str(dest)))
                    if not ok:
                        self.failed_count += 1
                        writer.writerow([
                            _dt.datetime.now().isoformat(timespec="seconds"),
                            "verify_failed_kept_source",
                            group_id,
                            group_type,
                            source_root,
                            rel_path,
                            source_display,
                            verify_message,
                        ])
                        self.progress.emit(idx + 1, len(self.moves), f"Not moved (check failed): {source_display}")
                        continue
                    writer.writerow([
                        _dt.datetime.now().isoformat(timespec="seconds"),
                        "moved_file",
                        group_id,
                        group_type,
                        source_root,
                        rel_path,
                        source_display,
                        str(dest),
                    ])

                    if self.prune_empty_dirs:
                        for removed_dir in remove_empty_parent_dirs(os.path.dirname(source_display), self.cleanup_roots):
                            writer.writerow([
                                _dt.datetime.now().isoformat(timespec="seconds"),
                                "removed_empty_folder",
                                group_id,
                                group_type,
                                source_root,
                                "",
                                removed_dir,
                                "",
                            ])

                    self.progress.emit(idx + 1, len(self.moves), f"Moved {idx + 1}/{len(self.moves)}")
            self.finished.emit(str(manifest_path), self.failed_count)
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Command worker (pip install / model download)
# -----------------------------

class CommandWorker(QtCore.QObject):
    """Run one command in a background thread and stream its output to the log.

    Used for the PyTorch install buttons and the ResNet50 preload so the window
    stays responsive during large downloads.
    """
    output = QtCore.Signal(str)
    finished = QtCore.Signal(int)
    failed = QtCore.Signal(str)

    def __init__(self, command: Sequence[str]):
        super().__init__()
        self.command = list(command)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system().lower() == "windows" else 0
            proc = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.output.emit(line)
            self.finished.emit(proc.wait())
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Restore worker
# -----------------------------

class RestoreWorker(QtCore.QObject):
    """Restore files moved by a previous move_manifest_*.csv.

    Safety rules:
      - Reads only rows whose action is moved_file.
      - Moves destination back to source only when destination exists and source does not already exist.
      - Never overwrites an existing source file.
      - Writes a restore manifest beside the selected move manifest.
    """
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(str, int, int)
    failed = QtCore.Signal(str)

    def __init__(self, manifest_path: str):
        super().__init__()
        self.manifest_path = manifest_path

    @QtCore.Slot()
    def run(self) -> None:
        try:
            manifest = Path(self.manifest_path)
            rows: List[Dict[str, str]] = []
            with open(manifest, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("action", "")).strip() == "moved_file":
                        rows.append({k: str(v or "") for k, v in row.items()})

            restore_path = manifest.with_name(f"restore_manifest_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            restored = 0
            skipped = 0
            # Reverse order is safest when nested folders/files were moved.
            rows = list(reversed(rows))
            with open(restore_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "action", "source", "destination", "message"])
                total = len(rows)
                for idx, row in enumerate(rows, start=1):
                    original_source = display_path(row.get("source", ""))
                    moved_destination = display_path(row.get("destination", ""))
                    if not original_source or not moved_destination:
                        skipped += 1
                        writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "skipped_bad_row", original_source, moved_destination, "Missing source or destination in manifest row"])
                        self.progress.emit(idx, max(total, 1), f"Skipped bad row {idx}/{total}")
                        continue
                    dest_fs = Path(prepare_path(moved_destination))
                    src_fs = Path(prepare_path(original_source))
                    if not dest_fs.exists():
                        skipped += 1
                        writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "skipped_missing_destination", original_source, moved_destination, "Moved file is not in the review folder anymore"])
                    elif src_fs.exists():
                        skipped += 1
                        writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "skipped_source_exists", original_source, moved_destination, "Original source path already exists; not overwriting"])
                    else:
                        src_fs.parent.mkdir(parents=True, exist_ok=True)
                        ok, verify_message = verified_move(str(dest_fs), prepare_path(str(src_fs)))
                        if ok:
                            restored += 1
                            writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "restored_file", original_source, moved_destination, "Restored"])
                        else:
                            skipped += 1
                            writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "skipped_verify_failed", original_source, moved_destination, verify_message])
                    self.progress.emit(idx, max(total, 1), f"Restoring files: {idx}/{total}")
            self.finished.emit(str(restore_path), restored, skipped)
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Standalone empty-folder cleanup worker
# -----------------------------

class EmptyFolderCleanupWorker(QtCore.QObject):
    """Preview or remove empty folders under cleanup roots without scanning duplicates.

    Safety rules:
      - Never removes files.
      - Never removes the selected cleanup roots themselves.
      - Removes only directories that are empty now or become empty after child
        empty folders are removed.
      - Writes a CSV manifest for preview/removal.
    """
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(str, int, bool)
    failed = QtCore.Signal(str)

    def __init__(self, roots: Sequence[str], skip_system_dirs: bool = True, dry_run: bool = True):
        super().__init__()
        self.roots = [display_path(r) for r in roots if r]
        self.skip_system_dirs = skip_system_dirs
        self.dry_run = dry_run

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(display_path(path)))

    def _removable_empty_folders(self) -> List[str]:
        """Return folders removable in deepest-first order.

        This simulates bottom-up removal, so a parent folder is included when it
        contains no files and all of its child folders are also removable.
        """
        root_norms = {self._norm(r) for r in self.roots if os.path.isdir(prepare_path(r))}
        info: Dict[str, Tuple[List[str], int]] = {}
        all_dirs: List[str] = []
        seen: set[str] = set()

        for root in self.roots:
            root_display = display_path(root)
            if not os.path.isdir(prepare_path(root_display)):
                continue
            try:
                walker = os.walk(prepare_path(root_display), topdown=True)
                for dirpath, dirnames, filenames in walker:
                    dir_display = display_path(dirpath)
                    if self.skip_system_dirs:
                        dirnames[:] = [
                            d for d in dirnames
                            if not is_probably_hidden_or_system_dir(os.path.join(dir_display, d))
                        ]
                    child_dirs = [display_path(os.path.join(dir_display, d)) for d in dirnames]
                    norm = self._norm(dir_display)
                    info[dir_display] = (child_dirs, len(filenames))
                    if norm not in root_norms and norm not in seen:
                        all_dirs.append(dir_display)
                        seen.add(norm)
            except Exception:
                # Skip roots/folders we cannot enumerate.
                continue

        all_dirs.sort(key=lambda p: len(os.path.abspath(display_path(p))), reverse=True)
        removable_norms: set[str] = set()
        removable: List[str] = []
        for folder in all_dirs:
            child_dirs, file_count = info.get(folder, ([], 1))
            child_norms = [self._norm(c) for c in child_dirs]
            if file_count == 0 and all(cn in removable_norms for cn in child_norms):
                removable_norms.add(self._norm(folder))
                removable.append(folder)
        return removable

    @QtCore.Slot()
    def run(self) -> None:
        try:
            action = "preview_empty_folder" if self.dry_run else "removed_empty_folder"
            manifest_path = Path.home() / f"MediaDuplicateFinder_empty_folders_{'preview' if self.dry_run else 'removed'}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.progress.emit(0, 0, "Finding empty folders...")
            removable = self._removable_empty_folders()
            total = len(removable)
            count = 0
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "action", "folder"])
                if self.dry_run:
                    for idx, folder in enumerate(removable, start=1):
                        writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), action, display_path(folder)])
                        if idx % 25 == 0 or idx == total:
                            self.progress.emit(idx, total, f"Previewed {idx}/{total} empty folders")
                    count = total
                else:
                    for idx, folder in enumerate(removable, start=1):
                        try:
                            os.rmdir(prepare_path(folder))
                            writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), action, display_path(folder)])
                            count += 1
                        except OSError as exc:
                            writer.writerow([_dt.datetime.now().isoformat(timespec="seconds"), "skipped_not_empty_or_locked", f"{display_path(folder)} | {exc}"])
                        if idx % 25 == 0 or idx == total:
                            self.progress.emit(idx, total, f"Removed {count}/{total} empty folders")
            self.finished.emit(str(manifest_path), count, self.dry_run)
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# GUI
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1500, 950)
        self.setMinimumSize(1100, 720)
        self.records: Dict[str, FileRecord] = {}
        self.groups: List[ResultGroup] = []
        self.scan_thread: Optional[QtCore.QThread] = None
        self.scan_worker: Optional[ScannerWorker] = None
        self.move_thread: Optional[QtCore.QThread] = None
        self.move_worker: Optional[MoveWorker] = None
        self.empty_thread: Optional[QtCore.QThread] = None
        self.empty_worker: Optional[EmptyFolderCleanupWorker] = None
        self.restore_thread: Optional[QtCore.QThread] = None
        self.command_thread: Optional[QtCore.QThread] = None
        self.command_worker: Optional[CommandWorker] = None
        self.restore_worker: Optional[RestoreWorker] = None
        self._build_ui()
        self._build_menu()
        self.on_acceleration_changed()
        self._update_device_label()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        save_action = QtGui.QAction("Save scan results cache...", self)
        load_action = QtGui.QAction("Load scan results cache...", self)
        clear_action = QtGui.QAction("Clear current results", self)
        exit_action = QtGui.QAction("Exit", self)
        save_action.triggered.connect(self.save_scan_cache)
        load_action.triggered.connect(self.load_scan_cache)
        clear_action.triggered.connect(self.clear_results)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(clear_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # v2.0: Use separate tabs so the results table gets almost the whole window.
        # This avoids the cramped three-section layout from earlier versions.
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # -----------------------------
        # Tab 1: setup / scan options
        # -----------------------------
        self.setup_tab = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(self.setup_tab)

        top_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top_split.setChildrenCollapsible(False)

        roots_box = QtWidgets.QGroupBox("1) Folders / drives to scan")
        roots_layout = QtWidgets.QVBoxLayout(roots_box)
        roots_layout.addWidget(QtWidgets.QLabel("Cleanup folders / drives: files here may be moved after review"))
        self.roots_list = QtWidgets.QListWidget()
        self.roots_list.setMinimumHeight(220)
        roots_layout.addWidget(self.roots_list, stretch=2)
        roots_buttons = QtWidgets.QHBoxLayout()
        self.add_folder_btn = QtWidgets.QPushButton("Add cleanup folder")
        self.add_drive_btn = QtWidgets.QPushButton("Add all drives")
        self.add_common_btn = QtWidgets.QPushButton("Add common photo/video folders")
        self.remove_root_btn = QtWidgets.QPushButton("Remove selected cleanup")
        roots_buttons.addWidget(self.add_folder_btn)
        roots_buttons.addWidget(self.add_drive_btn)
        roots_buttons.addWidget(self.add_common_btn)
        roots_buttons.addWidget(self.remove_root_btn)
        roots_layout.addLayout(roots_buttons)
        roots_layout.addSpacing(8)
        roots_layout.addWidget(QtWidgets.QLabel("Protected folders, optional: used as keep-only master copies"))
        self.protected_list = QtWidgets.QListWidget()
        self.protected_list.setMinimumHeight(140)
        roots_layout.addWidget(self.protected_list, stretch=1)
        protected_buttons = QtWidgets.QHBoxLayout()
        self.add_protected_btn = QtWidgets.QPushButton("Add protected folder")
        self.remove_protected_btn = QtWidgets.QPushButton("Remove selected protected")
        protected_buttons.addWidget(self.add_protected_btn)
        protected_buttons.addWidget(self.remove_protected_btn)
        roots_layout.addLayout(protected_buttons)
        top_split.addWidget(roots_box)

        options_box = QtWidgets.QGroupBox("2) Scan options")
        options_layout = QtWidgets.QGridLayout(options_box)
        self.recursive_cb = QtWidgets.QCheckBox("Scan all subfolders recursively")
        self.recursive_cb.setChecked(True)
        self.skip_system_cb = QtWidgets.QCheckBox("Skip Windows/system/program folders")
        self.skip_system_cb.setChecked(True)
        self.exact_cb = QtWidgets.QCheckBox("Find exact duplicates with SHA-256")
        self.exact_cb.setChecked(True)
        self.all_files_exact_cb = QtWidgets.QCheckBox("Include all file types for exact duplicates")
        self.all_files_exact_cb.setChecked(False)
        self.similar_images_cb = QtWidgets.QCheckBox("Find conservative resized/compressed photo copies")
        self.similar_images_cb.setChecked(True)
        self.possible_crops_cb = QtWidgets.QCheckBox("Find possible cropped/edited photo suggestions (review-only, unchecked)")
        self.possible_crops_cb.setChecked(True)
        self.similar_videos_cb = QtWidgets.QCheckBox("Find conservative similar videos / different resolution copies")
        self.similar_videos_cb.setChecked(True)
        self.check_similar_cb = QtWidgets.QCheckBox("Pre-check similar photo/video matches for moving (not recommended)")
        self.check_similar_cb.setChecked(False)
        self.deep_cb = QtWidgets.QCheckBox("Use AI/deep embeddings as extra confirmation only")
        self.deep_cb.setChecked(False)
        self.accel_combo = QtWidgets.QComboBox()
        self.accel_combo.addItem("Auto: use CUDA when available, otherwise CPU/fallback", "auto")
        self.accel_combo.addItem("NVIDIA desktop: require CUDA/PyTorch, AI still optional", "cuda_required")
        self.accel_combo.addItem("Laptop / CPU-only: skip PyTorch/CUDA deep scan", "hash_only")

        options_layout.addWidget(self.recursive_cb, 0, 0, 1, 2)
        options_layout.addWidget(self.skip_system_cb, 1, 0, 1, 2)
        options_layout.addWidget(self.exact_cb, 2, 0, 1, 2)
        options_layout.addWidget(self.all_files_exact_cb, 3, 0, 1, 2)
        options_layout.addWidget(self.similar_images_cb, 4, 0, 1, 2)
        options_layout.addWidget(self.possible_crops_cb, 5, 0, 1, 2)
        options_layout.addWidget(self.similar_videos_cb, 6, 0, 1, 2)
        options_layout.addWidget(self.check_similar_cb, 7, 0, 1, 2)
        options_layout.addWidget(self.deep_cb, 8, 0, 1, 2)
        options_layout.addWidget(QtWidgets.QLabel("Acceleration mode"), 9, 0)
        options_layout.addWidget(self.accel_combo, 9, 1)

        self.image_phash_spin = QtWidgets.QSpinBox()
        self.image_phash_spin.setRange(0, 64)
        self.image_phash_spin.setValue(8)
        self.image_crop_spin = QtWidgets.QSpinBox()
        self.image_crop_spin.setRange(0, 64)
        self.image_crop_spin.setValue(10)
        self.image_embed_spin = QtWidgets.QDoubleSpinBox()
        self.image_embed_spin.setRange(0.50, 0.99)
        self.image_embed_spin.setSingleStep(0.01)
        self.image_embed_spin.setDecimals(2)
        self.image_embed_spin.setValue(0.94)
        self.video_hash_spin = QtWidgets.QDoubleSpinBox()
        self.video_hash_spin.setRange(0.0, 64.0)
        self.video_hash_spin.setSingleStep(1.0)
        self.video_hash_spin.setValue(10.0)
        self.video_embed_spin = QtWidgets.QDoubleSpinBox()
        self.video_embed_spin.setRange(0.50, 0.99)
        self.video_embed_spin.setSingleStep(0.01)
        self.video_embed_spin.setDecimals(2)
        self.video_embed_spin.setValue(0.92)
        self.sample_frames_spin = QtWidgets.QSpinBox()
        self.sample_frames_spin.setRange(3, 32)
        self.sample_frames_spin.setValue(12)
        self.max_files_spin = QtWidgets.QSpinBox()
        self.max_files_spin.setRange(0, 10000000)
        self.max_files_spin.setValue(0)
        self.max_files_spin.setSpecialValueText("No cap")
        self.disk_guard_spin = QtWidgets.QDoubleSpinBox()
        self.disk_guard_spin.setRange(0.0, 500.0)
        self.disk_guard_spin.setDecimals(1)
        self.disk_guard_spin.setSingleStep(1.0)
        self.disk_guard_spin.setValue(5.0)
        self.disk_guard_spin.setSuffix(" GB")

        row = 10
        options_layout.addWidget(QtWidgets.QLabel("Photo hash threshold; higher = looser (8 conservative)"), row, 0)
        options_layout.addWidget(self.image_phash_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Photo crop threshold; higher = looser (10 conservative)"), row, 0)
        options_layout.addWidget(self.image_crop_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Photo embedding threshold; lower = looser (0.94 conservative)"), row, 0)
        options_layout.addWidget(self.image_embed_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Video frame hash threshold; higher = looser (10 conservative)"), row, 0)
        options_layout.addWidget(self.video_hash_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Video embedding threshold; lower = looser (0.92 conservative)"), row, 0)
        options_layout.addWidget(self.video_embed_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Video sample frames"), row, 0)
        options_layout.addWidget(self.sample_frames_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Max files for test scan"), row, 0)
        options_layout.addWidget(self.max_files_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("Disk guard: stop if free space below"), row, 0)
        options_layout.addWidget(self.disk_guard_spin, row, 1)

        options_layout.setRowStretch(row + 1, 1)
        top_split.addWidget(options_box)
        top_split.setSizes([620, 820])
        setup_layout.addWidget(top_split, stretch=1)
        self.tabs.addTab(self.setup_tab, "Setup / Scan")

        # -----------------------------
        # Tab 2: full-height results
        # -----------------------------
        self.results_tab = QtWidgets.QWidget()
        results_tab_layout = QtWidgets.QVBoxLayout(self.results_tab)
        results_box = QtWidgets.QGroupBox("3) Results — duplicates and similar media are separated")
        results_layout = QtWidgets.QVBoxLayout(results_box)
        count_layout = QtWidgets.QHBoxLayout()
        self.duplicate_count_label = QtWidgets.QLabel("Duplicates: 0 groups")
        self.similar_count_label = QtWidgets.QLabel("Similar media: 0 groups")
        count_layout.addWidget(self.duplicate_count_label)
        count_layout.addWidget(self.similar_count_label)
        count_layout.addStretch(1)
        results_layout.addLayout(count_layout)
        review_note = QtWidgets.QLabel("Duplicates are shown in their own tab. Similar photos/videos and crop suggestions are shown separately and are unchecked by default for safer review.")
        review_note.setWordWrap(True)
        results_layout.addWidget(review_note)

        self.result_subtabs = QtWidgets.QTabWidget()
        self.duplicates_tree = self.create_results_tree()
        self.similar_tree = self.create_results_tree()
        # Backward-compatible alias for older helper methods; most code now uses result_trees().
        self.results_tree = self.duplicates_tree
        self.result_subtabs.addTab(self.duplicates_tree, "Duplicates")
        self.result_subtabs.addTab(self.similar_tree, "Similar photos / videos")
        results_layout.addWidget(self.result_subtabs, stretch=1)

        move_layout = QtWidgets.QHBoxLayout()
        self.move_btn = QtWidgets.QPushButton("Move checked files to review folder")
        self.move_exact_btn = QtWidgets.QPushButton("Move checked exact duplicates only")
        self.check_exact_btn = QtWidgets.QPushButton("Check exact duplicate moves")
        self.check_lower_quality_similar_btn = QtWidgets.QPushButton("Check lower-quality similar files")
        self.check_lower_quality_similar_btn.setToolTip("Similar tab only. Checks resized/compressed photo or video candidates that look lower quality than the chosen KEEP file. Leaves possible cropped/edited suggestions unchecked.")
        self.uncheck_all_btn = QtWidgets.QPushButton("Uncheck all")
        self.open_selected_btn = QtWidgets.QPushButton("Open selected file")
        self.reveal_selected_btn = QtWidgets.QPushButton("Reveal selected in folder")
        self.export_csv_btn = QtWidgets.QPushButton("Export results CSV")
        self.expand_results_btn = QtWidgets.QPushButton("Go to full results tab")
        self.collapse_groups_btn = QtWidgets.QPushButton("Collapse groups")
        self.expand_groups_btn = QtWidgets.QPushButton("Expand groups")
        self.preserve_hierarchy_cb = QtWidgets.QCheckBox("Preserve folder hierarchy")
        self.preserve_hierarchy_cb.setChecked(True)
        self.preserve_hierarchy_cb.setToolTip("Move files into the review folder using their original subfolder paths instead of flattening them into group folders.")
        self.prune_empty_dirs_cb = QtWidgets.QCheckBox("Remove empty source folders")
        self.prune_empty_dirs_cb.setChecked(True)
        self.prune_empty_dirs_cb.setToolTip("After moving files, remove source folders only if they became completely empty. Cleanup roots themselves are never removed.")
        self.move_btn.setEnabled(False)
        self.move_exact_btn.setEnabled(False)
        self.check_exact_btn.setEnabled(False)
        self.check_lower_quality_similar_btn.setEnabled(False)
        self.uncheck_all_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        move_layout.addWidget(self.move_btn)
        move_layout.addWidget(self.move_exact_btn)
        move_layout.addWidget(self.check_exact_btn)
        move_layout.addWidget(self.check_lower_quality_similar_btn)
        move_layout.addWidget(self.uncheck_all_btn)
        move_layout.addWidget(self.open_selected_btn)
        move_layout.addWidget(self.reveal_selected_btn)
        move_layout.addWidget(self.export_csv_btn)
        move_layout.addWidget(self.preserve_hierarchy_cb)
        move_layout.addWidget(self.prune_empty_dirs_cb)
        move_layout.addWidget(self.expand_results_btn)
        move_layout.addWidget(self.collapse_groups_btn)
        move_layout.addWidget(self.expand_groups_btn)
        move_layout.addStretch(1)
        results_layout.addLayout(move_layout)
        results_tab_layout.addWidget(results_box, stretch=1)
        self.tabs.addTab(self.results_tab, "Results")

        # -----------------------------
        # Tab 3: log / system output
        # -----------------------------
        self.log_tab = QtWidgets.QWidget()
        log_tab_layout = QtWidgets.QVBoxLayout(self.log_tab)
        log_box = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(4000)
        self.log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log_text, stretch=1)
        log_tab_layout.addWidget(log_box, stretch=1)
        self.tabs.addTab(self.log_tab, "Log")

        # -----------------------------
        # Tab 4: standalone tools
        # -----------------------------
        self.tools_tab = QtWidgets.QWidget()
        tools_tab_layout = QtWidgets.QVBoxLayout(self.tools_tab)
        empty_box = QtWidgets.QGroupBox("Standalone empty-folder cleanup")
        empty_layout = QtWidgets.QVBoxLayout(empty_box)
        empty_help = QtWidgets.QLabel(
            "Uses the cleanup folders from Setup / Scan. This removes only folders that are empty; "
            "it never removes files and never removes the cleanup root itself. Preview first to create a CSV list before removing."
        )
        empty_help.setWordWrap(True)
        empty_layout.addWidget(empty_help)
        empty_buttons = QtWidgets.QHBoxLayout()
        self.preview_empty_folders_btn = QtWidgets.QPushButton("Preview empty folders")
        self.remove_empty_folders_btn = QtWidgets.QPushButton("Remove empty folders now")
        empty_buttons.addWidget(self.preview_empty_folders_btn)
        empty_buttons.addWidget(self.remove_empty_folders_btn)
        empty_buttons.addStretch(1)
        empty_layout.addLayout(empty_buttons)
        tools_tab_layout.addWidget(empty_box)

        restore_box = QtWidgets.QGroupBox("Undo / restore moved files")
        restore_layout = QtWidgets.QVBoxLayout(restore_box)
        restore_help = QtWidgets.QLabel(
            "Restores files using a move_manifest_*.csv created by this app. It moves files from the review folder back to their original paths, "
            "but never overwrites files that already exist at the original location."
        )
        restore_help.setWordWrap(True)
        restore_layout.addWidget(restore_help)
        self.restore_manifest_btn = QtWidgets.QPushButton("Restore moved files from manifest...")
        restore_layout.addWidget(self.restore_manifest_btn, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        tools_tab_layout.addWidget(restore_box)
        tools_tab_layout.addStretch(1)
        self.tabs.addTab(self.tools_tab, "Tools")

        # Bottom controls stay visible on every tab.
        action_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(action_layout)
        self.scan_btn = QtWidgets.QPushButton("Start scan")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.system_check_btn = QtWidgets.QPushButton("Check NVIDIA / PyTorch")
        self.preload_model_btn = QtWidgets.QPushButton("Preload ResNet50")
        self.install_cuda_btn = QtWidgets.QPushButton("Install CUDA PyTorch")
        self.install_cpu_torch_btn = QtWidgets.QPushButton("Install CPU PyTorch")
        self.device_label = QtWidgets.QLabel("")
        action_layout.addWidget(self.scan_btn)
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.system_check_btn)
        action_layout.addWidget(self.preload_model_btn)
        action_layout.addWidget(self.install_cuda_btn)
        action_layout.addWidget(self.install_cpu_torch_btn)
        action_layout.addWidget(self.device_label, stretch=1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QtWidgets.QLabel("Ready.")
        layout.addWidget(self.status_label)

        self.add_folder_btn.clicked.connect(self.add_folder)
        self.add_drive_btn.clicked.connect(self.add_all_drives)
        self.add_common_btn.clicked.connect(self.add_common_folders)
        self.remove_root_btn.clicked.connect(self.remove_selected_root)
        self.add_protected_btn.clicked.connect(self.add_protected_folder)
        self.remove_protected_btn.clicked.connect(self.remove_selected_protected)
        self.scan_btn.clicked.connect(self.start_scan)
        self.cancel_btn.clicked.connect(self.cancel_scan)
        self.system_check_btn.clicked.connect(self.show_system_check)
        self.preload_model_btn.clicked.connect(self.preload_resnet50_model)
        self.install_cuda_btn.clicked.connect(self.install_cuda_torch)
        self.install_cpu_torch_btn.clicked.connect(self.install_cpu_torch)
        self.accel_combo.currentIndexChanged.connect(self.on_acceleration_changed)
        self.move_btn.clicked.connect(self.move_checked)
        self.move_exact_btn.clicked.connect(lambda: self.move_checked(exact_only=True))
        self.check_exact_btn.clicked.connect(self.check_exact_duplicate_moves)
        self.check_lower_quality_similar_btn.clicked.connect(self.check_lower_quality_similar_moves)
        self.uncheck_all_btn.clicked.connect(self.uncheck_all_results)
        self.open_selected_btn.clicked.connect(self.open_selected)
        self.reveal_selected_btn.clicked.connect(self.reveal_selected)
        self.export_csv_btn.clicked.connect(self.export_results_csv)
        self.expand_results_btn.clicked.connect(self.expand_results_view)
        self.collapse_groups_btn.clicked.connect(self.collapse_active_result_groups)
        self.expand_groups_btn.clicked.connect(self.expand_active_result_groups)
        self.preview_empty_folders_btn.clicked.connect(self.preview_empty_folders)
        self.remove_empty_folders_btn.clicked.connect(self.remove_empty_folders_standalone)
        self.restore_manifest_btn.clicked.connect(self.restore_from_manifest)


    def expand_results_view(self) -> None:
        """Switch to the dedicated full-height results tab and restore readable columns."""
        if hasattr(self, "tabs") and hasattr(self, "results_tab"):
            self.tabs.setCurrentWidget(self.results_tab)
        self.resize_result_columns()

    def create_results_tree(self) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget()
        tree.setColumnCount(6)
        tree.setHeaderLabels(["Move?", "Type", "Score / info", "Size", "Resolution/duration", "Path"])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(18)
        tree.setUniformRowHeights(False)
        tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        tree.setMinimumHeight(520)
        tree.setStyleSheet(
            "QTreeView::item { min-height: 30px; padding: 5px 3px; } "
            "QHeaderView::section { padding: 6px 4px; font-weight: bold; }"
        )
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionsMovable(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        return tree

    def result_trees(self) -> List[QtWidgets.QTreeWidget]:
        trees: List[QtWidgets.QTreeWidget] = []
        for name in ("duplicates_tree", "similar_tree"):
            tree = getattr(self, name, None)
            if tree is not None:
                trees.append(tree)
        if not trees and hasattr(self, "results_tree"):
            trees.append(self.results_tree)
        return trees

    def active_results_tree(self) -> QtWidgets.QTreeWidget:
        if hasattr(self, "result_subtabs"):
            widget = self.result_subtabs.currentWidget()
            if isinstance(widget, QtWidgets.QTreeWidget):
                return widget
        return self.results_tree

    def collapse_active_result_groups(self) -> None:
        self.active_results_tree().collapseAll()

    def expand_active_result_groups(self) -> None:
        self.active_results_tree().expandAll()

    def resize_result_columns(self) -> None:
        """Restore readable default widths; users can still drag columns afterward."""
        widths = [80, 260, 180, 110, 190, 760]
        for tree in self.result_trees():
            for col, width in enumerate(widths):
                tree.setColumnWidth(col, width)

    def _update_device_label(self) -> None:
        mode = self.accel_combo.currentData() if hasattr(self, "accel_combo") else "auto"
        if mode == "hash_only":
            self.device_label.setText("Laptop/CPU-only mode: PyTorch/CUDA deep scan skipped.")
            return
        if torch is None:
            self.device_label.setText("PyTorch not installed: hash matching only. CUDA button can install GPU support.")
            return
        try:
            if torch.cuda.is_available():
                weights_text = "model cached" if resnet50_weights_cached() else "model not cached"
                self.device_label.setText(f"CUDA ready: {torch.cuda.get_device_name(0)} ({weights_text})")
            else:
                self.device_label.setText("PyTorch installed, but CUDA is not available. Use CPU/laptop mode or install CUDA PyTorch.")
        except Exception:
            self.device_label.setText("PyTorch installed, CUDA status unknown.")

    def on_acceleration_changed(self) -> None:
        mode = self.accel_combo.currentData()
        if mode == "hash_only":
            # CPU/laptop-safe mode always disables AI/deep embeddings.
            self.deep_cb.setChecked(False)
            self.deep_cb.setEnabled(False)
        else:
            # Auto and NVIDIA modes only decide whether CUDA is allowed/required.
            # They do NOT turn on AI matching. The user must explicitly check the AI box.
            self.deep_cb.setEnabled(True)
        self._update_device_label()

    def show_system_check(self) -> None:
        report, _nvidia_visible, torch_cuda_ok = build_system_check_report()
        self.log(report)
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("NVIDIA / PyTorch system check")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Information if torch_cuda_ok else QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText("System check complete")
        dialog.setDetailedText(report)
        if torch_cuda_ok and resnet50_weights_cached():
            info = "CUDA is ready and ResNet50 model weights are cached."
        elif torch_cuda_ok:
            info = "CUDA is ready, but ResNet50 model weights are not cached yet. Click Preload ResNet50 before the first deep scan."
        else:
            info = "CUDA is not ready for PyTorch. You can install CUDA PyTorch or choose Laptop / CPU-only mode."
        dialog.setInformativeText(info)
        dialog.exec()

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder or drive to scan")
        if folder:
            self.roots_list.addItem(folder)


    def add_protected_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose protected/master folder")
        if folder:
            self.protected_list.addItem(folder)

    def add_all_drives(self) -> None:
        existing = {self.roots_list.item(i).text() for i in range(self.roots_list.count())}
        added = 0
        if platform.system().lower() == "windows":
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive) and drive not in existing:
                    self.roots_list.addItem(drive)
                    added += 1
        else:
            for root in ["/", str(Path.home())]:
                if os.path.exists(root) and root not in existing:
                    self.roots_list.addItem(root)
                    added += 1
        self.log(f"Added {added} drive/root locations.")

    def add_common_folders(self) -> None:
        home = Path.home()
        candidates = [home / "Pictures", home / "Videos", home / "Downloads", home / "Desktop", home / "OneDrive" / "Pictures", home / "OneDrive" / "Videos"]
        existing = {self.roots_list.item(i).text() for i in range(self.roots_list.count())}
        added = 0
        for p in candidates:
            if p.exists() and str(p) not in existing:
                self.roots_list.addItem(str(p))
                added += 1
        self.log(f"Added {added} common folders.")

    def remove_selected_root(self) -> None:
        for item in self.roots_list.selectedItems():
            row = self.roots_list.row(item)
            self.roots_list.takeItem(row)

    def remove_selected_protected(self) -> None:
        for item in self.protected_list.selectedItems():
            row = self.protected_list.row(item)
            self.protected_list.takeItem(row)

    def read_settings(self) -> ScanSettings:
        roots = [self.roots_list.item(i).text() for i in range(self.roots_list.count())]
        protected_roots = [self.protected_list.item(i).text() for i in range(self.protected_list.count())]
        return ScanSettings(
            roots=roots,
            protected_roots=protected_roots,
            include_all_files_for_exact=self.all_files_exact_cb.isChecked(),
            scan_exact=self.exact_cb.isChecked(),
            scan_similar_images=self.similar_images_cb.isChecked(),
            scan_similar_videos=self.similar_videos_cb.isChecked(),
            recursive=self.recursive_cb.isChecked(),
            include_possible_crops=self.possible_crops_cb.isChecked(),
            check_similar_by_default=self.check_similar_cb.isChecked(),
            use_deep_embeddings=self.deep_cb.isChecked() and self.accel_combo.currentData() != "hash_only",
            acceleration_mode=str(self.accel_combo.currentData() or "auto"),
            skip_system_dirs=self.skip_system_cb.isChecked(),
            image_phash_threshold=self.image_phash_spin.value(),
            image_crop_threshold=self.image_crop_spin.value(),
            image_embedding_threshold=self.image_embed_spin.value(),
            video_frame_phash_threshold=self.video_hash_spin.value(),
            video_embedding_threshold=self.video_embed_spin.value(),
            video_sample_frames=self.sample_frames_spin.value(),
            max_files=self.max_files_spin.value(),
            min_free_space_gb=self.disk_guard_spin.value(),
        )

    def set_busy(self, busy: bool) -> None:
        self.scan_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.add_folder_btn.setEnabled(not busy)
        self.add_drive_btn.setEnabled(not busy)
        self.add_common_btn.setEnabled(not busy)
        self.remove_root_btn.setEnabled(not busy)
        self.add_protected_btn.setEnabled(not busy)
        self.remove_protected_btn.setEnabled(not busy)
        self.system_check_btn.setEnabled(not busy)
        self.preload_model_btn.setEnabled(not busy)
        self.install_cuda_btn.setEnabled(not busy)
        self.install_cpu_torch_btn.setEnabled(not busy)
        self.accel_combo.setEnabled(not busy)
        self.preserve_hierarchy_cb.setEnabled(not busy)
        self.prune_empty_dirs_cb.setEnabled(not busy)
        if hasattr(self, "move_exact_btn"):
            self.move_exact_btn.setEnabled((not busy) and bool(self.groups))
        if hasattr(self, "check_exact_btn"):
            self.check_exact_btn.setEnabled((not busy) and bool(self.groups))
        if hasattr(self, "check_lower_quality_similar_btn"):
            self.check_lower_quality_similar_btn.setEnabled((not busy) and any(self.is_similar_group(g) for g in self.groups))
        if hasattr(self, "uncheck_all_btn"):
            self.uncheck_all_btn.setEnabled((not busy) and bool(self.groups))
        if hasattr(self, "preview_empty_folders_btn"):
            self.preview_empty_folders_btn.setEnabled(not busy)
        if hasattr(self, "remove_empty_folders_btn"):
            self.remove_empty_folders_btn.setEnabled(not busy)
        if hasattr(self, "restore_manifest_btn"):
            self.restore_manifest_btn.setEnabled(not busy)

    def preview_empty_folders(self) -> None:
        self.run_empty_folder_cleanup(dry_run=True)

    def remove_empty_folders_standalone(self) -> None:
        self.run_empty_folder_cleanup(dry_run=False)

    def run_empty_folder_cleanup(self, dry_run: bool) -> None:
        roots = self.current_cleanup_roots()
        if not roots:
            QtWidgets.QMessageBox.warning(self, "No folders", "Add at least one cleanup folder or drive first.")
            return
        if not dry_run:
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Remove empty folders?",
                "This will remove empty folders under your cleanup folders as a standalone action.\n\n"
                "No files will be removed. Selected cleanup roots themselves will not be removed.\n"
                "A CSV manifest will be saved in your user folder.\n\n"
                "Continue?",
            )
            if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        else:
            self.log("Previewing empty folders. No folders will be removed.")
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Finding empty folders...")
        self.empty_thread = QtCore.QThread(self)
        self.empty_worker = EmptyFolderCleanupWorker(
            roots,
            skip_system_dirs=self.skip_system_cb.isChecked(),
            dry_run=dry_run,
        )
        self.empty_worker.moveToThread(self.empty_thread)
        self.empty_thread.started.connect(self.empty_worker.run)
        self.empty_worker.progress.connect(self.on_empty_folder_progress)
        self.empty_worker.finished.connect(self.on_empty_folder_finished)
        self.empty_worker.failed.connect(self.on_empty_folder_failed)
        self.empty_worker.finished.connect(self.empty_thread.quit)
        self.empty_worker.failed.connect(self.empty_thread.quit)
        self.empty_thread.finished.connect(self.empty_worker.deleteLater)
        self.empty_thread.finished.connect(self.empty_thread.deleteLater)
        self.empty_thread.start()

    @QtCore.Slot(int, int, str)
    def on_empty_folder_progress(self, value: int, maximum: int, message: str) -> None:
        if maximum <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, maximum)
            self.progress.setValue(value)
        self.status_label.setText(message)

    @QtCore.Slot(str, int, bool)
    def on_empty_folder_finished(self, manifest: str, count: int, dry_run: bool) -> None:
        self.set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if dry_run:
            msg = f"Preview complete. Found {count} empty folders. Manifest: {manifest}"
            box_title = "Preview complete"
            box_msg = f"Found {count} empty folders. No folders were removed.\n\nCSV manifest:\n{manifest}"
        else:
            msg = f"Standalone empty-folder cleanup complete. Removed {count} empty folders. Manifest: {manifest}"
            box_title = "Cleanup complete"
            box_msg = f"Removed {count} empty folders.\n\nCSV manifest:\n{manifest}"
        self.status_label.setText(msg)
        self.log(msg)
        QtWidgets.QMessageBox.information(self, box_title, box_msg)

    @QtCore.Slot(str)
    def on_empty_folder_failed(self, error: str) -> None:
        self.set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Empty-folder cleanup failed.")
        self.log(error)
        QtWidgets.QMessageBox.critical(self, "Empty-folder cleanup failed", error[-4000:])

    def restore_from_manifest(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose move manifest to restore from",
            "",
            "Move manifest CSV (move_manifest_*.csv);;CSV Files (*.csv);;All files (*.*)",
        )
        if not filename:
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Restore moved files?",
            "This reads the selected move manifest and moves files from the review folder back to their original paths.\n\n"
            "It will not overwrite files that already exist at the original location. A restore manifest will be created beside the selected manifest.\n\n"
            f"Manifest:\n{filename}\n\nContinue?",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Restoring moved files from manifest...")
        self.restore_thread = QtCore.QThread(self)
        self.restore_worker = RestoreWorker(filename)
        self.restore_worker.moveToThread(self.restore_thread)
        self.restore_thread.started.connect(self.restore_worker.run)
        self.restore_worker.progress.connect(self.on_restore_progress)
        self.restore_worker.finished.connect(self.on_restore_finished)
        self.restore_worker.failed.connect(self.on_restore_failed)
        self.restore_worker.finished.connect(self.restore_thread.quit)
        self.restore_worker.failed.connect(self.restore_thread.quit)
        self.restore_thread.finished.connect(self.restore_worker.deleteLater)
        self.restore_thread.finished.connect(self.restore_thread.deleteLater)
        self.restore_thread.start()

    @QtCore.Slot(int, int, str)
    def on_restore_progress(self, value: int, maximum: int, message: str) -> None:
        self.progress.setRange(0, maximum)
        self.progress.setValue(value)
        self.status_label.setText(message)

    @QtCore.Slot(str, int, int)
    def on_restore_finished(self, manifest: str, restored: int, skipped: int) -> None:
        self.set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        msg = f"Restore complete. Restored {restored} file(s), skipped {skipped}. Manifest: {manifest}"
        self.status_label.setText(msg)
        self.log(msg)
        QtWidgets.QMessageBox.information(self, "Restore complete", msg)

    @QtCore.Slot(str)
    def on_restore_failed(self, error: str) -> None:
        self.set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Restore failed.")
        self.log(error)
        QtWidgets.QMessageBox.critical(self, "Restore failed", error[-4000:])

    def start_scan(self) -> None:
        settings = self.read_settings()
        if not settings.roots:
            QtWidgets.QMessageBox.warning(self, "No folders", "Add at least one folder or drive to scan.")
            return
        if settings.acceleration_mode == "cuda_required":
            _report, _nvidia_visible, torch_cuda_ok = build_system_check_report()
            if not torch_cuda_ok:
                choice = QtWidgets.QMessageBox.question(
                    self,
                    "CUDA not ready",
                    "NVIDIA desktop mode requires PyTorch CUDA, but it is not ready. This mode does not enable AI matching by itself.\n\n"
                    "Choose Yes to switch to Laptop / CPU-only mode and continue.\n"
                    "Choose No to cancel so you can install CUDA PyTorch first.",
                )
                if choice == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.accel_combo.setCurrentIndex(self.accel_combo.findData("hash_only"))
                    settings = self.read_settings()
                else:
                    self.show_system_check()
                    return
        if settings.use_deep_embeddings and settings.acceleration_mode != "hash_only" and torch is not None and not resnet50_weights_cached():
            choice = QtWidgets.QMessageBox.question(
                self,
                "Model weights not cached",
                "PyTorch is installed, but the ResNet50 model weights are not cached yet.\n\n"
                "If you continue, the first deep scan may download the model file.\n"
                "Choose Yes to continue, or No to cancel and click Preload ResNet50 first.",
            )
            if choice != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.results_tree.clear()
        self.groups = []
        self.records = {}
        self.move_btn.setEnabled(False)
        self.move_exact_btn.setEnabled(False)
        self.check_exact_btn.setEnabled(False)
        self.check_lower_quality_similar_btn.setEnabled(False)
        self.uncheck_all_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.log_text.clear()
        self.progress.setRange(0, 0)
        self.status_label.setText("Starting scan...")
        self.set_busy(True)

        self.scan_thread = QtCore.QThread(self)
        self.scan_worker = ScannerWorker(settings)
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.on_scan_progress)
        self.scan_worker.log.connect(self.log)
        self.scan_worker.finished.connect(self.on_scan_finished)
        self.scan_worker.failed.connect(self.on_scan_failed)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.start()

    def cancel_scan(self) -> None:
        if self.scan_worker:
            self.scan_worker.cancel()
            self.status_label.setText("Cancel requested...")

    @QtCore.Slot(int, int, str)
    def on_scan_progress(self, value: int, maximum: int, message: str) -> None:
        if maximum <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, maximum)
            self.progress.setValue(value)
        self.status_label.setText(message)

    @QtCore.Slot(list, dict, str)
    def on_scan_finished(self, groups: List[ResultGroup], records: Dict[str, FileRecord], device_info: str) -> None:
        self.groups = groups
        self.records = records
        self.populate_results()
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status_label.setText(f"Scan complete: {len(groups)} groups found. {device_info}")
        self.log(f"Scan complete. Groups found: {len(groups)}")
        self.expand_results_view()
        self.set_busy(False)
        self.move_btn.setEnabled(bool(groups))
        self.move_exact_btn.setEnabled(bool(groups))
        self.check_exact_btn.setEnabled(bool(groups))
        self.check_lower_quality_similar_btn.setEnabled(any(self.is_similar_group(g) for g in groups))
        self.uncheck_all_btn.setEnabled(bool(groups))
        self.export_csv_btn.setEnabled(bool(groups))
        self._update_device_label()

    @QtCore.Slot(str)
    def on_scan_failed(self, error: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Scan failed or canceled.")
        self.log(error)
        self.set_busy(False)
        QtWidgets.QMessageBox.critical(self, "Scan failed", error[-4000:])

    def file_info_text(self, path: str) -> Tuple[str, str]:
        rec = self.records.get(path)
        size = human_bytes(rec.size) if rec else ""
        detail = ""
        if rec:
            if rec.media_type == "image" and rec.width and rec.height:
                detail = f"{rec.width}×{rec.height}"
            elif rec.media_type == "video":
                dims = f"{rec.width}×{rec.height}" if rec.width and rec.height else "unknown res"
                dur = f", {rec.duration:.1f}s" if rec.duration else ""
                detail = dims + dur
        return size, detail

    def is_exact_group(self, group: ResultGroup) -> bool:
        return group.group_type.lower().startswith("exact duplicate")

    def is_similar_group(self, group: ResultGroup) -> bool:
        return not self.is_exact_group(group)

    def add_group_to_tree(self, tree: QtWidgets.QTreeWidget, group: ResultGroup, index: int, total_groups: int) -> None:
        parent = QtWidgets.QTreeWidgetItem(["", f"Group {index}/{total_groups}: {group.group_type}", group.score, "", f"Keep: {Path(group.keep_path).name}", group.notes])
        parent.setSizeHint(0, QtCore.QSize(0, 32))
        parent.setFirstColumnSpanned(False)
        parent.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "group", "group_id": group.group_id, "group_type": group.group_type})
        font = parent.font(1)
        font.setBold(True)
        for col in range(6):
            parent.setFont(col, font)
        tree.addTopLevelItem(parent)
        for path in group.files:
            size, detail = self.file_info_text(path)
            rec = self.records.get(path)
            protected = bool(rec.protected) if rec else False
            action_label = "PROTECTED KEEP" if protected else ("KEEP" if path == group.keep_path else "MOVE")
            item = QtWidgets.QTreeWidgetItem(["", action_label, "", size, detail, display_path(path)])
            item.setSizeHint(0, QtCore.QSize(0, 30))
            item.setToolTip(5, display_path(path))
            is_keeper = path == group.keep_path
            base_flags = item.flags() | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
            if not is_keeper and not protected:
                base_flags = base_flags | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(base_flags)
            default_checked = bool(group.move_checked_by_default) and not is_keeper and not protected
            item.setCheckState(0, QtCore.Qt.CheckState.Checked if default_checked else QtCore.Qt.CheckState.Unchecked)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"kind": "file", "group_id": group.group_id, "group_type": group.group_type, "path": path})
            if is_keeper or protected:
                item.setToolTip(0, "Hard safety keeper: this file is not movable by this app.")
                brush = QtGui.QBrush(QtGui.QColor("#0b6b0b"))
                for c in range(6):
                    item.setForeground(c, brush)
            parent.addChild(item)
        parent.setExpanded(index <= 10 or total_groups <= 50)

    def populate_results(self) -> None:
        for tree in self.result_trees():
            tree.clear()
        duplicate_groups = [g for g in self.groups if self.is_exact_group(g)]
        similar_groups = [g for g in self.groups if self.is_similar_group(g)]

        if hasattr(self, "duplicate_count_label"):
            self.duplicate_count_label.setText(f"Duplicates: {len(duplicate_groups)} groups")
        if hasattr(self, "similar_count_label"):
            self.similar_count_label.setText(f"Similar media: {len(similar_groups)} groups")

        for index, group in enumerate(duplicate_groups, start=1):
            self.add_group_to_tree(self.duplicates_tree, group, index, len(duplicate_groups))
        for index, group in enumerate(similar_groups, start=1):
            self.add_group_to_tree(self.similar_tree, group, index, len(similar_groups))

        if hasattr(self, "result_subtabs"):
            self.result_subtabs.setTabText(0, f"Duplicates ({len(duplicate_groups)})")
            self.result_subtabs.setTabText(1, f"Similar photos / videos ({len(similar_groups)})")
            # Prefer the duplicate tab after scan, but automatically show Similar if no duplicates exist.
            self.result_subtabs.setCurrentIndex(0 if duplicate_groups else 1)
        self.resize_result_columns()

    def current_cleanup_roots(self) -> List[str]:
        return [self.roots_list.item(i).text() for i in range(self.roots_list.count())]

    def group_keep_map(self) -> Dict[int, str]:
        """Return the hard keep file for every result group.

        Safety rule: a scan group must always retain at least one file. The chosen
        keeper is never moved by the app, even if a checkbox state is changed by a
        UI quirk or a future code change.
        """
        return {int(g.group_id): str(g.keep_path) for g in self.groups if g.keep_path}

    def checked_moves(self, exact_only: bool = False) -> List[Tuple[int, str, str]]:
        moves: List[Tuple[int, str, str]] = []
        skipped_keepers = 0
        keep_map = self.group_keep_map()
        for tree in self.result_trees():
            root_count = tree.topLevelItemCount()
            for i in range(root_count):
                parent = tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.checkState(0) == QtCore.Qt.CheckState.Checked:
                        data = child.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
                        path = data.get("path")
                        group_id = int(data.get("group_id", 0))
                        group_type = str(data.get("group_type", "group"))
                        if exact_only and not group_type.lower().startswith("exact duplicate"):
                            continue
                        if not path:
                            continue
                        rec = self.records.get(path)
                        if rec and rec.protected:
                            continue
                        # Hard safety guard: never move the selected keeper for a group.
                        # This prevents accidentally moving every copy/version in a group.
                        if same_file_path(path, keep_map.get(group_id, "")):
                            skipped_keepers += 1
                            continue
                        moves.append((group_id, group_type, path))
        if skipped_keepers:
            self.log(f"Keeper safety guard skipped {skipped_keepers} checked KEEP file(s).")
        return moves

    def uncheck_all_results(self) -> None:
        for tree in self.result_trees():
            for i in range(tree.topLevelItemCount()):
                parent = tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                        child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        self.status_label.setText("Unchecked all movable result rows.")

    def check_exact_duplicate_moves(self) -> None:
        checked = 0
        for tree in self.result_trees():
            for i in range(tree.topLevelItemCount()):
                parent = tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    data = child.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
                    group_type = str(data.get("group_type", "")).lower()
                    if child.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                        if group_type.startswith("exact duplicate"):
                            child.setCheckState(0, QtCore.Qt.CheckState.Checked)
                            checked += 1
                        else:
                            child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        if hasattr(self, "result_subtabs"):
            self.result_subtabs.setCurrentWidget(self.duplicates_tree)
        self.status_label.setText(f"Checked {checked} exact-duplicate move rows and unchecked non-exact rows.")

    def quality_score(self, path: str) -> Tuple[int, int, int, int, int, float]:
        """Score a media file so larger/better originals are preferred over lower-quality copies.

        Tuple order is intentionally simple and conservative:
        protected copy, pixel area, video duration, file size, known metadata count, modified time.
        It is used only to pre-check review rows; the green KEEP guard still prevents moving all copies.
        """
        rec = self.records.get(path)
        st = safe_stat(path)
        mtime = st.st_mtime if st else 0.0
        size = st.st_size if st else (rec.size if rec else 0)
        protected = 1 if rec and rec.protected else 0
        area = int((rec.width or 0) * (rec.height or 0)) if rec else 0
        duration_ms = int((rec.duration or 0.0) * 1000) if rec else 0
        known = 0
        if rec:
            known += 1 if rec.width and rec.height else 0
            known += 1 if rec.duration else 0
            known += 1 if rec.size else 0
        return (protected, area, duration_ms, int(size), known, float(mtime))

    def is_lower_quality_than_keeper(self, candidate_path: str, keeper_path: str) -> bool:
        candidate = self.records.get(candidate_path)
        keeper = self.records.get(keeper_path)
        if not candidate or not keeper:
            return False
        # Protected files are never auto-checked, even when they look lower quality.
        if candidate.protected:
            return False
        # Images: lower resolution is lower quality. Same resolution with a clearly smaller
        # file is treated as a likely compressed/exported copy.
        if candidate.media_type == "image" and keeper.media_type == "image":
            c_area = int((candidate.width or 0) * (candidate.height or 0))
            k_area = int((keeper.width or 0) * (keeper.height or 0))
            if c_area and k_area:
                if c_area < k_area:
                    return True
                if c_area == k_area and candidate.size < int(keeper.size * 0.92):
                    return True
                return False
            return candidate.size < int(keeper.size * 0.92)
        # Videos: lower resolution is lower quality. Same resolution with similar duration
        # and a clearly smaller file is likely lower bitrate/re-encoded.
        if candidate.media_type == "video" and keeper.media_type == "video":
            c_area = int((candidate.width or 0) * (candidate.height or 0))
            k_area = int((keeper.width or 0) * (keeper.height or 0))
            if c_area and k_area and c_area < k_area:
                return True
            durations_close = True
            if candidate.duration and keeper.duration:
                durations_close = abs(candidate.duration - keeper.duration) <= max(2.0, keeper.duration * 0.03)
            if c_area == k_area and durations_close and candidate.size < int(keeper.size * 0.90):
                return True
            return False
        return False

    def check_lower_quality_similar_moves(self) -> None:
        if not hasattr(self, "similar_tree"):
            return
        checked = 0
        skipped_crop_groups = 0
        skipped_not_lower = 0
        keep_map = self.group_keep_map()

        # Only operate on the Similar tab. Duplicates stay untouched.
        tree = self.similar_tree
        for i in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(i)
            parent_data = parent.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
            group_id = int(parent_data.get("group_id", 0))
            group_type = str(parent_data.get("group_type", "")).lower()
            keeper = keep_map.get(group_id, "")

            # Crops/edits are risky because the crop may be the version the user wants.
            # Leave those unchecked for manual review.
            if "cropped" in group_type or "edited" in group_type or "crop" in group_type:
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable:
                        child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                skipped_crop_groups += 1
                continue

            for j in range(parent.childCount()):
                child = parent.child(j)
                if not (child.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable):
                    continue
                data = child.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
                path = data.get("path")
                if path and keeper and self.is_lower_quality_than_keeper(path, keeper):
                    child.setCheckState(0, QtCore.Qt.CheckState.Checked)
                    checked += 1
                else:
                    child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                    skipped_not_lower += 1

        if hasattr(self, "tabs") and hasattr(self, "results_tab"):
            self.tabs.setCurrentWidget(self.results_tab)
        if hasattr(self, "result_subtabs"):
            self.result_subtabs.setCurrentWidget(self.similar_tree)
        self.status_label.setText(
            f"Checked {checked} lower-quality similar rows. "
            f"Skipped {skipped_crop_groups} crop/edit group(s) for manual review."
        )

    def move_checked(self, checked: bool = False, exact_only: bool = False) -> None:
        # PySide clicked(bool) can pass a checked flag; exact_only is only true when
        # called by the exact-only button's lambda.
        moves = self.checked_moves(exact_only=exact_only)
        if not moves:
            message = "Check exact duplicate files first, or use the 'Check exact duplicate moves' button." if exact_only else "Check the files you want moved first."
            QtWidgets.QMessageBox.information(self, "Nothing checked", message)
            return
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose review folder to move checked files into")
        if not dest:
            return
        scanned_roots = self.current_cleanup_roots() + [self.protected_list.item(i).text() for i in range(self.protected_list.count())]
        inside = [root for root in scanned_roots if is_subpath(dest, root)]
        if inside:
            QtWidgets.QMessageBox.warning(
                self,
                "Choose a different review folder",
                "The review folder cannot be inside a folder that is being scanned:\n"
                + "\n".join(inside)
                + "\n\nFiles moved there would be found again by the next scan. "
                "Please choose a folder outside the scanned folders.",
            )
            return
        preserve_text = "preserving the original folder hierarchy" if self.preserve_hierarchy_cb.isChecked() else "using group folders"
        prune_text = "Empty source folders will be removed only if they become completely empty." if self.prune_empty_dirs_cb.isChecked() else "Empty source folders will be left in place."
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Move checked files?",
            f"Move {len(moves)} checked {'exact duplicate ' if exact_only else ''}files into:\n{dest}\n\n"
            f"Files will be moved {preserve_text}.\n"
            f"{prune_text}\n\n"
            "Hard safety rule: the green KEEP file in each group will never be moved, so one copy/version stays behind.\n\n"
            "This does not delete files, but it will move them from their current folders.",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.set_busy(True)
        self.move_btn.setEnabled(False)
        self.progress.setRange(0, len(moves))
        self.progress.setValue(0)
        self.status_label.setText("Moving files...")

        self.move_thread = QtCore.QThread(self)
        self.move_worker = MoveWorker(
            moves,
            dest,
            cleanup_roots=self.current_cleanup_roots(),
            preserve_hierarchy=self.preserve_hierarchy_cb.isChecked(),
            prune_empty_dirs=self.prune_empty_dirs_cb.isChecked(),
        )
        self.move_worker.moveToThread(self.move_thread)
        self.move_thread.started.connect(self.move_worker.run)
        self.move_worker.progress.connect(self.on_move_progress)
        self.move_worker.finished.connect(self.on_move_finished)
        self.move_worker.failed.connect(self.on_move_failed)
        self.move_worker.finished.connect(self.move_thread.quit)
        self.move_worker.failed.connect(self.move_thread.quit)
        self.move_thread.finished.connect(self.move_worker.deleteLater)
        self.move_thread.finished.connect(self.move_thread.deleteLater)
        self.move_thread.start()

    @QtCore.Slot(int, int, str)
    def on_move_progress(self, value: int, maximum: int, message: str) -> None:
        self.progress.setRange(0, maximum)
        self.progress.setValue(value)
        self.status_label.setText(message)

    @QtCore.Slot(str, int)
    def on_move_finished(self, manifest: str, failed_count: int = 0) -> None:
        self.set_busy(False)
        self.status_label.setText(f"Move complete. Manifest: {manifest}")
        self.log(f"Move complete. Manifest: {manifest}")
        if failed_count:
            self.log(f"{failed_count} file(s) were not moved because the moved copy did not match the original. The originals were kept.")
            QtWidgets.QMessageBox.warning(
                self,
                "Move complete with problems",
                f"{failed_count} file(s) were NOT moved because the copy did not match the original "
                "(size or SHA-256 check failed). The originals were left where they were.\n\n"
                f"See rows marked verify_failed_kept_source in the manifest:\n{manifest}",
            )
            return
        QtWidgets.QMessageBox.information(self, "Move complete", f"Files moved with the selected folder layout. Manifest saved at:\n{manifest}")

    @QtCore.Slot(str)
    def on_move_failed(self, error: str) -> None:
        self.set_busy(False)
        self.log(error)
        QtWidgets.QMessageBox.critical(self, "Move failed", error[-4000:])

    def selected_file_path(self) -> Optional[str]:
        item = self.active_results_tree().currentItem()
        if not item:
            return None
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        return data.get("path")

    def open_selected(self) -> None:
        path = self.selected_file_path()
        if not path:
            QtWidgets.QMessageBox.information(self, "No file selected", "Select a file row first.")
            return
        try:
            path_display = display_path(path)
            if platform.system().lower() == "windows":
                os.startfile(path_display)  # type: ignore[attr-defined]
            elif platform.system().lower() == "darwin":
                subprocess.Popen(["open", path_display])
            else:
                subprocess.Popen(["xdg-open", path_display])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not open", str(exc))

    def reveal_selected(self) -> None:
        path = self.selected_file_path()
        if not path:
            QtWidgets.QMessageBox.information(self, "No file selected", "Select a file row first.")
            return
        reveal_in_file_manager(path)

    def clear_results(self) -> None:
        self.groups = []
        self.records = {}
        for tree in self.result_trees():
            tree.clear()
        self.move_btn.setEnabled(False)
        self.move_exact_btn.setEnabled(False)
        self.check_exact_btn.setEnabled(False)
        self.check_lower_quality_similar_btn.setEnabled(False)
        self.uncheck_all_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.status_label.setText("Results cleared.")
        self.log("Results cleared.")

    def save_scan_cache(self) -> None:
        if not self.groups and not self.records:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run or load a scan first.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save scan results cache",
            "media_duplicate_scan.media-cache.json.gz",
            "Media scan cache (*.media-cache.json.gz);;JSON gzip (*.json.gz)",
        )
        if not filename:
            return
        data = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "cleanup_roots": self.current_cleanup_roots(),
            "protected_roots": [self.protected_list.item(i).text() for i in range(self.protected_list.count())],
            "groups": [group_to_cache_dict(g) for g in self.groups],
            "records": [record_to_cache_dict(r) for r in self.records.values()],
        }
        try:
            with gzip.open(filename, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "Cache saved", f"Saved scan cache:\n{filename}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def load_scan_cache(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load scan results cache",
            "",
            "Media scan cache (*.media-cache.json.gz *.json.gz);;All files (*.*)",
        )
        if not filename:
            return
        try:
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                data = json.load(f)
            cleanup_roots = [str(x) for x in data.get("cleanup_roots", []) if x]
            protected_roots = [str(x) for x in data.get("protected_roots", []) if x]
            if cleanup_roots:
                self.roots_list.clear()
                self.roots_list.addItems(cleanup_roots)
            if protected_roots:
                self.protected_list.clear()
                self.protected_list.addItems(protected_roots)
            self.records = {r.path: r for r in (record_from_cache_dict(x) for x in data.get("records", [])) if r.path}
            self.groups = [group_from_cache_dict(x) for x in data.get("groups", [])]
            self.populate_results()
            self.move_btn.setEnabled(bool(self.groups))
            self.move_exact_btn.setEnabled(bool(self.groups))
            self.check_exact_btn.setEnabled(bool(self.groups))
            self.uncheck_all_btn.setEnabled(bool(self.groups))
            self.export_csv_btn.setEnabled(bool(self.groups))
            self.status_label.setText(f"Loaded cache: {len(self.groups)} groups.")
            self.log(f"Loaded cache from {filename}. Groups: {len(self.groups)}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

    def export_results_csv(self) -> None:
        if not self.groups:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save results CSV", "duplicate_results.csv", "CSV Files (*.csv)")
        if not filename:
            return
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["group_id", "group_type", "score", "recommended_keep", "move_checked_default", "size", "details", "path", "notes"])
                for group in self.groups:
                    for path in group.files:
                        size, detail = self.file_info_text(path)
                        writer.writerow([group.group_id, group.group_type, group.score, display_path(group.keep_path), bool(group.move_checked_by_default) and path != group.keep_path, size, detail, display_path(path), group.notes])
            QtWidgets.QMessageBox.information(self, "Export complete", f"Saved:\n{filename}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt name)
        if self.command_thread is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Please wait",
                "An install or download is still running. Please wait for it to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _start_command(self, command: Sequence[str], busy_text: str, on_done) -> None:
        """Run a command on a background QThread. on_done(return_code or None, error_text) runs on the GUI thread."""
        if self.command_thread is not None:
            QtWidgets.QMessageBox.information(self, "Please wait", "Another install or download is still running.")
            return
        self.set_busy(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status_label.setText(busy_text)
        self.log("Running: " + " ".join(command))

        thread = QtCore.QThread(self)
        worker = CommandWorker(command)
        worker.moveToThread(thread)
        self.command_thread = thread
        self.command_worker = worker

        def clear_refs() -> None:
            self.command_thread = None
            self.command_worker = None

        def finish(code: Optional[int], error: str) -> None:
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if code == 0 else 0)
            self.set_busy(False)
            on_done(code, error)

        thread.started.connect(worker.run)
        worker.output.connect(self.log)
        worker.finished.connect(lambda code: finish(int(code), ""))
        worker.failed.connect(lambda err: finish(None, str(err)))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(clear_refs)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _run_pip_install(self, packages: Sequence[str], index_url: str, label: str) -> None:
        command = [sys.executable, "-m", "pip", "install", *packages, "--index-url", index_url]

        def done(code: Optional[int], error: str) -> None:
            if code == 0:
                self.status_label.setText(f"{label} install finished. Close and reopen the app.")
                QtWidgets.QMessageBox.information(
                    self,
                    "Install finished",
                    f"{label} install finished. Close and reopen this app so Python can reload PyTorch.",
                )
            else:
                self.status_label.setText(f"{label} install failed. See the log.")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Install failed",
                    error or f"pip stopped with exit code {code}. See the log for details.",
                )

        self._start_command(command, f"Installing {label}. This can take a long time. Watch the log...", done)

    def preload_resnet50_model(self) -> None:
        if torch is None or models is None:
            QtWidgets.QMessageBox.warning(
                self,
                "PyTorch not loaded",
                "PyTorch/torchvision is not loaded in this Python. Install CUDA PyTorch or CPU PyTorch first, then close and reopen the app.",
            )
            return
        path = resnet50_weights_cache_path()
        if path and path.exists():
            QtWidgets.QMessageBox.information(self, "Already cached", f"ResNet50 model weights are already cached:\n{path}")
            self._update_device_label()
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Preload ResNet50 model?",
            "This downloads the ResNet50 model weights used by deep image/video similarity.\n\n"
            "Doing this now prevents the first real scan from starting a surprise download.\n\n"
            f"Expected cache location:\n{path or 'Torch cache/checkpoints folder'}",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.log("Preloading ResNet50 model weights...")
        code = (
            "from torchvision import models\n"
            "models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n"
            "print('ResNet50 weights are ready.')\n"
        )

        def done(return_code: Optional[int], error: str) -> None:
            if return_code == 0:
                self.status_label.setText("ResNet50 model weights are ready.")
                self.log(resnet50_weights_summary())
                self._update_device_label()
                QtWidgets.QMessageBox.information(self, "Preload complete", resnet50_weights_summary())
            else:
                self.status_label.setText("ResNet50 preload failed.")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Preload failed",
                    error or f"The download stopped with exit code {return_code}. See the log for details.",
                )

        self._start_command(
            [sys.executable, "-u", "-c", code],
            "Preloading ResNet50 model weights (about 100 MB). Watch the log...",
            done,
        )

    def install_cuda_torch(self) -> None:
        report, nvidia_visible, torch_cuda_ok = build_system_check_report()
        if torch_cuda_ok and resnet50_weights_cached():
            QtWidgets.QMessageBox.information(self, "CUDA already ready", "PyTorch CUDA and the ResNet50 model cache already appear ready.\n\n" + report)
            return
        if torch_cuda_ok and not resnet50_weights_cached():
            QtWidgets.QMessageBox.information(self, "CUDA ready", "PyTorch CUDA is installed. The only missing piece is the ResNet50 model cache. Click Preload ResNet50 to download it before scanning.\n\n" + report)
            return
        msg = (
            "This will install PyTorch with CUDA 12.8 wheels using pip.\n\n"
            "This is a large download (about 2.5 GB) and can take a long time. "
            "Close and reopen the app after it finishes.\n\n"
            "Use this on a computer with an NVIDIA graphics card. For a computer without NVIDIA CUDA, choose Laptop / CPU-only mode instead.\n\n"
            "Command:\n"
            f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {PYTORCH_CUDA_INDEX_URL}"
        )
        if not nvidia_visible:
            msg += "\n\nWarning: nvidia-smi did not show an NVIDIA GPU/driver. Installing CUDA PyTorch may not help on this computer."
        confirm = QtWidgets.QMessageBox.question(self, "Install CUDA PyTorch?", msg)
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run_pip_install(["torch", "torchvision", "torchaudio"], PYTORCH_CUDA_INDEX_URL, "CUDA PyTorch")

    def install_cpu_torch(self) -> None:
        msg = (
            "This installs CPU-only PyTorch. It is useful for laptops or computers without NVIDIA CUDA.\n\n"
            "CPU PyTorch can still run deep embeddings, but it may be slow. The Laptop / CPU-only mode skips deep embeddings entirely.\n\n"
            "This is a download of several hundred MB. Close and reopen the app after it finishes.\n\n"
            "Command:\n"
            f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {PYTORCH_CPU_INDEX_URL}"
        )
        confirm = QtWidgets.QMessageBox.question(self, "Install CPU PyTorch?", msg)
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._run_pip_install(["torch", "torchvision", "torchaudio"], PYTORCH_CPU_INDEX_URL, "CPU PyTorch")

# -----------------------------
# Entry point
# -----------------------------


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Personal Tools")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
