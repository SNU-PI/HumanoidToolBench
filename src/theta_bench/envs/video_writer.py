"""
THETA(θ)-Bench

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """Path of an ffmpeg binary, or None.

    Prefers the system ``ffmpeg`` on PATH; falls back to the static binary
    bundled by the optional ``imageio-ffmpeg`` package, which needs no root.
    Cached because a writer is constructed per camera per episode.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoWriter:
    """Encode RGB frames straight to an H.264 mp4 in a single pass.

    Frames are piped as raw RGB into one ``ffmpeg`` process that runs alongside
    the simulator, so the episode ends with a finished ``.mp4`` and no
    re-encode. ``close()`` preserves the filename; ``release(success)`` closes
    the encoder and renames the file to ``<name>_success.mp4`` / ``<name>_failed.mp4``.

    The encoder is pinned to one thread: many eval workers run concurrently on
    a shared node and must not each fan out across every core.

    ``ffmpeg`` is required (system PATH or the ``imageio-ffmpeg`` package);
    constructing a writer without it raises ``RuntimeError``.
    """

    def __init__(self, filename, framerate, resolution, write_png=False):
        # resolution = (w, h), matching cv2.VideoWriter's convention.
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if os.path.exists(filename):
            print("remove existing file:", filename)
            os.unlink(filename)

        self.resize = False
        width, height = int(resolution[0]), int(resolution[1])
        if self.resize:
            width, height = width // 2, height // 2
        self.resolution = (width, height)
        self.framerate = framerate
        self.filename = filename
        self.write_png = write_png
        self.frame_idx = 0

        self._closed = False
        self._discarded = False

        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required to record videos but was not found on PATH. "
                "Install it (`apt install ffmpeg`, or `pip install imageio-ffmpeg` "
                "for a static binary without root), or run with --no-save-video."
            )
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(framerate),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                # yuv420p is what browsers decode; x264 needs even dimensions.
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                filename,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # ------------------------------------------------------------------ frames
    def write(self, image):
        """Append one RGB uint8 frame of shape (H, W, 3)."""
        assert image.dtype == np.uint8, f"expected uint8 frames, got {image.dtype}"
        if self.resize:
            import cv2

            h, w = image.shape[:2]
            image = cv2.resize(image, (w // 2, h // 2))

        width, height = self.resolution
        if image.shape[:2] != (height, width) or image.shape[-1] != 3:
            raise ValueError(
                f"{self.filename}: frame shape {image.shape} does not match "
                f"writer resolution (H={height}, W={width}, 3)"
            )

        try:
            self._proc.stdin.write(memoryview(np.ascontiguousarray(image)).cast("B"))  # type: ignore[union-attr]
        except BrokenPipeError:
            raise RuntimeError(
                f"ffmpeg exited while writing {self.filename}: {self._drain_stderr()}"
            ) from None

        if self.write_png:
            import cv2

            cv2.imwrite(
                f"{self.filename}_{self.frame_idx:03d}.png",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        self.frame_idx += 1

    # ----------------------------------------------------------------- closing
    def _drain_stderr(self) -> str:
        try:
            _, stderr = self._proc.communicate(timeout=30)
            return (stderr or b"").decode(errors="replace").strip()
        except Exception:
            return ""

    def _close_encoder(self) -> None:
        if self._closed:
            return
        _, stderr = self._proc.communicate(timeout=30)
        rc = self._proc.returncode
        if rc != 0:
            err = (stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg exited with code {rc} for {self.filename}: {err}"
            )
        self._closed = True

    def close(self) -> None:
        """Finish encoding without changing the output filename."""
        self._close_encoder()

    def discard(self) -> None:
        """Stop encoding and delete the partial file (e.g. when restarting a recording)."""
        self._discarded = True
        error = None
        try:
            if not self._closed:
                if self._proc.poll() is None:
                    self._proc.kill()
                self._proc.communicate(timeout=5)
                self._closed = True
        except BaseException as exc:
            error = exc
        try:
            if os.path.exists(self.filename):
                os.remove(self.filename)
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error

    def release(self, success=True):
        """Finish the file and rename it to ``<name>_success.mp4`` / ``<name>_failed.mp4``."""
        if self._discarded:
            return
        self._close_encoder()
        if not os.path.exists(self.filename):
            return  # already renamed by an earlier release(), or discarded

        suffix = "success" if success else "failed"
        newfilename = f"{self.filename[:-4]}_{suffix}.mp4"
        if os.path.exists(newfilename):
            print(f"remove existing file: {newfilename}")
            os.remove(newfilename)
        os.rename(self.filename, newfilename)

    def __del__(self):
        # Never leave an encoder process behind if the writer is dropped.
        try:
            if getattr(self, "_proc", None) is not None and not self._closed:
                self.discard()
        except Exception:
            pass
