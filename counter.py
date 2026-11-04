"""Core detection / tracking / line-crossing logic for the vehicle counter.

This is the notebook pipeline (YOLO11 + ByteTrack + red/blue virtual lines)
moved into a reusable module so the Streamlit app can call it.
"""
import os
import subprocess
import time
from collections import defaultdict

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

DEFAULT_VEHICLES = ["bicycle", "car", "motorcycle", "bus", "truck"]

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


def load_model(weights) -> YOLO:
    """Load a YOLO11 model (a new instance per run gives fresh tracker state)."""
    return YOLO(str(weights))


def _even(x: int) -> int:
    """H.264 (yuv420p) needs even width/height."""
    return max(2, x - (x % 2))


def target_size(width: int, height: int, max_width: int):
    """Downscale to max_width (keeping aspect ratio); max_width=0 keeps the original."""
    if max_width and width > max_width:
        scale = max_width / width
        width, height = int(width * scale), int(height * scale)
    return _even(width), _even(height)


def read_first_frame(video_path: str, max_width: int = 0):
    """Return (frame_bgr, fps, total_frames) or (None, 0, 0) if unreadable."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not ok:
        return None, 0.0, 0
    w, h = target_size(frame.shape[1], frame.shape[0], max_width)
    return cv2.resize(frame, (w, h)), fps, total


def draw_lines(frame, red_y: int, blue_y: int):
    """Draw the red and blue counting lines across the full frame width."""
    w = frame.shape[1]
    cv2.line(frame, (0, red_y), (w, red_y), RED, 3)
    cv2.putText(frame, "Red Line", (10, red_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (0, blue_y), (w, blue_y), BLUE, 3)
    cv2.putText(frame, "Blue Line", (10, blue_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class LineCounter:
    """Direction-based counting with two horizontal lines (red above blue).

    Down = crosses red then blue (moving down the screen).
    Up   = crosses blue then red (moving up the screen).

    Unlike a +/-5 px "is the centre on the line" check, this compares the
    previous and current centre position, so fast vehicles that jump over a
    line between two frames are still counted, and each track is counted once.
    """

    def __init__(self, red_y: int, blue_y: int):
        self.red_y, self.blue_y = red_y, blue_y
        self.prev_cy = {}
        self.armed_down, self.armed_up = set(), set()
        self.counted_down, self.counted_up = set(), set()
        self.down = defaultdict(int)
        self.up = defaultdict(int)

    def update(self, track_id: int, class_name: str, cy: int):
        prev = self.prev_cy.get(track_id)
        self.prev_cy[track_id] = cy
        if prev is None:
            return
        # moving down: red first, then blue
        if prev < self.red_y <= cy:
            self.armed_down.add(track_id)
        if prev < self.blue_y <= cy and track_id in self.armed_down and track_id not in self.counted_down:
            self.counted_down.add(track_id)
            self.down[class_name] += 1
        # moving up: blue first, then red
        if prev > self.blue_y >= cy:
            self.armed_up.add(track_id)
        if prev > self.red_y >= cy and track_id in self.armed_up and track_id not in self.counted_up:
            self.counted_up.add(track_id)
            self.up[class_name] += 1


def class_ids_for(model: YOLO, names):
    """Map class names to COCO ids for the loaded model."""
    wanted = set(names)
    return [i for i, n in model.names.items() if n in wanted]


def process_video(
    video_path: str,
    out_path: str,
    model: YOLO,
    conf: float = 0.25,
    red_frac: float = 0.55,
    blue_frac: float = 0.74,
    class_ids=None,
    max_seconds: float = 0,
    max_width: int = 960,
    device="cpu",
    progress_cb=None,
):
    """Detect, track and count vehicles; write an annotated H.264 video to out_path.

    progress_cb(frame_index, total_frames, annotated_frame_bgr) is called regularly.
    Returns a dict with per-class 'down' / 'up' counts and run statistics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = target_size(src_w, src_h, max_width)

    max_frames = int(max_seconds * fps) if max_seconds else 0
    if max_frames and total:
        total = min(total, max_frames)
    elif max_frames:
        total = max_frames

    red_y, blue_y = int(red_frac * height), int(blue_frac * height)
    counter = LineCounter(red_y, blue_y)

    raw_path = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    start = time.time()
    n = 0
    while cap.isOpened():
        if max_frames and n >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            classes=class_ids or None,
            device=device,
            verbose=False,
        )
        boxes = results[0].boxes
        draw_lines(frame, red_y, blue_y)

        # boxes.id is None on frames where nothing is being tracked
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            for (x1, y1, x2, y2), tid, ci in zip(xyxy, ids, clss):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = model.names[ci]
                counter.update(tid, name, cy)
                cv2.circle(frame, (cx, cy), 4, RED, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, f"ID: {tid} {name}", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

        # running counts on the frame (same style as the notebook)
        y_off = 30
        for name, c in counter.down.items():
            cv2.putText(frame, f"{name} (Down): {c}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
            y_off += 30
        y_off += 20
        for name, c in counter.up.items():
            cv2.putText(frame, f"{name} (Up): {c}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2, cv2.LINE_AA)
            y_off += 30

        writer.write(frame)
        n += 1
        if progress_cb and (n % 5 == 0 or n == 1):
            progress_cb(n, total, frame)

    cap.release()
    writer.release()
    elapsed = time.time() - start

    # Re-encode to H.264 so the video plays in browsers (mp4v does not).
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", out_path],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg re-encode failed: " + proc.stderr.decode(errors="ignore")[-400:])

    os.remove(raw_path)

    return {
        "down": dict(counter.down),
        "up": dict(counter.up),
        "frames": n,
        "seconds": elapsed,
        "proc_fps": n / elapsed if elapsed else 0.0,
        "size": (width, height),
    }
