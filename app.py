import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

from counter import (DEFAULT_VEHICLES, class_ids_for, draw_lines, load_model,
                     process_video, read_first_frame)

HERE = Path(__file__).parent
WEIGHTS = HERE / "yolo11n.pt"
SAMPLE = HERE / "3.mp4"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Vehicle Detection & Counting", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Detection, Tracking & Counting")
st.caption(
    "Upload a traffic video. YOLO11 detects and classifies vehicles, ByteTrack gives each one "
    "a unique ID, and two virtual lines count them by class and direction."
)


@st.cache_data(show_spinner=False)
def coco_names():
    return list(load_model(WEIGHTS if WEIGHTS.exists() else "yolo11n.pt").names.values())


# ---------- 1. choose a video ----------
uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
use_sample = False
if uploaded is None and SAMPLE.exists():
    use_sample = st.button("▶ No video handy? Try the sample traffic video")
    if use_sample:
        st.session_state["use_sample"] = True
if uploaded is not None:
    st.session_state["use_sample"] = False

if uploaded is not None:
    source_id, source_label = f"up:{uploaded.name}:{uploaded.size}", uploaded.name
elif st.session_state.get("use_sample"):
    source_id, source_label = "sample", "3.mp4 (sample)"
else:
    st.info("Upload a video (mp4, mov, avi, mkv) to get started.")
    st.stop()

# new source -> reset the line positions to sensible defaults
if st.session_state.get("source_id") != source_id:
    st.session_state["source_id"] = source_id
    st.session_state["lines"] = (55, 74) if source_id == "sample" else (50, 65)
    st.session_state.pop("result", None)

# write the video to disk once per source (OpenCV needs a file path)
if st.session_state.get("video_path_for") != source_id:
    old = st.session_state.get("video_path")
    if old and old != str(SAMPLE) and os.path.exists(old):
        os.remove(old)
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        st.session_state["video_path"] = tmp.name
    else:
        st.session_state["video_path"] = str(SAMPLE)
    st.session_state["video_path_for"] = source_id
video_path = st.session_state["video_path"]

# ---------- 2. settings ----------
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05,
                     help="Higher = fewer false detections, but may miss small or distant vehicles.")

    detect_all = st.checkbox("Detect all 80 object types", value=False,
                             help="Off = vehicles only (best for counting traffic).")
    names = coco_names()
    if detect_all:
        chosen = names
        st.caption("Detecting everything the model knows (people, signs, animals, ...).")
    else:
        chosen = st.multiselect("Classes to detect & count", names,
                                default=[n for n in DEFAULT_VEHICLES if n in names])

    red_pct, blue_pct = st.slider(
        "Line positions (% of frame height)", 5, 95, key="lines",
        help="Left handle = red line, right handle = blue line. Place them across the road.")

    max_width = st.select_slider("Processing width (px)", options=[640, 960, 1280, 1920], value=960,
                                 help="Smaller = faster. The output video uses this size.")
    max_seconds = st.slider("Max seconds to process", 5, 300, 60, 5,
                            help="Limits run time, especially on CPU.")
    st.caption(f"Running on: **{'GPU' if DEVICE != 'cpu' else 'CPU'}**")

# ---------- 3. preview with lines ----------
frame0, fps, total_frames = read_first_frame(video_path, max_width)
if frame0 is None:
    st.error("Could not read this video. Try another file or format.")
    st.stop()

h = frame0.shape[0]
red_y, blue_y = int(red_pct / 100 * h), int(blue_pct / 100 * h)

left, right = st.columns([3, 2])
with left:
    st.subheader("1. Place the counting lines")
    preview = draw_lines(frame0.copy(), red_y, blue_y)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width="stretch")
    dur = total_frames / fps if total_frames else 0
    st.caption(f"{source_label} · {frame0.shape[1]}x{h} · {fps:.0f} fps"
               + (f" · {dur:.0f}s" if dur else ""))
    st.caption("Vehicles that cross **red → blue** are counted as *Down*; **blue → red** as *Up*.")

with right:
    st.subheader("2. Run")
    if not chosen:
        st.warning("Pick at least one class in the sidebar.")
    if dur and dur > max_seconds:
        st.warning(f"Video is {dur:.0f}s; only the first {max_seconds}s will be processed.")
    run = st.button("Detect & count", type="primary", disabled=not chosen, width="stretch")
    status = st.empty()
    bar = st.empty()
    live = st.empty()

# ---------- 4. run ----------
if run:
    model = load_model(WEIGHTS if WEIGHTS.exists() else "yolo11n.pt")  # fresh model = fresh tracker state
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    progress = bar.progress(0.0)

    def on_progress(i, total, frame):
        progress.progress(min(i / total, 1.0) if total else 0.0)
        status.write(f"Processing frame {i}" + (f" / {total}" if total else ""))
        live.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")

    try:
        stats = process_video(
            video_path, out_path, model, conf=conf,
            red_frac=red_pct / 100, blue_frac=blue_pct / 100,
            class_ids=class_ids_for(model, chosen), max_seconds=max_seconds,
            max_width=max_width, device=DEVICE, progress_cb=on_progress,
        )
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.session_state["result"] = {"stats": stats, "video": video_bytes, "source": source_id}
    except Exception as e:  # show a friendly message instead of a stack trace
        st.error(f"Processing failed: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    status.empty(); bar.empty(); live.empty()

# ---------- 5. results ----------
res = st.session_state.get("result")
if res and res["source"] == source_id:
    stats = res["stats"]
    st.divider()
    st.subheader("Results")

    classes = sorted(set(stats["down"]) | set(stats["up"]))
    df = pd.DataFrame(
        [{"Class": c, "Down": stats["down"].get(c, 0), "Up": stats["up"].get(c, 0),
          "Total": stats["down"].get(c, 0) + stats["up"].get(c, 0)} for c in classes],
        columns=["Class", "Down", "Up", "Total"],
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total counted", int(df["Total"].sum()) if not df.empty else 0)
    m2.metric("Down", int(df["Down"].sum()) if not df.empty else 0)
    m3.metric("Up", int(df["Up"].sum()) if not df.empty else 0)
    m4.metric("Frames processed", f"{stats['frames']} ({stats['proc_fps']:.1f} fps)")

    v, t = st.columns([3, 2])
    with v:
        st.video(res["video"])
        st.download_button("⬇ Download annotated video", res["video"],
                           file_name="vehicle_count_output.mp4", mime="video/mp4")
    with t:
        if df.empty:
            st.info("No vehicles crossed both lines. Try moving the lines across the road, "
                    "lowering the confidence, or processing a longer clip.")
        else:
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption("Total counted per class")
            st.bar_chart(df.set_index("Class")["Total"])
            st.download_button("⬇ Download counts (CSV)", df.to_csv(index=False),
                               file_name="vehicle_counts.csv", mime="text/csv")
