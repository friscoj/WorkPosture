import argparse
import json
import math
import os
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
CONFIG_PATH = ROOT / "config.json"
ALERT_TITLE = "Posture Alert"

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
SIDES = ((LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER))

# World-landmark distances below this are too small to trust the axis sign.
MIN_SHOULDER_HEIGHT_M = 0.05
MIN_NOSE_DEPTH_M = 0.02
MIN_EAR_ABOVE_SHOULDER_M = 0.02
MIN_VISIBILITY = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a webcam and alert when your head drifts forward of your shoulders."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the camera window and print the live angle. Press q to quit.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help=(
            "Forward-head angle, in degrees, that counts as a slouch. "
            "Sit upright with --debug, read the angle, and set this a little above it. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=6.0,
        help="Degrees the angle must fall back under the threshold before the timer resets. Default: 6.",
    )
    parser.add_argument(
        "--persist",
        type=float,
        default=5.0,
        help="Seconds the slouch must last before an alert. Default: 5.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=30.0,
        help="Minimum seconds between alerts. Default: 30.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between samples. Default: 1.",
    )
    args = parser.parse_args()
    if args.threshold <= 0:
        parser.error("--threshold must be positive")
    if not 0 <= args.hysteresis < args.threshold:
        parser.error("--hysteresis must be at least 0 and less than --threshold")
    if args.persist < 0 or args.cooldown < 0:
        parser.error("--persist and --cooldown must be at least 0")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


@contextmanager
def silenced_native_stderr():
    """Hide LiteRT's one-time startup log, which ignores Python log levels."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def resolve_config_path(base, value, label):
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"config {label} must be a file path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"config {label} not found: {path}")
    return path


def load_config(path):
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Missing config file: {path}") from None
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    missing = [key for key in ("sound", "picture", "message") if key not in data]
    if missing:
        raise SystemExit(f"{path} is missing {', '.join(missing)}")
    message = data["message"]
    if not isinstance(message, str):
        raise SystemExit(f"{path} message must be a string")
    base = path.parent
    return (
        resolve_config_path(base, data["sound"], "sound"),
        resolve_config_path(base, data["picture"], "picture"),
        message,
    )


def load_sound(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be 16-bit PCM audio")
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return np.ascontiguousarray(samples), sample_rate


def landmark_visible(landmark):
    if landmark.x is None or landmark.y is None or landmark.z is None:
        return False
    if landmark.visibility is not None and landmark.visibility < MIN_VISIBILITY:
        return False
    if landmark.presence is not None and landmark.presence < MIN_VISIBILITY:
        return False
    return True


def forward_angle(image_landmarks, world_landmarks):
    """Degrees the ears sit forward of the shoulders, 0 when stacked above them.

    World axes are signed from the pose itself: shoulders sit above the hip
    origin, and the nose sits closer to the camera than the shoulders.
    """
    if not all(landmark_visible(image_landmarks[index]) for index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)):
        return None

    shoulder_y = (world_landmarks[LEFT_SHOULDER].y + world_landmarks[RIGHT_SHOULDER].y) / 2
    shoulder_z = (world_landmarks[LEFT_SHOULDER].z + world_landmarks[RIGHT_SHOULDER].z) / 2
    if abs(shoulder_y) < MIN_SHOULDER_HEIGHT_M:
        return None
    depth = world_landmarks[NOSE].z - shoulder_z
    if abs(depth) < MIN_NOSE_DEPTH_M:
        return None
    y_up = 1.0 if shoulder_y > 0 else -1.0
    z_toward_camera = 1.0 if depth > 0 else -1.0

    angles = []
    for ear_index, shoulder_index in SIDES:
        if not landmark_visible(image_landmarks[ear_index]):
            continue
        ear = world_landmarks[ear_index]
        shoulder = world_landmarks[shoulder_index]
        up = (ear.y - shoulder.y) * y_up
        forward = (ear.z - shoulder.z) * z_toward_camera
        if up <= MIN_EAR_ABOVE_SHOULDER_M:
            continue
        angles.append(math.degrees(math.atan2(forward, up)))
    if not angles:
        return None
    return sum(angles) / len(angles)


def show_notification(samples, sample_rate, picture, message):
    try:
        subprocess.run(
            [
                "notify-send",
                ALERT_TITLE,
                message,
                "-i",
                str(picture),
                "-t",
                "5000",
            ],
            check=False,
        )
    except OSError as error:
        print(f"Notification failed: {error}", file=sys.stderr)
    try:
        sd.play(samples, sample_rate)
    except sd.PortAudioError as error:
        print(f"Sound failed: {error}", file=sys.stderr)


def draw_debug(frame, image_landmarks, angle, threshold, slouching):
    if image_landmarks is not None:
        height, width = frame.shape[:2]
        for index in (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER):
            landmark = image_landmarks[index]
            if not landmark_visible(landmark):
                continue
            center = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, center, 5, (0, 220, 0), -1)
    if angle is None:
        label = "no pose"
    else:
        label = f"{angle:.1f} deg   threshold {threshold:.0f}"
    color = (0, 0, 255) if slouching else (0, 200, 0)
    cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imshow("Posture Monitor", frame)


def open_camera(index):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def monitor(landmarker, samples, sample_rate, picture, message, args):
    camera = open_camera(args.camera)
    bad_since = None
    last_alert = None
    first_frame = True
    try:
        while camera.isOpened():
            started = time.monotonic()
            # One discard covers drivers that ignore the buffer-size request.
            camera.grab()
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames", file=sys.stderr)
                break

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # The first call logs harmless LiteRT warnings straight to the C stderr.
            if first_frame:
                with silenced_native_stderr():
                    result = landmarker.detect(image)
                first_frame = False
            else:
                result = landmarker.detect(image)
            image_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            angle = (
                forward_angle(image_landmarks, world_landmarks)
                if image_landmarks is not None and world_landmarks is not None
                else None
            )

            slouching = False
            if angle is None:
                bad_since = None
            else:
                limit = args.threshold if bad_since is None else args.threshold - args.hysteresis
                if angle > limit:
                    slouching = True
                    now = time.monotonic()
                    if bad_since is None:
                        bad_since = now
                    cooled_down = last_alert is None or now - last_alert >= args.cooldown
                    if now - bad_since >= args.persist and cooled_down:
                        show_notification(samples, sample_rate, picture, message)
                        last_alert = now
                        print(f"Posture alert at {angle:.1f} deg")
                else:
                    bad_since = None

            if args.debug:
                held = 0 if bad_since is None else time.monotonic() - bad_since
                angle_text = "none" if angle is None else f"{angle:.1f}"
                print(f"angle {angle_text} deg  slouch {slouching}  held {held:.1f}s")
                draw_debug(frame, image_landmarks, angle, args.threshold, slouching)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Missing pose model: {MODEL_PATH}\n"
            "Download the MediaPipe pose weights (Apache-2.0) with:\n"
            f"curl -L -o {MODEL_PATH.name} {MODEL_URL}"
        )
    sound_path, picture_path, message = load_config(CONFIG_PATH)
    samples, sample_rate = load_sound(sound_path)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    print(
        f"Watching camera {args.camera}. "
        f"Alert when the forward-head angle stays above {args.threshold:.0f} deg. "
        "Run with --debug to read your upright angle."
    )
    landmarker = None
    try:
        with silenced_native_stderr():
            landmarker = PoseLandmarker.create_from_options(options)
        try:
            monitor(landmarker, samples, sample_rate, picture_path, message, args)
        finally:
            if landmarker is not None:
                with silenced_native_stderr():
                    landmarker.close()
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
