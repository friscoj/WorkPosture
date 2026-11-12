# WorkPosture

WorkPosture watches a webcam and sends a desktop notification when your head stays forward of your shoulders. It runs on a Linux desktop.

Camera frames are used for that check and then discarded. The program does not save video, open a network connection, or upload anything. The pose model is a local file.

## Requirements

- Python 3.13
- A webcam, with your user allowed to open it (the `video` group on Debian)
- A desktop session, so `notify-send` can display the alert (`libnotify-bin`)
- PortAudio, which `sounddevice` uses to play the alert (`libportaudio2`)

On Debian or Ubuntu:

```bash
sudo apt install python3 python3-venv libnotify-bin libportaudio2
```

## Setup

From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

`pose_landmarker_lite.task` is Google's MediaPipe pose model, licensed under Apache-2.0. It is downloaded separately and is not part of this repository. See `NOTICE`.

## Calibrate, then run

The alert threshold is an angle in degrees, and it depends on how you sit in front of this camera. The default is 30. Read your own upright angle before trusting it:

```bash
.venv/bin/python main.py --debug
```

Sit upright and note the printed angle. Press `q` in the camera window to quit. Start the monitor with a threshold a little above that upright reading:

```bash
.venv/bin/python main.py --threshold 50
```

Stop it with Ctrl+C.

An alert fires after the angle stays past the threshold for 5 seconds, and then at most once every 30 seconds. `--help` lists `--camera`, `--hysteresis`, `--persist`, `--cooldown`, and `--interval`.

## Alert config

`config.json` sets the sound, the notification icon, and the message:

```json
{
  "sound": "sound.wav",
  "picture": "posture.png",
  "message": "Fix your posture!"
}
```

Paths are relative to `config.json` unless they are absolute. `sound` must be 16-bit PCM WAV. Both files have to exist when the program starts.

`posture.png` and `sound.wav` in this repository are original works and are covered by the project license.

## License

The code, `posture.png`, and `sound.wav` are released under the MIT license. See `LICENSE`.

The pose model is a separate Apache-2.0 work from Google. See `NOTICE`.
