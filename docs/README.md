# facegate

Face detection, recognition and access decisions on an Orange Pi Zero 3W
(Allwinner A733, VIPLite v2.0 NPU), with detection and embedding both running on
the NPU. Also runs on the Allwinner T527 (Orange Pi 4A) with its own `.nb` graphs.

The operator view is a web UI in the browser: live camera view with annotated
faces, enrolment, the access log, and the tunables. `facegate` runs on the
board, needs no cloud service, and streams over your LAN.

```
webcam  ──►  YuNet detect (CPU)        faces + 5 landmarks
              │
              ├─ similarity transform → 112x112
              │
              ├─ SFace embed (NPU)      128-d vector, ~15 ms
              │
              ├─ cosine match vs enrolled database
              │
              └─ MiniFASNet (CPU)       live or spoofed?
                        │
              GRANT ───┴─── DENY / DENY_SPOOF
```

## What it does

- **Detects** faces with YuNet and tracks their 5 landmarks.
- **Recognises** them by embedding each face into a 128-d vector and matching
  it against an enrolled database with cosine similarity.
- **Rejects spoofs** with MiniFASNet before granting access, because a
  photograph held to the lens matches a face database perfectly.
- **Fails closed.** An unrecognised face is denied, not waved through.

## Requirements

- Orange Pi Zero 3W (Allwinner A733) or Orange Pi 4A (Allwinner T527), or another aarch64 Linux board
- Python 3.10+
- A USB webcam (resolved by USB vendor/product id, not by `/dev/videoN`)
- Optional: the board NPU, for faster detection and embedding (A733 `.nb` graphs are included)

## Quick start

```bash
git clone <this-repo>
cd facegate
./install.sh          # venv, dependencies, and the three ONNX models
./enroll.sh           # build the face database from photos in faces/<name>/
./run.sh              # start recognition + the web UI
```

Open `http://<board-ip>:8080`.

## Documentation

| Document | What is in it |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | Full install, camera setup, boot service |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every setting, and which ones matter |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pipeline works and why |
| [docs/NPU.md](docs/NPU.md) | Running detection and embedding on the NPU |
| [docs/API.md](docs/API.md) | HTTP API, usable with `curl` |
| [docs/SECURITY.md](docs/SECURITY.md) | Threats, limits, and honest caveats |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Measured numbers and how to reproduce them |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems and what causes them |

## Measured performance

Orange Pi 4A, 8x Cortex-A55 @ 1.8 GHz, 4 GB RAM, 640x480 MJPG webcam.

| | |
|---|---|
| Face embedding (NPU) | **~15 ms** |
| Face embedding (CPU fallback) | ~68 ms |
| Face detection (YuNet, 256px) | ~17 ms |
| Liveness (only on faces that would be granted) | ~17 ms |
| Steady state, face in view | **~23 ms/frame = 43 fps** |
| CPU per frame | ~133 ms |

Full numbers and the methodology are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Models

| Model | Role | Source |
|---|---|---|
| YuNet | face detection + landmarks | OpenCV Zoo |
| SFace | 128-d face embedding | OpenCV Zoo |
| MiniFASNet V2 SE | live / spoof classification | facenox/face-antispoof-onnx |

Both the detector and the embedder also ship as Vivante `.nb` graphs for the
A733 NPU (`models/npu/yunet.nb`, `models/npu/sface_int8.nb`); they are committed.
For other targets (e.g. T527) they must be rebuilt - see [docs/NPU.md](docs/NPU.md).

## Security note, up front

This is a single-board hobby-grade access system on a trusted LAN. It has **no
authentication on its web interface**, and its anti-spoofing is **not validated
against real print or screen attacks**. Read
[docs/SECURITY.md](docs/SECURITY.md) before putting it on a real door.

## Licence

MIT. The models are redistributed by their upstream projects under their own
terms; see [docs/INSTALL.md](docs/INSTALL.md).
