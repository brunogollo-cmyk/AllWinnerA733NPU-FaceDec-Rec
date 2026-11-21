# Using the A733/T527 NPU for face embedding and detection

The Orange Pi Zero 3W (Allwinner A733, VeriSilicon NPU, VIPLite v2.0) and Orange
Pi 4A (Allwinner T527, VIP9000) can run face workloads on the NPU. `facegate` uses
it for the SFace embedding (the most expensive step) and for YuNet face detection.

**The A733 builds of both graphs are committed** (`models/npu/yunet.nb` and
`models/npu/sface_int8.nb`, target `VIP9000NANODI_PLUS_PID0X1000003B`), so both
functions can run on the NPU on an Orange Pi Zero 3W without rebuilding anything.

The userspace runtime is included as well (`tools/npu_runtime/A733/`:
`libNBGlinker.so` + `libVIPhal.so`); install it with
`sudo ./tools/install_npu_runtime.sh` if the board does not already ship it.
The kernel module `vipcore` (which creates `/dev/vipcore`) is bundled too, but
installed opt-in and only when its `vermagic` matches the running kernel:
`sudo ./tools/install_npu_runtime.sh --with-kernel-module`. Most vendor images
already provide it. Check the whole path with `./.venv/bin/python tools/check_npu.py`.

| | CPU | NPU (A733 int8) |
|---|---|---|
| SFace embed | ~68 ms | **~2 ms per graph run** |
| YuNet detect (640x640 graph) | ~17 ms @256px | ~2.4 ms per graph run |
| Agrees with the CPU path | — | cosine **≈0.98** (embedding) |

Embedding is on by default (`recognizer.use_npu: true`); detection is selected
with `detector.use_npu`. Set `use_npu: false` to go back to the CPU; the enrolled
database stays valid either way.

**Per-function CPU/NPU selector:** the web UI (Live tab and Settings) lets you
choose CPU or NPU independently for detection and recognition. The choice is saved
to `config.yaml` and applied on the next service restart (models/graphs are loaded
at startup). If the selected NPU path is unavailable, it falls back to CPU.

## How it works

The NPU does not run ONNX. It runs Vivante **`.nb` network binaries**, so the
SFace model was converted offline:

```
face_recognition_sface_2021dec.onnx     37 MB float32, 9.67M params
        │  ACUITY: pegasus import onnx
        ▼
   sface.json + sface.data
        │  pegasus quantize   perchannel_symmetric_affine, int8, KL divergence
        ▼
   sface.quantize                          activation ranges
        │  pegasus export ovxlib --pack-nbg-unify
        ▼
   network_binary.nb                       7.9 MB, runs on VIP9000
```

The compile target is `VIP9000NANOSI_PLUS_PID0X10000016`, which is this board's
chip id — the runtime reports `cid=0x10000016`. The wrong target produces a
`.nb` the driver silently refuses to run.

At load time, `facegate/npu.py` reads the graph's own buffer descriptors
(shape, data format, quantisation parameters) rather than hardcoding them, so a
re-converted model with different dimensions works without code changes.

## Requirements

- A kernel driver exposing `/dev/vipcore` (module `vipcore` loaded).
- The VIPLite runtime libraries.

**Orange Pi Zero 3W (A733, AI SDK v2.0, recommended):** the runtime is installed
system-wide. Point `detector.viplite_dir` / `recognizer.viplite_dir` to `/usr/lib`
(default). The loader looks for:
- `libNBGlinker.so` + `libVIPhal.so` (A733 / v2.0), or
- `libVIPlite.so` + `libVIPuser.so` (T527 / v1.x fallback).

**T527 / SDK v1.x:**

```bash
git clone --depth 1 -b product-aiot-stable \
    https://gitlab.com/tina5.0_aiot/ai-sdk.git
```

Use the `glibc-gcc10_2_0` build for v1.x (glibc 2.17+). Verify `/dev/vipcore`
exists.

## Configuration

```yaml
detector:
  use_npu: false                 # true = detecção na NPU (requer yunet.nb convertido)
  npu_model: "models/npu/yunet.nb"
  viplite_dir: "/usr/lib"        # A733 (AI SDK v2.0); em T527 use o caminho do SDK v1.x

recognizer:
  use_npu: true                  # embedding na NPU (requer sface_int8.nb)
  npu_model: "models/npu/sface_int8.nb"
  viplite_dir: "/usr/lib"
```

Os seletores **CPU/NPU** da interface web alteram `detector.use_npu` e
`recognizer.use_npu` via `POST /api/config` e exigem reinício do serviço para
carregar o backend escolhido. O endpoint `GET /api/status` expõe `detect_backend`,
`recognizer_backend` e `npu_available`.

## Verifying it is actually running

```bash
./run.sh
```

At startup the log should say:

```
INFO facegate.npu: NPU graph loaded: sface_int8.nb driver=0x00010d00 cid=0x10000016 in=75264B ...
INFO facegate.recognizer: Face embedding on NPU (sface_int8.nb), ~15 ms vs ~68 ms on CPU
```

The Live tab's HUD shows `emb` per frame. On a fresh face you should see a
non-zero value around 15; on reused frames it shows 0 by design.

There is also a standalone tool that prints one embedding, which is the easiest
way to A/B the graph against the CPU model:

```bash
gcc -O2 -I<viplite>/inc -o /tmp/npu_embed tools/npu_embed.c \
    -L<viplite> -lVIPlite -lVIPuser -lm

# float16 NCHW 1x3x112x112, RGB
LD_LIBRARY_PATH=<viplite> /tmp/npu_embed \
    models/npu/sface_int8.nb input.dat
```

## Rebuilding the graph

The converter only runs inside an x86_64 Docker image that Allwinner
distributes, so this is done on another machine and the result copied over.
The full recipe — including the exact commands and the traps — is in
[`../tools/README-npu.md`](../tools/README-npu.md).

Two things that cost real time if you do not know them:

- **Use `v1.8.x` for the T527.** The `v2.0.x` image is for the A733. Mixing
  them produces a graph the driver rejects.
- **Calibrate with real faces.** Activation ranges come from the calibration
  set; random noise quantises badly and quietly degrades the embedding.

## Accuracy

Quantising to int8 does cost some precision. Measured on 20 real camera crops:

```
cosine(NPU, CPU) = 0.9916   (min 0.9879, max 0.9946)
```

More importantly, in a live run against the enrolled database the NPU and CPU
paths produced the **same identity decision on 70/70 frames**, with mean
pairwise cosine 0.9998. That is why switching `use_npu` does not invalidate
`data/facedb.json` — **nobody has to re-enrol**.

If you re-enrol anyway (for example after changing the lighting at the door),
do it with the backend you intend to run. Both directions are safe; mixing
databases across a *model* change is not.

## When it falls back to the CPU

`facegate` never fails because the NPU is unavailable. It logs and continues:

```
NPU backend unavailable (<reason>) - using CPU
```

The reasons it will fall back:

- `use_npu: false` in the config
- the `.nb` file is missing, or `viplite_dir` is wrong
- `libVIPlite.so` / `libVIPuser.so` not found
- `/dev/vipcore` does not exist — the NPU kernel driver is not loaded
- `vip_init` or `vip_create_network` fails

If the NPU errors *during* a frame, that one face is embedded on the CPU and
the loop continues. The error counter is visible as repeated warnings.

## Troubleshooting

**`viplite doesn't support this buffer format`** — the graph was compiled for a
different chip. Check the `--optimize` target against the `cid` printed at
startup.

**`input buffer alloc failed` with a garbage shape** — a struct layout
mismatch. The VIPLite v1.13 `vip_buffer_create_params_t` puts `memory_type`
**last** and sizes `sizes[6]`; the v2.0 layout is different (56 bytes vs 32 for
the allocation struct) and the driver returns `status=-11`. Use the v1.13
runtime that ships with the T527 SDK.

**`vip_create_network ... not support this create network type=0x0`** —
`VIP_CREATE_NETWORK_FROM_FILE` is `0x01`, not `0`.

**Everything returns the previous call's result** — a missing
`vip_flush_buffer(buffer, VIP_BUFFER_OPER_TYPE_FLUSH)` after writing input. It
produces no error at all; the output is just one call stale.
