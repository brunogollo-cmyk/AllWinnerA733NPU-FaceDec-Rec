# (Re)building the `.nb` graphs

The NPU only executes Vivante **`.nb`** network binaries, so the ONNX models are
converted offline with Allwinner's **ACUITY/pegasus** toolkit and compiled for the
target chip. The A733 builds used by default are committed:

| File | Source model | Input | Output |
|---|---|---|---|
| `models/npu/yunet.nb` | `models/yunet.onnx` | `1x3x640x640` uint8 image (HWC) | 12 raw heads (cls/obj/bbox/kps × strides 8/16/32) |
| `models/npu/sface_int8.nb` | `face_recognition_sface_2021dec.onnx` | `1x3x112x112` float16 | `1x128` embedding |

Target: **`VIP9000NANODI_PLUS_PID0X1000003B`** (`cid=0x1000003B`), VIPLite 2.0.
A `.nb` built for another target is rejected by the driver (`vip_create_network
failed: -7`), so the target string and the platform are not interchangeable.

## A733 build (current, ACUITY v2.0)

The converter only runs on x86_64, inside Allwinner's Docker image.

1. **Get and load the image** (Radxa docs → Allwinner netdisk, A733 tab):

   ```bash
   unzip docker_images_v2.0.x.zip
   cd docker_images_v2.0.x
   unzip ubuntu-npu_v2.0.10.2.tar.zip
   sudo docker load -i ubuntu-npu_v2.0.10.2.tar
   sudo docker run --ipc=host -itd -v "$PWD:/workspace" \
       --name acuity_a733 ubuntu-npu:v2.0.10.2 /bin/bash
   ```

   Inside the container: ACUITY at `/root/acuity-toolkit-whl-6.30.22/bin`
   (pegasus 6.30.22), Vivante IDE at
   `/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools`.

2. **Run the model-zoo scripts** per model (import → quantize → export). The
   `scripts_model_convert/pegasus_*.sh` from Allwinner's AWNPU Model Zoo are
   copied into `tools/nb_convert_a733/scripts/` and used like this:

   ```bash
   export ACUITY_PATH=/root/acuity-toolkit-whl-6.30.22/bin
   export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
   export PATH=$ACUITY_PATH:$PATH

   ./pegasus_import.sh sface
   ./pegasus_quantize.sh sface uint8 10
   ./pegasus_export_ovx_nbg.sh sface uint8 a733
   # -> wksp/sface_uint8_nbg_unify/sface_uint8_a733.nb
   ```

   `a733` maps to `VIP9000NANODI_PLUS_PID0X1000003B` in
   `pegasus_export_ovx_nbg.sh`. Do the same with `yunet`.

   Calibration data: `dataset_widerface_10/` (images resized to the model input)
   works for detection. For SFace, prefer aligned 112x112 face crops; the
   embedding space shifts slightly with the calibration set, so validate
   `cosine(NPU, CPU)` after building.

3. **Copy the result** into `models/npu/` and run
   `./.venv/bin/python tools/check_npu.py` on the board.

`tools/nb_convert_a733/` keeps the source ONNX files, the calibration set, the
model-zoo scripts and starter `config_yml.py` files, so the conversion can be
reproduced without hunting for inputs.

### YuNet specifics

The exported YuNet graph keeps the raw detection heads. `facegate/detector.py`
takes the 12 outputs, groups them by stride, applies `score = cls * obj`,
decodes boxes (`cx + dx·s`, `exp(dw)·s`, …), maps the 5 keypoints and runs NMS.
The graph input is a raw uint8 image (the ACUITY preproc node does resize /
channel order), so Python only resizes to 640×640.

### SFace specifics

Input is a float16 NCHW tensor of the aligned 112×112 crop; output is one
uint8-asymmetric-affine tensor of 128 values, divided by its scale and
L2-normalised in `NpuRecognizer`. Keep that output layout: the enrolled
`data/facedb.json` is only valid if the embedding space stays the same.

## T527 (legacy)

The same flow works on the T527, but with different artifacts:

- image `ubuntu-npu:v1.8.13` (`docker_images_v1.8.x.zip`), ACUITY v1.8.13;
- platform/target `VIP9000NANOSI_PLUS_PID0X10000016` (`cid=0x10000016`);
- runtime `libVIPlite.so` + `libVIPuser.so` from SDK v1.x instead of
  `libNBGlinker.so` + `libVIPhal.so`.

Do **not** mix the images: a v2.0 graph does not run on a v1.x runtime, and vice
versa.

## Runtime and kernel module

The A733 userspace runtime is in `tools/npu_runtime/A733/`; install it with
`sudo ./tools/install_npu_runtime.sh`. The kernel module `vipcore` that creates
`/dev/vipcore` is part of the board image and is **not** redistributed here —
see `tools/npu_runtime/README.md`.

## Validating a build

`npu.py` mirrors the vendor's `vpm_run` call sequence and queries buffer
descriptors from the graph instead of hardcoding them, so a differently shaped
model works unchanged. On the board:

```bash
./.venv/bin/python tools/check_npu.py
```

prints driver version, chip id, input/output shapes and one inference per graph.
`tools/npu_embed.c` is a standalone C CLI for A/B checking a single embedding
against the CPU model.
