"""YuNet face detection wrapper, CPU (OpenCV) or NPU (VIPLite .nb graph).

The CPU path is the default and always available (opencv-contrib, YuNet ONNX).
The NPU path runs a converted YuNet ``.nb`` on the Allwinner VIPLite runtime when
``detector.use_npu`` is enabled and the graph/library are present; if anything is
unavailable it falls back to the CPU transparently.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig
from .npu import load_session_reason as load_npu_session_reason

log = logging.getLogger(__name__)


@dataclass
class Face:
    """One detected face: bounding box, landmarks and detector confidence."""

    box: tuple[int, int, int, int]  # x, y, w, h in source-frame pixels
    landmarks: np.ndarray  # (5, 2) float32, in source-frame pixels
    score: float


class FaceDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        model = cfg.path()
        if not model.exists():
            raise FileNotFoundError(
                f"Detector model not found at {model}. Run ./install.sh to fetch models."
            )
        self.cfg = cfg
        self.last_ms = 0.0
        self.backend = "cpu"
        self.backend_detail = "cpu"

        size = (cfg.input_size, cfg.input_size)
        self._cpu_net = cv2.FaceDetectorYN.create(
            str(model), "", size, cfg.score_threshold, cfg.nms_threshold, cfg.top_k
        )
        self._net = self._cpu_net

        self.npu = None
        self.npu_error = None
        self._timvx_active = False
        if cfg.use_npu:
            # 1) VIPLite .nb graph (converted YuNet for this NPU target)
            self.npu, v_err = load_npu_session_reason(
                cfg.npu_model_path(), cfg.viplite_path()
            )
            if self.npu is not None:
                self.backend = "npu"
                self.backend_detail = "npu-viplite"
                log.info("Face detection on NPU VIPLite (%s)", cfg.npu_model_path().name)
            else:
                self.npu_error = v_err
                # 2) OpenCV DNN TIM-VX backend running YuNet ONNX on the NPU
                timvx_net, t_err = self._try_timvx(str(model), size, cfg)
                if timvx_net is not None:
                    self._net = timvx_net
                    self._timvx_active = True
                    self.backend = "npu"
                    self.backend_detail = "npu-timvx"
                    if v_err:
                        log.info(
                            "Face detection on NPU TIM-VX (%s) (VIPLite unavailable: %s)",
                            model.name,
                            v_err,
                        )
                    else:
                        log.info("Face detection on NPU TIM-VX (%s)", model.name)
                else:
                    reasons = []
                    if v_err:
                        reasons.append(f"viplite: {v_err}")
                    if t_err:
                        reasons.append(f"timvx: {t_err}")
                    self.npu_error = "; ".join(reasons) if reasons else "NPU indisponível"
                    log.warning(
                        "Detector NPU requested but unavailable (%s) - detection on CPU",
                        self.npu_error,
                    )

    def _try_timvx(
        self, model_path: str, size: tuple[int, int], cfg: DetectorConfig
    ) -> tuple[cv2.FaceDetectorYN | None, str | None]:
        if not hasattr(cv2, "dnn") or not hasattr(cv2.dnn, "DNN_BACKEND_TIMVX"):
            return None, "OpenCV sem suporte a DNN_BACKEND_TIMVX"
        if not hasattr(cv2.dnn, "DNN_TARGET_NPU"):
            return None, "OpenCV sem suporte a DNN_TARGET_NPU"
        # OpenCV 5.x "new graph engine" pode aceitar argumentos TIM-VX mas executar
        # em CPU sem despacho real à NPU (nenhuma carga observada). Só habilita
        # TIM-VX se o backend/target for reportado como disponível; caso contrário
        # exige grafo .nb VIPLite compilado para o A733.
        try:
            get_targets = getattr(cv2.dnn, "getAvailableTargets", None)
            if get_targets is not None:
                targets = list(get_targets(cv2.dnn.DNN_BACKEND_TIMVX) or [])
                if cv2.dnn.DNN_TARGET_NPU not in targets:
                    return None, (
                        "backend TIM-VX/NPU indisponível neste OpenCV (sem aceleração NPU real); "
                        "forneça .nb VIPLite compilado para A733 (ex.: models/npu/yunet.nb, alvo 0x1000003B)"
                    )
        except Exception:  # noqa: BLE001
            pass
        try:
            net = cv2.FaceDetectorYN.create(
                model_path,
                "",
                size,
                cfg.score_threshold,
                cfg.nms_threshold,
                cfg.top_k,
                cv2.dnn.DNN_BACKEND_TIMVX,
                cv2.dnn.DNN_TARGET_NPU,
            )
            dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            net.setInputSize(size)
            net.detect(dummy)
            return net, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    # ----------------------------------------------------------------- helpers

    def _rescale(self, faces: np.ndarray | None, w: int, h: int) -> list[Face]:
        if faces is None:
            return []
        side = float(self.cfg.input_size)
        sx = w / side
        sy = h / side
        out: list[Face] = []
        for f in faces:
            x, y = int(round(f[0] * sx)), int(round(f[1] * sy))
            fw, fh = int(round(f[2] * sx)), int(round(f[3] * sy))
            landmarks = np.array(
                [
                    [f[4] * sx, f[5] * sy],
                    [f[6] * sx, f[7] * sy],
                    [f[8] * sx, f[9] * sy],
                    [f[10] * sx, f[11] * sy],
                    [f[12] * sx, f[13] * sy],
                ],
                dtype=np.float32,
            )
            out.append(Face(box=(x, y, fw, fh), landmarks=landmarks, score=float(f[-1])))
        return out

    def _detect_with_net(self, net, resized: np.ndarray) -> np.ndarray | None:
        _, faces = net.detect(resized)
        return faces

    def _detect_cpu(self, resized: np.ndarray) -> np.ndarray | None:
        return self._detect_with_net(self._cpu_net, resized)

    def _detect_npu(self, resized: np.ndarray) -> np.ndarray | None:
        """Run converted YuNet .nb on VIPLite.

        Supports two export layouts:
        1) Decoded rows [x,y,w,h, 5 landmarks, score] (15 values/face) - legacy.
        2) Raw YuNet heads (12 outputs: cls/obj/bbox/kps for strides 8,16,32),
           as produced by the ACUITY export in this repo. We decode to layout (1).
        """
        if self.npu is None:
            return None
        target = resized
        if getattr(self.npu, "input_is_image_u8", False):
            # Graph carries a preproc node and takes a raw HWC uint8 image
            # (BGR; the node reverses channels per the exported input meta).
            side = int(getattr(self.npu, "input_side", 0) or 0)
            if side > 0 and (resized.shape[0] != side or resized.shape[1] != side):
                target = cv2.resize(resized, (side, side))
            tensor = np.ascontiguousarray(target, dtype=np.uint8)
        else:
            in_bytes = int(getattr(self.npu, "in_size", 0) or 0)
            side = int(round((in_bytes / 6) ** 0.5)) if in_bytes else 0
            if side and side * side * 6 == in_bytes and (
                resized.shape[0] != side or resized.shape[1] != side
            ):
                target = cv2.resize(resized, (side, side))
            rgb = target[:, :, ::-1]
            tensor = np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[None, ...].astype(np.float16)
            )

        try:
            outs = self.npu.run_all(tensor) if hasattr(self.npu, "run_all") else [self.npu.run(tensor)]
        except Exception:  # noqa: BLE001
            self.npu.failures += 1
            log.exception("NPU detection failed")
            return None
        if not outs:
            return None

        # Layout 1: single decoded output
        if len(outs) == 1:
            raw = np.asarray(outs[0], dtype=np.float32).reshape(-1)
            if raw.size >= 15 and raw.size % 15 == 0:
                faces = raw.reshape(-1, 15)
                faces = faces[faces[:, 14] >= float(self.cfg.score_threshold)]
                if faces.size == 0:
                    return None
                top_k = max(1, int(self.cfg.top_k))
                if faces.shape[0] > top_k:
                    order = np.argsort(faces[:, 14])[::-1][:top_k]
                    faces = faces[order]
                # Coords are in target/input space; rescale to resized space if needed
                if target.shape[0] != resized.shape[0] or target.shape[1] != resized.shape[1]:
                    sx = resized.shape[1] / float(target.shape[1])
                    sy = resized.shape[0] / float(target.shape[0])
                    faces = faces.copy()
                    faces[:, 0] *= sx; faces[:, 2] *= sx
                    faces[:, 1] *= sy; faces[:, 3] *= sy
                    for k in range(5):
                        faces[:, 4 + k*2] *= sx
                        faces[:, 5 + k*2] *= sy
                return faces

        # Layout 2: raw YuNet heads (12 outputs)
        faces = self._decode_yunet_heads(outs, target.shape[1], target.shape[0])
        if faces is None or faces.size == 0:
            return None
        if target.shape[0] != resized.shape[0] or target.shape[1] != resized.shape[1]:
            sx = resized.shape[1] / float(target.shape[1])
            sy = resized.shape[0] / float(target.shape[0])
            faces = faces.copy()
            faces[:, 0] *= sx; faces[:, 2] *= sx
            faces[:, 1] *= sy; faces[:, 3] *= sy
            for k in range(5):
                faces[:, 4 + k*2] *= sx
                faces[:, 5 + k*2] *= sy
        return faces

    def _decode_yunet_heads(self, outs: list[np.ndarray], W: int, H: int) -> np.ndarray | None:
        """Decode YuNet raw heads to [x,y,w,h, 5lms, score] rows in input-image space."""
        # Expect 12 outputs: cls_8,16,32,obj_8,16,32,bbox_8,16,32,kps_8,16,32
        # in some order. Identify by shape: (N,1) cls/obj, (N,4) bbox, (N,10) kps.
        by_shape = {(int(o.reshape(-1, o.shape[-1]).shape[0]), int(o.shape[-1])): [] for o in outs}
        buckets = {}
        for o in outs:
            a = np.asarray(o, dtype=np.float32)
            a = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(-1, 1)
            buckets.setdefault((a.shape[0], a.shape[1]), []).append(a)
        # stride mapping by N
        stride_map = {6400: 8, 1600: 16, 400: 32}
        # For 640 input: 80*80=6400,40*40=1600,20*20=400. For other sizes scale accordingly.
        # Try to infer from common YuNet grid sizes relative to input
        def n_to_stride(n):
            if n in stride_map: return stride_map[n]
            # generic: find stride s where (W/s)*(H/s)==n
            for s in (8,16,32,4,64):
                if (W % s == 0 and H % s == 0 and (W//s)*(H//s) == n):
                    return s
            return None
        cls = {}
        obj = {}
        bbox = {}
        kps = {}
        for (n,c), arrs in buckets.items():
            s = n_to_stride(n)
            if s is None:
                continue
            if c == 1:
                # two per stride: cls and obj
                if s not in cls:
                    cls[s] = arrs[0]
                    if len(arrs) > 1:
                        obj[s] = arrs[1]
                else:
                    obj[s] = arrs[0]
            elif c == 4:
                bbox[s] = arrs[0]
            elif c == 10:
                kps[s] = arrs[0]
        strides = sorted(set(cls.keys()) & set(obj.keys()) & set(bbox.keys()) & set(kps.keys()))
        if not strides:
            return None
        rows = []
        score_thr = float(self.cfg.score_threshold)
        for s in strides:
            c = cls[s].reshape(-1)
            o = obj[s].reshape(-1)
            b = bbox[s].reshape(-1, 4)
            k = kps[s].reshape(-1, 10)
            score = (c * o).reshape(-1)
            keep = np.where(score >= score_thr)[0]
            if keep.size == 0:
                continue
            n = c.shape[0]
            gw = W // s
            gh = H // s
            if gw * gh != n:
                # try transpose
                gw, gh = gh, gw
                if gw * gh != n:
                    continue
            # anchor centers
            ys = (np.arange(gh, dtype=np.float32) + 0.5) * s
            xs = (np.arange(gw, dtype=np.float32) + 0.5) * s
            ax, ay = np.meshgrid(xs, ys)
            ax = ax.reshape(-1)
            ay = ay.reshape(-1)
            for i in keep:
                sc = float(score[i])
                # YuNet decode: bbox = [dx*s, dy*s, exp(dw)*s, exp(dh)*s] around anchor? 
                # OpenCV YuNet: x = (cx - bbox[0]*s), y=(cy-bbox[1]*s), w=exp(bbox[2])*s, h=exp(bbox[3])*s? Actually
                # standard YuNet decode: cx=anchor_x + bbox[i,0]*s; cy=anchor_y+bbox[i,1]*s; w=exp(bbox[i,2])*s; h=exp(bbox[i,3])*s
                dx, dy, dw, dh = b[i]
                cx_ = ax[i] + dx * s
                cy_ = ay[i] + dy * s
                w_ = float(np.exp(np.clip(dw, -10, 10)) * s)
                h_ = float(np.exp(np.clip(dh, -10, 10)) * s)
                x_ = cx_ - w_ / 2.0
                y_ = cy_ - h_ / 2.0
                lm = np.zeros(10, dtype=np.float32)
                for j in range(5):
                    lm[j*2] = ax[i] + k[i, j*2] * s
                    lm[j*2+1] = ay[i] + k[i, j*2+1] * s
                rows.append([x_, y_, w_, h_, *lm.tolist(), sc])
        if not rows:
            return None
        faces = np.asarray(rows, dtype=np.float32)
        top_k = max(1, int(self.cfg.top_k))
        if faces.shape[0] > top_k:
            order = np.argsort(faces[:, 14])[::-1][:top_k]
            faces = faces[order]
        # NMS
        boxes = faces[:, :4].copy()
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scores = faces[:, 14]
        keep_idx = self._nms(boxes, scores, float(self.cfg.nms_threshold))
        faces = faces[keep_idx]
        return faces

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1)
            h = np.maximum(0.0, yy2-yy1)
            inter = w*h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= thr)[0]
            order = order[inds+1]
        return keep

    # ----------------------------------------------------------------- public

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        side = self.cfg.input_size

        # Keep the network input fixed at cfg.input_size and resize into it.
        # Setting YuNet's input size to the frame size instead would make the
        # network run at full camera resolution: on this board 640x480 costs
        # ~88ms/frame versus ~20ms at 256x256, for no accuracy gain at the
        # detection distances an access-control camera actually sees.
        if w != side or h != side:
            resized = cv2.resize(frame, (side, side))
        else:
            resized = frame

        start = time.perf_counter()
        faces: np.ndarray | None = None
        if self.npu is not None:
            try:
                faces = self._detect_npu(resized)
            except Exception:  # noqa: BLE001 - prefer a correct result on CPU
                log.warning(
                    "NPU VIPLite detector output not usable - falling back to CPU"
                )
                faces = None
            if faces is None:
                faces = self._detect_with_net(self._cpu_net, resized)
        else:
            try:
                faces = self._detect_with_net(self._net, resized)
            except Exception:  # noqa: BLE001 - fall back to CPU rather than break
                if self._timvx_active:
                    log.exception("TIM-VX detection failed - falling back to CPU")
                    self._net = self._cpu_net
                    self._timvx_active = False
                    self.backend = "cpu"
                    self.backend_detail = "cpu-fallback"
                    self.npu_error = (self.npu_error or "") + "; timvx runtime failure"
                    faces = self._detect_with_net(self._cpu_net, resized)
                else:
                    raise
        self.last_ms = (time.perf_counter() - start) * 1000

        return self._rescale(faces, w, h)
