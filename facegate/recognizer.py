"""SFace embedding extraction and the enrolled-face database."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import RecognizerConfig, SFACE_REF_POINTS
from .detector import Face
from .npu import load_recognizer_reason as load_npu_reason

log = logging.getLogger(__name__)

ALIGN_SIZE = 112


@dataclass
class Match:
    name: str | None
    similarity: float

    @property
    def is_known(self) -> bool:
        return self.name is not None


class FaceRecognizer:
    """Turns a detected face into a 128-d embedding, aligned via its landmarks.

    Backends (when ``recognizer.use_npu`` is enabled):
      1. VIPLite .nb graph (A733-compatible) via NpuRecognizer
      2. OpenCV DNN TIM-VX (ONNX on NPU) if available
      3. CPU fallback

    The embedding space matches SFace (128-d, L2-normalised) so the enrolled
    facedb remains valid across backends.
    """

    def __init__(self, cfg: RecognizerConfig) -> None:
        model = cfg.model_path()
        if not model.exists():
            raise FileNotFoundError(
                f"Recognizer model not found at {model}. Run ./install.sh to fetch models."
            )
        self.cfg = cfg
        self._cpu_net = cv2.FaceRecognizerSF.create(str(model), "")
        self._net = self._cpu_net
        self.last_ms = 0.0
        self.backend = "cpu"
        self.backend_detail = "cpu"
        self.npu = None
        self.npu_error = None
        self._timvx_active = False

        if cfg.use_npu:
            # 1) VIPLite .nb (requires graph compiled for this NPU target, e.g. A733)
            self.npu, v_err = load_npu_reason(cfg.npu_model_path(), cfg.viplite_path())
            if self.npu is not None:
                self.backend = "npu"
                self.backend_detail = "npu-viplite"
                log.info(
                    "Face embedding on NPU VIPLite (%s)",
                    cfg.npu_model_path().name,
                )
            else:
                self.npu_error = v_err
                # 2) OpenCV DNN TIM-VX backend running the ONNX model on the NPU
                timvx_net, t_err = self._try_timvx(str(model))
                if timvx_net is not None:
                    self._net = timvx_net
                    self._timvx_active = True
                    self.backend = "npu"
                    self.backend_detail = "npu-timvx"
                    # keep VIPLite error for diagnostics but NPU is active via TIM-VX
                    if v_err:
                        log.info(
                            "Face embedding on NPU TIM-VX (%s) (VIPLite unavailable: %s)",
                            model.name,
                            v_err,
                        )
                    else:
                        log.info("Face embedding on NPU TIM-VX (%s)", model.name)
                else:
                    reasons = []
                    if v_err:
                        reasons.append(f"viplite: {v_err}")
                    if t_err:
                        reasons.append(f"timvx: {t_err}")
                    self.npu_error = "; ".join(reasons) if reasons else "NPU indisponível"
                    log.warning(
                        "NPU requested but unavailable (%s) - embedding on CPU",
                        self.npu_error,
                    )

    def _try_timvx(self, model_path: str) -> tuple[cv2.FaceRecognizerSF | None, str | None]:
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
                        "forneça .nb VIPLite compilado para A733 (ex.: models/npu/sface_int8.nb, alvo 0x1000003B)"
                    )
        except Exception:  # noqa: BLE001
            pass
        try:
            net = cv2.FaceRecognizerSF.create(
                model_path,
                "",
                cv2.dnn.DNN_BACKEND_TIMVX,
                cv2.dnn.DNN_TARGET_NPU,
            )
            # Smoke test: ensure inference runs on this device/runtime.
            dummy = np.zeros((ALIGN_SIZE, ALIGN_SIZE, 3), dtype=np.uint8)
            feat = net.feature(dummy)
            if feat is None:
                return None, "TIM-VX retornou feature nula"
            return net, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray | None:
        aligned = self.align(frame, face)
        if aligned is None:
            return None
        return self.embed_aligned(aligned)

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray | None:
        """Embed an already-aligned 112x112 crop, skipping the affine warp."""
        if self.npu is not None:
            feature = self.npu.embed_aligned(aligned)
            # The NPU returns a value already L2-normalised, and sets its own
            # last_ms. A None here means the graph could not run, which should
            # not silently deny everyone, so fall through to the CPU.
            if feature is not None:
                self.last_ms = self.npu.last_ms
                return feature
            log.warning("NPU VIPLite inference returned nothing - using CPU for this face")

        start = time.perf_counter()
        try:
            feature = self._net.feature(aligned)
        except Exception:  # noqa: BLE001 - fall back to CPU rather than break
            if self._timvx_active:
                log.exception("TIM-VX embedding failed - falling back to CPU")
                self._net = self._cpu_net
                self._timvx_active = False
                self.backend = "cpu"
                self.backend_detail = "cpu-fallback"
                self.npu_error = (self.npu_error or "") + "; timvx runtime failure"
                feature = self._net.feature(aligned)
            else:
                raise
        self.last_ms = (time.perf_counter() - start) * 1000

        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if norm < 1e-6:
            return None
        # SFace returns unnormalised features (measured norm ~9.95), so cosine
        # similarity has to divide the norm out. Normalise once, here, so the
        # database and every query live in the same space.
        return feature / norm

    def align(self, frame: np.ndarray, face: Face) -> np.ndarray | None:
        matrix = cv2.estimateAffinePartial2D(
            face.landmarks, SFACE_REF_POINTS, method=cv2.LMEDS
        )[0]
        if matrix is None:
            return None
        return cv2.warpAffine(
            frame, matrix, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0
        )


class FaceDB:
    """Multiple L2-normalised embeddings per person, matched by nearest cosine.

    Storing several embeddings per person and taking the best match avoids the
    accuracy loss that comes from averaging them into a single vector: averaging
    blurs across expressions and head angles and raises false rejects.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._names: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, 128), dtype=np.float32)
        self._owner: list[int] = []

    @property
    def people(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def add(self, name: str, feature: np.ndarray) -> None:
        if name not in self._names:
            self._names.append(name)
        owner = self._names.index(name)
        self._matrix = np.vstack([self._matrix, feature.reshape(1, -1).astype(np.float32)])
        self._owner.append(owner)

    def remove_person(self, name: str) -> bool:
        """Drop every embedding for a person and renumber the owner indices."""
        if name not in self._names:
            return False

        keep = [i for i, o in enumerate(self._owner) if self._names[o] != name]
        if len(keep) == len(self._owner):
            return False  # name present but owned no embeddings

        self._matrix = (
            self._matrix[keep] if keep else np.zeros((0, 128), dtype=np.float32)
        )
        old_owner = [self._owner[i] for i in keep]
        dropped = {o for o in range(len(self._names)) if self._names[o] == name}
        remap = {o: i for i, o in enumerate(o for o in range(len(self._names)) if o not in dropped)}
        self._names = [n for i, n in enumerate(self._names) if i not in dropped]
        self._owner = [remap[o] for o in old_owner]
        return True

    def match(self, feature: np.ndarray, threshold: float) -> Match:
        if self._matrix.shape[0] == 0:
            return Match(name=None, similarity=0.0)

        query = feature.reshape(1, -1).astype(np.float32)
        # Vectors are already L2-normalised, so a dot product is the cosine.
        # Flatten first: argmax over a 2-D array would yield a row index.
        sims = (self._matrix @ query.T).reshape(-1)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best < threshold:
            return Match(name=None, similarity=best)
        return Match(name=self._names[self._owner[idx]], similarity=best)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dim": int(self._matrix.shape[1]) if self._matrix.size else 128,
            "people": [
                {
                    "name": name,
                    "embeddings": self._matrix[
                        [i for i, o in enumerate(self._owner) if self._names[o] == name]
                    ]
                    .tolist(),
                }
                for name in self._names
            ],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(payload, fh)
        tmp.replace(self.path)
        log.info("Saved %d people / %d embeddings to %s", len(self._names), len(self._owner), self.path)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with self.path.open() as fh:
            payload = json.load(fh)
        self._names = []
        self._owner = []
        vectors: list[np.ndarray] = []
        for person in payload.get("people", []):
            name = person["name"]
            self._names.append(name)
            owner = len(self._names) - 1
            for vec in person["embeddings"]:
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm < 1e-6:
                    continue
                vectors.append(arr / norm)
                self._owner.append(owner)
        self._matrix = (
            np.vstack(vectors) if vectors else np.zeros((0, 128), dtype=np.float32)
        )
        log.info("Loaded %d people / %d embeddings from %s", len(self._names), len(self._owner), self.path)
        return True

    def mean_intra_person_similarity(self) -> dict[str, float]:
        """Mean best-match similarity within each person.

        A low value means the enrolled images of that person are inconsistent, and
        the access threshold cannot be trusted for them.
        """
        out: dict[str, float] = {}
        for i, name in enumerate(self._names):
            idx = [j for j, o in enumerate(self._owner) if o == i]
            if len(idx) < 2:
                out[name] = float("nan")
                continue
            block = self._matrix[idx]
            sims = block @ block.T
            np.fill_diagonal(sims, 0.0)
            out[name] = float(sims.max(axis=1).mean())
        return out
