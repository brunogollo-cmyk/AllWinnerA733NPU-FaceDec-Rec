"""NPU backend for face embeddings and detection, via Allwinner VIPLite.

The Orange Pi Zero 3W carries an Allwinner A733 (2x Cortex-A76 + 6x Cortex-A55,
powerVR GPU, VeriSilicon NPU). The face *embedding* runs on that NPU, which is the
most expensive step of the pipeline; face *detection* can run there too.

The NPU does not run ONNX. It runs Vivante ``.nb`` network binaries produced by
the Allwinner ACUITY toolkit (``import onnx -> quantize -> export ovxlib
--pack-nbg-unify``). The converted graph is loaded through the VIPLite runtime
that ships with the board's AI SDK.

On this board (Allwinner A733, AI SDK v2.0) the runtime is exposed as
``libNBGlinker.so`` (which pulls in ``libVIPhal.so``) from the system path, with
the kernel module ``vipcore`` providing ``/dev/vipcore``. The older T527 board
used the SDK v1.x runtime (``libVIPlite.so`` + ``libVIPuser.so``) instead, so the
candidates below cover both.

Measured on the A733: the int8 SFace graph embeds a face in ~15 ms on the NPU
against ~68 ms for the float32 ONNX model on the CPU, with cosine agreement
~0.99 over real camera crops.
"""
from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ALIGN_SIZE = 112
EMBED_DIM = 128

# Enumerations from vip_lite.h (AI SDK v2.0, Allwinner A733).
VIP_SUCCESS = 0
VIP_BUFFER_MEMORY_TYPE_DEFAULT = 0
VIP_BUFFER_QUANTIZE_NONE = 0
VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT = 1
VIP_BUFFER_QUANTIZE_TF_ASYMM = 2
VIP_CREATE_NETWORK_FROM_FILE = 0x01
VIP_BUFFER_OPER_TYPE_FLUSH = 1
VIP_BUFFER_OPER_TYPE_INVALIDATE = 2

VIP_BUFFER_MEMORY_TYPE_HOST = 1
VIP_BUFFER_FORMAT_FP32 = 0
VIP_BUFFER_FORMAT_FP16 = 1
VIP_BUFFER_FORMAT_UINT8 = 2
VIP_BUFFER_FORMAT_INT8 = 3

# Bytes per element for vip_buffer_format_e.
_FORMAT_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 2,
                 8: 4, 9: 4, 10: 8, 11: 8, 12: 8, 16: 1}
_FORMAT_CTYPES = {0: ctypes.c_float, 1: ctypes.c_uint16, 2: ctypes.c_uint8,
                  3: ctypes.c_int8, 4: ctypes.c_uint16, 5: ctypes.c_int16,
                  6: ctypes.c_int8, 8: ctypes.c_int32, 9: ctypes.c_uint32}

_libc = ctypes.CDLL(None, use_errno=True)


def _aligned_alloc(size: int, align: int) -> ctypes.c_void_p:
    """Allocate host memory aligned to `align`, as the VIPLite API requires."""
    ptr = ctypes.c_void_p()
    rc = _libc.posix_memalign(
        ctypes.byref(ptr), ctypes.c_size_t(align), ctypes.c_size_t(size)
    )
    if rc != 0 or not ptr:
        raise NpuUnavailable(f"posix_memalign failed (rc={rc}, size={size}, align={align})")
    return ptr

# Values from vip_buffer_property_e in vip_lite.h.
VIP_BUFFER_PROP_QUANT_FORMAT = 0
VIP_BUFFER_PROP_NUM_OF_DIMENSION = 1
VIP_BUFFER_PROP_SIZES_OF_DIMENSION = 2
VIP_BUFFER_PROP_DATA_FORMAT = 3
VIP_BUFFER_PROP_FIXED_POINT_POS = 4
VIP_BUFFER_PROP_TF_SCALE = 5
VIP_BUFFER_PROP_TF_ZERO_POINT = 6
VIP_BUFFER_PROP_NAME = 7

# Values from vip_network_property_e in vip_lite.h (AI SDK v2.0).
VIP_NETWORK_PROP_LAYER_COUNT = 0
VIP_NETWORK_PROP_INPUT_COUNT = 1
VIP_NETWORK_PROP_OUTPUT_COUNT = 2


class NpuUnavailable(RuntimeError):
    """The NPU path cannot be used; fall back to the CPU."""


class _Affine(ctypes.Structure):
    _fields_ = [("scale", ctypes.c_float), ("zeroPoint", ctypes.c_int32)]


class _QuantData(ctypes.Union):
    """The quant_data union from vip_buffer_create_params_t."""

    _fields_ = [
        ("dfp", ctypes.c_int32),
        ("affine", _Affine),
    ]


class _BufferParams(ctypes.Structure):
    """Mirrors vip_buffer_create_params_t exactly.

    Field order matters: ctypes lays the struct out in declaration order, and
    the driver validates it. Note that `sizes` has six entries and `memory_type`
    comes last, which is the opposite of the intuitive ordering.
    """

    _fields_ = [
        ("num_of_dims", ctypes.c_uint32),
        ("sizes", ctypes.c_uint32 * 6),
        ("data_format", ctypes.c_uint32),
        ("quant_format", ctypes.c_uint32),
        ("quant_data", _QuantData),
        ("memory_type", ctypes.c_uint32),
    ]


def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name, None)
    if fn is None:
        raise NpuUnavailable(f"libVIPlite has no symbol {name}")
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# Runtime library candidates, in preference order. The Allwinner A733 ships the
# AI SDK v2.0 runtime (libNBGlinker.so + its HAL libVIPhal.so); older boards such
# as the T527 used libVIPlite.so + libVIPuser.so from SDK v1.x.
_LIB_PAIRS = (
    ("libNBGlinker.so", "libVIPhal.so"),
    ("libVIPlite.so", "libVIPuser.so"),
)

# VIPLite's global init is process-wide; track it so multiple sessions/graphs do
# not re-init or tear it down while others are active.
_VIP_INIT_COUNT = 0
_VIP_LOCK = __import__("threading").Lock()


class NpuSession:
    """Generic VIPLite session: one ``.nb`` graph, run to inference.

    The session owns the ctypes binding to libVIPlite/libNBGlinker, the graph
    handle and the input/output buffers, and turns a float16 NCHW tensor into the
    dequantised output array of the graph.
    """

    def __init__(self, nb_path: Path, lib_dir: Path) -> None:
        self.nb_path = Path(nb_path)
        self.lib_dir = Path(lib_dir)
        self.last_ms = 0.0
        self.failures = 0
        self._closed = False

        lib_path = self._find_lib()
        if not lib_path.exists():
            raise NpuUnavailable(f"{lib_path} not found")
        # Companion libraries live next to the linker library that was found.
        self._lib_dir = lib_path.parent
        if not self.nb_path.exists():
            raise NpuUnavailable(f"{self.nb_path} not found")
        if not Path("/dev/vipcore").exists():
            raise NpuUnavailable("/dev/vipcore missing (is vipcore.ko loaded?)")

        # libVIPhal carries the hardware abstraction the linker needs, and the
        # linker resolves the vip_* symbols; load both globally first.
        for name in _LIB_PAIRS[0][1:]:
            candidate = self._lib_dir / name
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        self._bind_api()
        self._load_graph()

    # ----------------------------------------------------------------- setup

    def _find_lib(self):
        # Look where the config points first, then the usual install locations.
        dirs = [self.lib_dir, Path("/usr/local/lib"), Path("/usr/lib")]
        for names in _LIB_PAIRS:
            for directory in dirs:
                primary = directory / names[0]
                if primary.exists():
                    return primary
        return self.lib_dir / "libNBGlinker.so"

    def _bind_api(self) -> None:
        lib = self._lib
        self._vip_init = _bind(lib, "vip_init", ctypes.c_int, [])
        self._vip_get_version = _bind(lib, "vip_get_version", ctypes.c_uint32, [])
        self._vip_query_hardware = _bind(
            lib, "vip_query_hardware", ctypes.c_int,
            [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_create_network = _bind(
            lib, "vip_create_network", ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int,
             ctypes.POINTER(ctypes.c_void_p)],
        )
        self._vip_query_network = _bind(
            lib, "vip_query_network", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_query_input = _bind(
            lib, "vip_query_input", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_query_output = _bind(
            lib, "vip_query_output", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        self._vip_create_buffer = _bind(
            lib, "vip_create_buffer", ctypes.c_int,
            [ctypes.POINTER(_BufferParams), ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
        )
        # VIPLite on A733 expects host-mapped buffers (see npulib); fall back to
        # driver-allocated buffers only if the symbol is missing.
        self._vip_destroy = None
        fn = getattr(lib, "vip_destroy", None)
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = []
            self._vip_destroy = fn

        self._vip_create_buffer_from_handle = None
        fn = getattr(lib, "vip_create_buffer_from_handle", None)
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.POINTER(_BufferParams), ctypes.c_void_p, ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._vip_create_buffer_from_handle = fn
        self._vip_get_buffer_size = _bind(
            lib, "vip_get_buffer_size", ctypes.c_uint32, [ctypes.c_void_p]
        )
        self._vip_map_buffer = _bind(lib, "vip_map_buffer", ctypes.c_void_p, [ctypes.c_void_p])
        self._vip_unmap_buffer = _bind(lib, "vip_unmap_buffer", ctypes.c_int, [ctypes.c_void_p])
        self._vip_flush_buffer = _bind(
            lib, "vip_flush_buffer", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32]
        )
        self._vip_set_input = _bind(
            lib, "vip_set_input", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        )
        self._vip_set_output = _bind(
            lib, "vip_set_output", ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        )
        self._vip_prepare_network = _bind(
            lib, "vip_prepare_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_run_network = _bind(
            lib, "vip_run_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_finish_network = _bind(
            lib, "vip_finish_network", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_destroy_buffer = _bind(
            lib, "vip_destroy_buffer", ctypes.c_int, [ctypes.c_void_p]
        )
        self._vip_destroy_network = _bind(
            lib, "vip_destroy_network", ctypes.c_int, [ctypes.c_void_p]
        )

        global _VIP_INIT_COUNT
        with _VIP_LOCK:
            if _VIP_INIT_COUNT == 0:
                status = self._vip_init()
                if status != VIP_SUCCESS:
                    raise NpuUnavailable(f"vip_init failed: {status}")
            _VIP_INIT_COUNT += 1
            self._inited = True

        cid = ctypes.c_uint32(0)
        # property 0 = CID; pass a writable pointer and its size.
        self._vip_query_hardware(ctypes.c_uint32(0), ctypes.c_uint32(ctypes.sizeof(cid)), ctypes.byref(cid))
        self.cid = cid.value
        self.version = self._vip_get_version()

    def _query_params(self, index: int, is_input: bool) -> _BufferParams:
        param = _BufferParams()
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT
        query = self._vip_query_input if is_input else self._vip_query_output
        net = self._network

        # Query into standalone variables and copy in. Reading a struct field
        # through ctypes yields a plain int that byref() rejects, and byref(obj,
        # offset) does not write back, so the safe route is a local per query.
        data_format = ctypes.c_uint32(0)
        num_of_dims = ctypes.c_uint32(0)
        quant_format = ctypes.c_uint32(0)
        sizes = (ctypes.c_uint32 * 6)()

        query(net, index, VIP_BUFFER_PROP_DATA_FORMAT, ctypes.byref(data_format))
        query(net, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, ctypes.byref(num_of_dims))
        query(net, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes)
        query(net, index, VIP_BUFFER_PROP_QUANT_FORMAT, ctypes.byref(quant_format))

        param.data_format = data_format.value
        param.num_of_dims = num_of_dims.value
        param.quant_format = quant_format.value
        for i in range(len(sizes)):
            param.sizes[i] = sizes[i]
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT

        # Carry the graph's own quantisation parameters into the buffer request;
        # the driver uses them to place the data, not just to describe it.
        if param.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM:
            scale = ctypes.c_float(0.0)
            zero = ctypes.c_int32(0)
            query(net, index, VIP_BUFFER_PROP_TF_SCALE, ctypes.byref(scale))
            query(net, index, VIP_BUFFER_PROP_TF_ZERO_POINT, ctypes.byref(zero))
            param.quant_data.affine.scale = scale.value
            param.quant_data.affine.zeroPoint = zero.value
        elif param.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
            pos = ctypes.c_int32(0)
            query(net, index, VIP_BUFFER_PROP_FIXED_POINT_POS, ctypes.byref(pos))
            param.quant_data.dfp = pos.value
        return param

    def _load_graph(self) -> None:
        net = ctypes.c_void_p()
        status = self._vip_create_network(
            str(self.nb_path).encode(), 0, VIP_CREATE_NETWORK_FROM_FILE,
            ctypes.byref(net),
        )
        if status != VIP_SUCCESS:
            hint = ""
            # -4/-7 commonly indicate a graph compiled for a different NPU target/chip.
            if status in (-1, -4, -7):
                hint = " (grafo .nb incompatível com esta NPU/alvo? recompile para o A733)"
            raise NpuUnavailable(f"vip_create_network failed: {status}{hint} [{self.nb_path}]")
        self._network = net

        counts = []
        for prop in (VIP_NETWORK_PROP_INPUT_COUNT, VIP_NETWORK_PROP_OUTPUT_COUNT):
            value = ctypes.c_uint32(0)
            self._vip_query_network(net, prop, ctypes.byref(value))
            counts.append(value.value)
        self.input_count, self.output_count = counts
        if self.input_count < 1 or self.output_count < 1:
            raise NpuUnavailable(f"bad io counts {counts}")

        self._align = 64 if self.version <= 0x00020003 else 256
        self._host_ptrs: list[ctypes.c_void_p] = []
        self._prepared = False

        self._in_param = self._query_params(0, True)
        self._in_buf, self._in_ptr, self._in_bytes, _ = self._create_host_buffer(
            self._in_param
        )
        self.input_data_format = int(self._in_param.data_format)
        self.input_dims = [
            int(self._in_param.sizes[i]) for i in range(self._in_param.num_of_dims)
        ]
        # u8 image input (graph carries a preproc node): HWC interleaved bytes.
        self.input_is_image_u8 = (
            self.input_data_format == VIP_BUFFER_FORMAT_UINT8
            and len(self.input_dims) >= 3
            and self.input_dims[2] == 3
        )
        self.input_side = int(self.input_dims[0]) if self.input_dims else 0

        self._out_bufs = []
        self._out_ptrs = []
        self._out_elems = []
        self._out_dtypes = []
        self._out_scales = []
        self._out_params = []
        for i in range(self.output_count):
            param = self._query_params(i, False)
            buf, ptr, _nbytes, elements = self._create_host_buffer(param)
            self._out_bufs.append(buf)
            self._out_ptrs.append(ptr)
            self._out_elems.append(elements)
            self._out_dtypes.append(int(param.data_format))
            self._out_params.append(param)
            scale = ctypes.c_float(0.0)
            zero = ctypes.c_int32(0)
            self._vip_query_output(self._network, i, VIP_BUFFER_PROP_TF_SCALE, ctypes.byref(scale))
            self._vip_query_output(self._network, i, VIP_BUFFER_PROP_TF_ZERO_POINT, ctypes.byref(zero))
            self._out_scales.append((scale.value, zero.value))
        self._out_buf = self._out_bufs[0]
        self._out_param = self._out_params[0]
        self.out_scale, self.out_zero = self._out_scales[0]

        # Prepare and bind buffers once, exactly like Allwinner's npulib. Calling
        # vip_prepare_network again after it succeeded returns an error on VIPLite.
        self._ensure_prepared()

        self.in_size = self._in_bytes
        log.info(
            "NPU graph loaded: %s driver=0x%08x cid=0x%x in=%dB fmt=%d dims=%s outs=%d out_scale=%.9f zp=%d",
            self.nb_path.name, self.version, self.cid, self.in_size,
            self.input_data_format, self.input_dims, self.output_count,
            self.out_scale, self.out_zero,
        )

    def _create_host_buffer(self, param):
        """Allocate an aligned host buffer and wrap it in a VIPLite buffer."""
        elements = 1
        for i in range(param.num_of_dims):
            elements *= int(param.sizes[i])
        nbytes = elements * _FORMAT_BYTES.get(int(param.data_format), 1)
        alloc = (nbytes + self._align - 1) & ~(self._align - 1)
        ptr = _aligned_alloc(alloc, self._align)
        self._host_ptrs.append(ptr)
        if self._vip_create_buffer_from_handle is not None:
            param.memory_type = VIP_BUFFER_MEMORY_TYPE_HOST
            buf = ctypes.c_void_p()
            status = self._vip_create_buffer_from_handle(
                ctypes.byref(param), ptr, alloc, ctypes.byref(buf)
            )
        else:
            param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT
            buf = ctypes.c_void_p()
            status = self._vip_create_buffer(
                ctypes.byref(param), ctypes.sizeof(param), ctypes.byref(buf)
            )
        if status != VIP_SUCCESS:
            raise NpuUnavailable(
                f"buffer alloc failed: {status} (elements={elements}, "
                f"fmt={param.data_format}, bytes={nbytes})"
            )
        return buf, ptr, nbytes, elements

    def _ensure_prepared(self) -> None:
        if self._prepared:
            return
        net = self._network
        status = self._vip_prepare_network(net)
        if status != VIP_SUCCESS:
            raise NpuUnavailable(f"prepare failed: {status}")
        status = self._vip_set_input(net, 0, self._in_buf)
        if status != VIP_SUCCESS:
            raise NpuUnavailable(f"set_input failed: {status}")
        for i, obuf in enumerate(self._out_bufs):
            status = self._vip_set_output(net, i, obuf)
            if status != VIP_SUCCESS:
                raise NpuUnavailable(f"set_output[{i}] failed: {status}")
        self._prepared = True

    # ------------------------------------------------------------- inference

    def run(self, tensor: np.ndarray) -> np.ndarray | None:
        """Run one inference and return the dequantised output 0 as float32.

        The tensor is RGB, NCHW. It may be uint8 (u8 quantized input) or
        float16/float32 (fp16 input) as long as its byte size matches the
        graph input buffer. For multi-output graphs, use run_all().
        """
        outs = self.run_all(tensor)
        if not outs:
            return None
        return outs[0]

    def run_all(self, tensor: np.ndarray) -> list[np.ndarray] | None:
        """Run one inference and return all dequantised outputs as float32 arrays.

        The tensor is copied as raw bytes into the host input buffer. The caller
        decides the layout (HWC u8 image for graphs with a preproc node, CHW
        float16 for tensor inputs).
        """
        if tensor is None or tensor.size == 0:
            return None
        t = np.ascontiguousarray(tensor)
        data = t.view(np.uint8).reshape(-1)
        if data.size > self._in_bytes:
            raise NpuUnavailable(
                f"input {data.size}B larger than graph buffer {self._in_bytes}B"
            )
        ctypes.memmove(self._in_ptr, data.ctypes.data, data.size)
        self._vip_flush_buffer(self._in_buf, VIP_BUFFER_OPER_TYPE_FLUSH)

        net = self._network
        status = self._vip_run_network(net)
        if status != VIP_SUCCESS:
            raise NpuUnavailable(f"run failed: {status}")

        results = []
        for i, ptr in enumerate(self._out_ptrs):
            self._vip_flush_buffer(self._out_bufs[i], VIP_BUFFER_OPER_TYPE_INVALIDATE)
            ctype = _FORMAT_CTYPES.get(self._out_dtypes[i], ctypes.c_uint8)
            arr = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctype)), (self._out_elems[i],)
            )
            raw = arr.astype(np.float32)
            scale, zp = self._out_scales[i]
            results.append((raw - np.float32(zp)) * np.float32(scale))
        self.last_ms = 0.0
        return results

    # -------------------------------------------------------------- teardown

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handles = [getattr(self, "_in_buf", None)]
        handles += list(getattr(self, "_out_bufs", []) or [])
        for handle in handles:
            if handle:
                try:
                    self._vip_destroy_buffer(handle)
                except Exception:  # noqa: BLE001 - teardown is best effort
                    pass
        try:
            if getattr(self, "_network", None):
                if getattr(self, "_prepared", False):
                    try:
                        self._vip_finish_network(self._network)
                    except Exception:  # noqa: BLE001
                        pass
                self._vip_destroy_network(self._network)
        except Exception:  # noqa: BLE001
            pass
        for ptr in getattr(self, "_host_ptrs", []) or []:
            try:
                _libc.free(ptr)
            except Exception:  # noqa: BLE001
                pass
        self._host_ptrs = []
        global _VIP_INIT_COUNT
        if getattr(self, "_inited", False):
            with _VIP_LOCK:
                _VIP_INIT_COUNT = max(0, _VIP_INIT_COUNT - 1)
                # Release the device when the last session goes away. Without this,
                # a restarted service hangs in vip_init because the previous process
                # exited while still owning /dev/vipcore.
                if _VIP_INIT_COUNT == 0 and self._vip_destroy is not None:
                    try:
                        self._vip_destroy()
                    except Exception:  # noqa: BLE001 - teardown is best effort
                        pass
            self._inited = False

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass


class NpuRecognizer(NpuSession):
    """Runs the converted SFace embedding graph on the NPU."""

    def __init__(self, nb_path: Path, lib_dir: Path) -> None:
        super().__init__(nb_path, lib_dir)
        self.backend = "npu"

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray | None:
        """Embed an already-aligned 112x112 BGR crop, returning a 128-d unit vector."""
        if aligned is None or aligned.size == 0:
            return None
        if aligned.shape[0] != ALIGN_SIZE or aligned.shape[1] != ALIGN_SIZE:
            import cv2

            aligned = cv2.resize(aligned, (ALIGN_SIZE, ALIGN_SIZE))

        # The graph takes RGB in NCHW. Feed the same crop the CPU model sees so
        # the two paths stay comparable.
        rgb = aligned[:, :, ::-1]
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None, ...].astype(np.float16)
        )

        start = time.perf_counter()
        out = self.run(tensor)
        self.last_ms = (time.perf_counter() - start) * 1000
        if out is None:
            return None

        values = np.asarray(out, dtype=np.float32).reshape(-1)
        if values.size < EMBED_DIM:
            return None
        feature = values[:EMBED_DIM]
        norm = float(np.linalg.norm(feature))
        if norm < 1e-6:
            return None
        # SFace returns unnormalised features (measured norm ~9.95), so cosine
        # similarity has to divide the norm out. Normalise once, here, so the
        # database and every query live in the same space.
        return feature / norm


def load_recognizer(nb_path: Path, lib_dir: Path) -> NpuRecognizer | None:
    """Build the NPU recognizer backend, or None if it cannot be used."""
    obj, _ = load_recognizer_reason(nb_path, lib_dir)
    return obj


def load_recognizer_reason(nb_path: Path, lib_dir: Path) -> tuple[NpuRecognizer | None, str | None]:
    """Build the NPU recognizer backend, or (None, reason) if unavailable."""
    try:
        return NpuRecognizer(nb_path, lib_dir), None
    except (NpuUnavailable, OSError, AttributeError, ValueError) as exc:
        log.warning("NPU backend unavailable (%s) - using CPU", exc)
        return None, str(exc)


def load_session(nb_path: Path, lib_dir: Path) -> NpuSession | None:
    """Build a generic NPU session for any converted graph, or None if unusable."""
    obj, _ = load_session_reason(nb_path, lib_dir)
    return obj


def load_session_reason(nb_path: Path, lib_dir: Path) -> tuple[NpuSession | None, str | None]:
    """Build a generic NPU session, or (None, reason) if unavailable."""
    try:
        return NpuSession(nb_path, lib_dir), None
    except (NpuUnavailable, OSError, AttributeError, ValueError) as exc:
        log.warning("NPU backend unavailable (%s) - using CPU", exc)
        return None, str(exc)
