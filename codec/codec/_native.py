"""ctypes loader for the optional native IMA core.

The loader prefers a prebuilt ``_ima*.so`` shipped next to this file, then a
cached build under the user cache directory, and finally tries to compile
``_ima.c`` on the fly with the available C compiler.  If anything fails,
:func:`load` returns ``None`` and :mod:`codec.core` uses pure Python.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

BACKEND = "python-fallback"


def _package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_prebuilt():
    d = _package_dir()
    try:
        for name in sorted(os.listdir(d)):
            if name.startswith("_ima") and (
                name.endswith(".so") or name.endswith(".dylib") or name.endswith(".dll")
            ):
                return os.path.join(d, name)
    except OSError:
        pass
    return None


def _cache_paths():
    cdir = os.environ.get("CODEC_NATIVE_CACHE")
    if cdir:
        base = cdir
    else:
        root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        base = os.path.join(root, "adpcm_codec")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        base = tempfile.gettempdir()
    src = os.path.join(_package_dir(), "_ima.c")
    try:
        digest = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
    except OSError:
        digest = "nosrc"
    so = os.path.join(base, "_ima_%s_%s.so" % (sys.version_info[0], digest))
    return src, so


def _compile(src, so):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc or not os.path.exists(src):
        return False
    cmd = [cc, "-O3", "-fPIC", "-shared", "-std=c11", "-lm", src, "-o", so]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        )
        return proc.returncode == 0 and os.path.exists(so)
    except (OSError, subprocess.SubprocessError):
        return False


def _configure(lib):
    i64 = ctypes.c_int64
    pd = ctypes.POINTER(ctypes.c_double)
    pu8 = ctypes.POINTER(ctypes.c_ubyte)
    pi64 = ctypes.POINTER(ctypes.c_int64)

    lib.adpcm_check_finite.argtypes = [pd, i64, pi64]
    lib.adpcm_check_finite.restype = ctypes.c_int

    lib.adpcm_encode_primed.argtypes = [pd, i64, pu8, ctypes.c_int, ctypes.c_double, pd]
    lib.adpcm_encode_primed.restype = ctypes.c_int

    lib.adpcm_initial_index.argtypes = [pd, i64]
    lib.adpcm_initial_index.restype = ctypes.c_int

    lib.adpcm_decode_primed.argtypes = [
        ctypes.c_char_p, i64, pd, ctypes.c_double, ctypes.c_int, ctypes.c_double
    ]
    lib.adpcm_decode_primed.restype = None

    lib.adpcm_decode_standard.argtypes = [ctypes.c_char_p, i64, pd]
    lib.adpcm_decode_standard.restype = None

    lib.adpcm_init.argtypes = []
    lib.adpcm_init.restype = None
    try:
        lib.adpcm_init()
    except Exception:
        pass
    return lib


def load():
    """Return configured ctypes library, or None if unavailable."""
    global BACKEND
    path = _candidate_prebuilt()
    if path is None:
        try:
            src, so = _cache_paths()
            if not os.path.exists(so):
                _compile(src, so)
            if os.path.exists(so):
                path = so
        except Exception:
            path = None
    if path is None:
        BACKEND = "python-fallback"
        return None
    try:
        lib = ctypes.CDLL(path)
        _configure(lib)
        BACKEND = "native:" + os.path.basename(path)
        return lib
    except OSError:
        # A stale/cache binary may fail to load; try a fresh compile once.
        try:
            src, so = _cache_paths()
            try:
                os.remove(so)
            except OSError:
                pass
            if _compile(src, so):
                lib = ctypes.CDLL(so)
                _configure(lib)
                BACKEND = "native:rebuilt"
                return lib
        except Exception:
            pass
        BACKEND = "python-fallback"
        return None
