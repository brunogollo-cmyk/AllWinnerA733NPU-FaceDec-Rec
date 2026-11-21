#!/usr/bin/env bash
# Install the Allwinner A733 VIPLite userspace runtime used by facegate.
#
# Usage:
#   sudo ./tools/install_npu_runtime.sh            # install to /usr/local/lib
#   sudo DEST=/usr/lib ./tools/install_npu_runtime.sh
#   ./tools/install_npu_runtime.sh --check         # only verify, change nothing
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/npu_runtime/A733"
DEST="${DEST:-/usr/local/lib}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

check() {
    local ok=1
    say "Kernel module / device"
    if [ -e /dev/vipcore ]; then
        echo "  /dev/vipcore present"
    else
        warn "/dev/vipcore missing: the 'vipcore' kernel module is not loaded/present."
        warn "Try: sudo modprobe vipcore (it must come with your board image/SDK)."
        ok=0
    fi

    say "Userspace runtime"
    local found=""
    for dir in "$DEST" /usr/local/lib /usr/lib; do
        if [ -e "$dir/libNBGlinker.so" ]; then
            found="$dir"
            break
        fi
    done
    if [ -n "$found" ]; then
        echo "  found libNBGlinker.so in $found"
        if [ -e "$found/libVIPhal.so" ] || ldconfig -p 2>/dev/null | grep -q libVIPhal.so; then
            echo "  libVIPhal.so resolvable"
        else
            warn "libVIPhal.so not found next to libNBGlinker.so"
            ok=0
        fi
    else
        warn "libNBGlinker.so not found (run without --check to install it)"
        ok=0
    fi

    if [ "$ok" = 1 ]; then
        say "NPU runtime looks usable"
    else
        warn "NPU will not be used until the points above are fixed; CPU still works."
    fi
}

if [ "${1:-}" = "--check" ]; then
    check
    exit 0
fi

[ -f "$SRC/lib/libNBGlinker.so" ] || die "runtime files not found under $SRC"
[ "$(id -u)" = 0 ] || die "run as root (sudo) or use DEST=<dir you own>"

say "Installing VIPLite libraries into $DEST"
install -d -m 0755 "$DEST"
install -m 0644 "$SRC/lib/libNBGlinker.so" "$DEST/libNBGlinker.so"
install -m 0644 "$SRC/lib/libVIPhal.so" "$DEST/libVIPhal.so"

if [ "$DEST" = "/usr/lib" ] || [ "$DEST" = "/usr/local/lib" ]; then
    say "Refreshing the dynamic linker cache"
    ldconfig
else
    warn "DEST is not in the linker search path; add it to /etc/ld.so.conf.d/ and run ldconfig."
fi

check
