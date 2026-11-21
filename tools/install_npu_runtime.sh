#!/usr/bin/env bash
# Install the Allwinner A733 VIPLite runtime used by facegate.
#
# Usage:
#   sudo ./tools/install_npu_runtime.sh                    # userspace libs -> /usr/local/lib
#   sudo DEST=/usr/lib ./tools/install_npu_runtime.sh      # libs -> /usr/lib
#   sudo ./tools/install_npu_runtime.sh --with-kernel-module
#                                                          # also install/load vipcore.ko
#                                                          # (only if it matches this kernel)
#   ./tools/install_npu_runtime.sh --check                 # verify only, change nothing
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/npu_runtime/A733"
DEST="${DEST:-/usr/local/lib}"
KERNEL_KO="$SRC/kernel/vipcore.ko"
WITH_KO=0

for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --with-kernel-module) WITH_KO=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

ko_vermagic() {
    [ -f "$1" ] || return 1
    strings -a "$1" 2>/dev/null | sed -n 's/^vermagic=//p' | head -1
}

ko_matches_kernel() {
    local file=$1 vm
    vm="$(ko_vermagic "$file")" || return 1
    [ -n "$vm" ] || return 1
    # vermagic starts with the kernel release, e.g.
    # "6.6.98-sun60iw2 SMP preempt mod_unload aarch64"
    case "$vm" in
        "$(uname -r) "*) ;;
        *) return 1 ;;
    esac
    case "$vm" in
        *aarch64*) ;;
        *) return 1 ;;
    esac
    return 0
}

check() {
    local ok=1
    say "Kernel module / device"
    if [ -e /dev/vipcore ]; then
        echo "  /dev/vipcore present"
        if lsmod 2>/dev/null | grep -q '^vipcore'; then
            echo "  vipcore loaded"
        fi
    else
        warn "/dev/vipcore missing: the 'vipcore' kernel module is not loaded/present."
        if [ -f "$KERNEL_KO" ] && ko_matches_kernel "$KERNEL_KO"; then
            warn "The bundled module matches this kernel; install it with:"
            warn "  sudo $0 --with-kernel-module"
        else
            warn "The bundled vipcore.ko does not match kernel $(uname -r)."
            warn "Use the module/package that ships with your board image."
        fi
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

if [ "${CHECK_ONLY:-0}" = 1 ]; then
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

if [ "$WITH_KO" = 1 ]; then
    say "Installing the vipcore kernel module"
    [ -f "$KERNEL_KO" ] || die "bundled module not found: $KERNEL_KO"
    vm="$(ko_vermagic "$KERNEL_KO")"
    echo "  module vermagic: $vm"
    echo "  running kernel : $(uname -r)"
    if ! ko_matches_kernel "$KERNEL_KO"; then
        die "module does not match this kernel; refusing to install it.
Use the vipcore module that ships with your board image (e.g. the
linux-image-current-* package) instead."
    fi
    if lsmod 2>/dev/null | grep -q '^vipcore'; then
        echo "  vipcore is already loaded; skipping install"
    else
        kdir="/lib/modules/$(uname -r)/extra"
        install -d -m 0755 "$kdir"
        install -m 0644 "$KERNEL_KO" "$kdir/vipcore.ko"
        command -v depmod >/dev/null && depmod -a
        modprobe vipcore || die "modprobe vipcore failed; check dmesg"
        echo "  loaded"
    fi
    if [ -e /dev/vipcore ]; then
        echo "  /dev/vipcore created"
    else
        warn "/dev/vipcore still missing after modprobe; check dmesg"
    fi
    echo "  (to load at boot: echo vipcore | sudo tee /etc/modules-load.d/vipcore.conf)"
fi

check
