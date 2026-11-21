#!/usr/bin/env bash
# Requires an Ubuntu VM with mkfs.ext4, debugfs, sudo and loop mounts.
set -euo pipefail
WORK="$(mktemp -d /tmp/pr12_ext4.XXXXXX)"
IMAGE="$WORK/pr12.img"
MOUNT="$WORK/mnt"
mkdir -p "$MOUNT"
cleanup() {
  mountpoint -q "$MOUNT" && sudo umount "$MOUNT" || true
  rm -rf -- "$WORK"
}
trap cleanup EXIT
truncate -s 16M "$IMAGE"
mkfs.ext4 -F -N 128 "$IMAGE" >/dev/null
sudo mount -o loop "$IMAGE" "$MOUNT"
echo 'ext4 data\n' | sudo tee "$MOUNT/sample.txt" >/dev/null
sync
sudo umount "$MOUNT"
echo '=== Superblock ==='
debugfs -R stats "$IMAGE" 2>&1 | head -55
echo '=== File inode ==='
debugfs -R 'stat /sample.txt' "$IMAGE" 2>&1
echo '=== Extents ==='
debugfs -R 'extents /sample.txt' "$IMAGE" 2>&1
printf '\nCopy actual output into README. This image is deleted on exit.\n'
