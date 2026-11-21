#!/usr/bin/env bash
# Destructive only to a new isolated 16MiB loop image under /tmp; requires sudo.
set -euo pipefail
WORK="$(mktemp -d /tmp/pr12_inode.XXXXXX)"
IMAGE="$WORK/pr12.img"
MOUNT="$WORK/mnt"
mkdir -p "$MOUNT"
cleanup() {
 mountpoint -q "$MOUNT" && sudo umount "$MOUNT" || true
 rm -rf -- "$WORK"
}
trap cleanup EXIT
truncate -s 16M "$IMAGE"
mkfs.ext4 -F -N 128 -E "root_owner=$(id -u):$(id -g)" "$IMAGE" >/dev/null
sudo mount -o loop "$IMAGE" "$MOUNT"
echo '=== BEFORE ==='
df -h "$MOUNT"
df -i "$MOUNT"
count=0
while touch "$MOUNT/file_$count" 2>/dev/null; do count=$((count+1)); done
printf 'Created %s empty files before inode exhaustion\n' "$count"
echo '=== AFTER ==='
df -h "$MOUNT"
df -i "$MOUNT"
