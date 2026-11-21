#!/usr/bin/env bash
# Execute only on an Ubuntu VM with sudo access. Writes only under /tmp.
set -euo pipefail
BASE=/tmp/pr12_level_a
rm -rf -- "$BASE"
mkdir -p "$BASE/docs" "$BASE/images" "$BASE/big"
chmod 755 "$BASE" "$BASE/docs" "$BASE/images" "$BASE/big"
printf 'Example text\n' > "$BASE/docs/original.txt"
printf 'another txt\n' > "$BASE/docs/notes.txt"
truncate -s 2M "$BASE/big/large.bin"
echo '=== 1. Directory and inode ==='
ls -li "$BASE" "$BASE/docs"
echo '=== 2. Links before deletion ==='
ln "$BASE/docs/original.txt" "$BASE/docs/hard.txt"
ln -s "$BASE/docs/original.txt" "$BASE/docs/symbolic.txt"
ls -li "$BASE/docs"
rm "$BASE/docs/original.txt"
echo '=== 2. Links after deletion ==='
ls -li "$BASE/docs"
cat "$BASE/docs/hard.txt"
if cat "$BASE/docs/symbolic.txt"; then echo 'Unexpected: symlink still readable'; else echo 'Expected: broken symbolic link'; fi
echo '=== 3. Permissions ==='
printf 'private example\n' > "$BASE/docs/permissions.txt"
chmod u=rw,g=r,o= "$BASE/docs/permissions.txt"
ls -l "$BASE/docs/permissions.txt"
chmod 640 "$BASE/docs/permissions.txt"
ls -l "$BASE/docs/permissions.txt"
if ! command -v sudo >/dev/null; then
  echo 'sudo is required for user-check segment' >&2; exit 1
fi
sudo useradd -M pr12_other 2>/dev/null || true
sudo useradd -M pr12_group 2>/dev/null || true
# Set group to the second test user's primary group while preserving file owner.
sudo chgrp "$(id -gn pr12_group)" "$BASE/docs/permissions.txt"
chmod 640 "$BASE/docs/permissions.txt"
echo 'Group member (should succeed):'
sudo -u pr12_group cat "$BASE/docs/permissions.txt"
echo 'Other user (should fail):'
if sudo -u pr12_other cat "$BASE/docs/permissions.txt"; then
  echo 'ERROR: other user could read file'; exit 1
else
  echo 'Expected: Permission denied'
fi
echo '=== 4. Find recent files >1MiB ==='
find "$BASE" -type f -size +1M -mtime -1
find "$BASE" -type f -name '*.txt' -print0 | xargs -0 -r wc -c
printf '\nDone. Copy actual terminal output to README.md.\n'
