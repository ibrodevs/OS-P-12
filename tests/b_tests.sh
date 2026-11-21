#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src/sub" "$TMP/out" "$TMP/dup"
printf 'first\n' > "$TMP/src/sub/a.txt"
chmod 640 "$TMP/src/sub/a.txt"
bash "$ROOT/backup.sh" backup "$TMP/src" "$TMP/out"
test "$(cat "$TMP/out/sub/a.txt")" = first
test "$(stat -c '%a' "$TMP/out/sub/a.txt")" = 640
before="$(sha256sum "$TMP/out/sub/a.txt" | cut -d' ' -f1)"
bash "$ROOT/backup.sh" backup "$TMP/src" "$TMP/out"
test "$(sha256sum "$TMP/out/sub/a.txt" | cut -d' ' -f1)" = "$before"
printf 'second\n' > "$TMP/out/sub/a.txt"
bash "$ROOT/backup.sh" restore "$TMP/out" "$TMP/src"
test "$(cat "$TMP/src/sub/a.txt")" = second
test "$(cat "$TMP/src/sub/a.txt.before_restore")" = first
bash "$ROOT/backup.sh" restore "$TMP/out" "$TMP/src"
test "$(cat "$TMP/src/sub/a.txt.before_restore")" = first
if bash "$ROOT/backup.sh" backup "$TMP/src" "$TMP/src/nested" 2>/dev/null; then
 echo 'ERROR: accepted nested backup'; exit 1
fi
printf 'duplicate payload\n' > "$TMP/dup/a"
cp "$TMP/dup/a" "$TMP/dup/b"
chmod 600 "$TMP/dup/a" "$TMP/dup/b"
python3 "$ROOT/dedup.py" "$TMP/dup"
test "$(stat -c '%i' "$TMP/dup/a")" = "$(stat -c '%i' "$TMP/dup/b")"
python3 "$ROOT/dedup.py" "$TMP/dup"
printf 'OK: B tests passed\n'
