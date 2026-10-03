#!/bin/bash
# Create data/<NAME> symlinks to datasets that already exist on this machine.
#
# Symlinks only. This script never copies, never downloads, never writes into a dataset.
# There are already ~1.35 GB of redundant READ 2016 derivatives across sibling projects
# this exists so the project does not add another copy. See data/README.md.
#
# Usage:
#   scripts/setup_dataset_links.sh              create/refresh links from the registry
#   scripts/setup_dataset_links.sh --check      report only, change nothing
#   scripts/setup_dataset_links.sh --clean      remove the links (never the targets)
#
# Paths come from configs/datasets_registry.yaml. Nothing is hard-coded here.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
REGISTRY="$ROOT/configs/datasets_registry.yaml"
LINKDIR="${HAND_DATA_ROOT:-$ROOT/data}"
MODE="${1:-create}"

[ -f "$REGISTRY" ] || { echo "registry not found: $REGISTRY" >&2; exit 1; }

# --- read name/path/available from the registry -------------------------------------
entries=$(python3 - "$REGISTRY" <<'PY'
import os, sys, yaml
reg = yaml.safe_load(open(sys.argv[1]))
for name, d in reg.get("datasets", {}).items():
    if d.get("available") is not True:      # skip stubs and absent datasets
        continue
    p = d.get("path")
    if not p:
        continue
    # Registry rule 3: ${VAR} expands from the environment. Corpora outside this
    # repository are written as ${HAND_EXTERNAL_DATA}/<corpus>; an unset variable
    # leaves the literal in place, which fails the -d test below and is reported
    # as NOTARGET rather than silently linking to a nonsense path.
    p = os.path.expandvars(p)
    if not os.path.isabs(p):
        p = os.path.abspath(p)              # repo-relative entries; cwd is the repo root
    print(f"{name}\t{p}")
PY
)
[ -z "$entries" ] && { echo "no available datasets in the registry"; exit 0; }

if [ "$MODE" = "--clean" ]; then
    n=0
    while IFS=$'\t' read -r name path; do
        l="$LINKDIR/$name"
        # -L guarantees we only ever remove a symlink, never a real directory
        if [ -L "$l" ]; then rm "$l" && echo "  removed link  $l" && n=$((n+1)); fi
    done <<< "$entries"
    echo "removed $n link(s). No dataset was touched."
    exit 0
fi

[ "$MODE" = "--check" ] || mkdir -p "$LINKDIR"
printf "%-14s %-8s %s\n" "DATASET" "STATUS" "TARGET"
printf -- "-%.0s" {1..96}; echo

ok=0; missing=0; wrong=0
while IFS=$'\t' read -r name path; do
    link="$LINKDIR/$name"

    if [ ! -d "$path" ]; then
        printf "%-14s %-8s %s\n" "$name" "NOTARGET" "$path"
        missing=$((missing+1)); continue
    fi

    if [ -e "$link" ] && [ ! -L "$link" ]; then
        # a real directory sitting where a link belongs: refuse, never delete
        printf "%-14s %-8s %s\n" "$name" "REALDIR" "$link is a real directory - refusing to touch it"
        wrong=$((wrong+1)); continue
    fi

    current=$(readlink "$link" 2>/dev/null || true)
    if [ "$current" = "$path" ]; then
        printf "%-14s %-8s %s\n" "$name" "OK" "$path"
        ok=$((ok+1)); continue
    fi

    if [ "$MODE" = "--check" ]; then
        printf "%-14s %-8s %s\n" "$name" "MISSING" "$path"
        missing=$((missing+1))
    else
        [ -L "$link" ] && rm "$link"
        ln -s "$path" "$link"
        printf "%-14s %-8s %s\n" "$name" "LINKED" "$path"
        ok=$((ok+1))
    fi
done <<< "$entries"

echo
echo "linked/ok: $ok   missing: $missing   conflicts: $wrong"
echo
echo "Disk used by $LINKDIR (links only - should be a few kilobytes):"
du -sh "$LINKDIR" 2>/dev/null | sed 's/^/  /'
echo "Apparent size if these were copies:"
du -shL "$LINKDIR" 2>/dev/null | sed 's/^/  /'

[ "$wrong" -gt 0 ] && exit 2
exit 0
