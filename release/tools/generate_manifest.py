#!/usr/bin/env python3
"""Regenerate release/MANIFEST.md from the working tree. Run from the repository root."""
import hashlib, os, datetime, subprocess, sys

HEADER = """# MANIFEST

Every file published in this repository, with its SHA-256 **as published here**. Nothing
else: no private path, no internal document, no staging artefact. This file is the only
exclusion, since it cannot carry its own digest.

**The digests below are the ones a reader can check.** Where a published evidence artefact
was redacted before release (paths, a hostname, a user name, see below) its
digest differs from the development archive's, and the digest recorded here is the
post-redaction one. `docs/REPRODUCIBILITY_ARTIFACTS.md` section 5 lists every redaction and
what it touched; each is metadata-only and changes no measured value.

Two digests in this repository deliberately do **not** agree, and that is explained rather
than reconciled: the `effective_code_state.per_file_sha256` block inside each published run
record is the digest of the source **as it executed**, before the CeCILL-C notices of
`release/NOTICE.md` section 1 were restored. Those restorations are comment-only, so the
behaviour each record describes is unchanged. See `docs/REPRODUCIBILITY_ARTIFACTS.md`
section 5.

Verify the whole tree from the repository root:

```bash
sed -nE 's/^\\| `([^`]+)` \\| `([0-9a-f]{64})`.*/\\2  \\1/p' release/MANIFEST.md > /tmp/hand.sha256
sha256sum -c /tmp/hand.sha256
```
"""

GROUPS = [
 ('Root — project metadata, environment, licence', lambda p: '/' not in p),
  ('Documentation', lambda p: p.startswith('docs/')),
 ('Library — hand/', lambda p: p.startswith('hand/')),
 ('Tests — tests/', lambda p: p.startswith('tests/')),
 ('Entry points — tools/', lambda p: p.startswith('tools/')),
 ('Reproducibility scripts — scripts/', lambda p: p.startswith('scripts/')),
 ('Configuration — configs/', lambda p: p.startswith('configs/')),
 ('Release payload — licences, notices, inference contract, model configuration', lambda p: p.startswith('release/')),
 ('Reproducibility artefacts — run records, profiling, split manifests', lambda p: p.startswith('experiments/')),
 ('Reproducibility artefacts — measured metrics', lambda p: p.startswith('results_real/') or p.startswith('results_recovery/')),
 ('Stub directories — README only; contents are git-ignored', lambda p: p.startswith('data/') or p.startswith('models/')),
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

SKIP_DIRS = ('.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
             '.venv', 'venv', 'env', 'ENV', '.idea', '.vscode', 'wandb', 'mlruns',
             'lightning_logs', 'node_modules', '.eggs', 'build', 'dist')


def tracked_files():
    """Prefer git's own view: a release manifest should list exactly what is published.

    Falls back to a filesystem walk when this is not a git checkout, so the script
    still works from an unpacked archive.
    """
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], capture_output=True, check=True)
        names = [n for n in out.stdout.decode().split('\0') if n]
        if names:
            return [os.path.normpath(n) for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for root, dirs, fs in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in fs:
            found.append(os.path.normpath(os.path.join(root, f)))
    return found


def main():
    files = sorted(p for p in tracked_files()
                   if not p.endswith(('.pyc', '.egg-info'))
                   and p != 'release/MANIFEST.md')

    body, assigned, total = [], set(), 0
    for title, pred in GROUPS:
        sel = [p for p in files if p not in assigned and pred(p)]
        if not sel:
            continue
        assigned.update(sel)
        body += ['\n## %s\n' % title, '| File | SHA-256 | Bytes |', '|---|---|---:|']
        for p in sel:
            if not os.path.exists(p):   # tracked in git but deleted in the working tree
                continue
            n = os.path.getsize(p)
            total += n
            body.append('| `%s` | `%s` | %s |' % (p, sha256(p), format(n, ',')))

    stray = [p for p in files if p not in assigned]
    if stray:
        print('unclassified paths, refusing to write:', stray, file=sys.stderr)
        return 1

    out = [HEADER.rstrip(),
           '\n**%d files, %s bytes.** Generated %s. Regenerate with this script after any'
           ' change to the tree.\n' % (len(files), format(total, ','), datetime.date.today().isoformat())]
    open('release/MANIFEST.md', 'w', encoding='utf-8').write('\n'.join(out + body) + '\n')
    print('MANIFEST.md: %d files, %s bytes' % (len(files), format(total, ',')))
    return 0

if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
