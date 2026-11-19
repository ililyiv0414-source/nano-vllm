"""Archive source and raw evidence without rerunning GPU workloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='New .tar.gz path outside the source directory')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(a.out).resolve()
    if target == root or root in target.parents or target.exists():
        p.error('Output must be new and outside source directory')
    files = []
    for name in ('nanovllm', 'experiments', 'tests', 'docs', 'results'):
        folder = root / name
        if folder.exists():
            files.extend(f for f in folder.rglob('*') if f.is_file()
                         and '__pycache__' not in f.parts and f.suffix != '.pyc')
    files.extend(f for f in root.iterdir() if f.is_file() and
                 (f.suffix in ('.md', '.toml', '.txt', '.json') or f.name == '.gitignore'))
    files = sorted(set(files))
    if not (root / 'results').exists():
        p.error('No results directory; run this on the GPU server source checkout')
    def command(args):
        try:
            r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=30)
            return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            return dict(error=str(e))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                    python=sys.version, executable=sys.executable,
                    pip=command([sys.executable, '-m', 'pip', 'freeze']),
                    git_head=command(['git', 'rev-parse', 'HEAD']),
                    git_status=command(['git', 'status', '--short']),
                    note='Current snapshot, not proof of the exact historical worker source. Worker JSON fingerprints retained. Model weights and virtualenv excluded.',
                    files=[])
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting existing archives.
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for f in files:
            digest = hashlib.sha256()
            with f.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            relative = f.relative_to(root).as_posix()
            manifest['files'].append(dict(path=relative, bytes=f.stat().st_size, sha256=digest.hexdigest()))
            tar.add(f, arcname='source/' + relative, recursive=False)
        import io
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256()
    with target.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    checksum = digest.hexdigest()
    target.with_name(target.name + '.sha256').write_text(checksum + '  ' + target.name + '\n', encoding='utf-8')
    print(json.dumps(dict(archive=str(target), files=len(files), bytes=target.stat().st_size,
                          sha256=checksum), indent=2))


if __name__ == '__main__':
    main()
