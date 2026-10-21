"""Fetch pinned research sources and Blender; no Python packages or system changes."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def source(root, name, spec, plan=False):
    target = root / 'vendor' / name
    if target.exists():
        if run('git', '-C', str(target), 'rev-parse', 'HEAD') != spec['commit']:
            raise RuntimeError(f'{target}: другая ревизия; существующие файлы не изменены')
        if run('git', '-C', str(target), 'status', '--porcelain'):
            raise RuntimeError(f'{target}: локальные изменения; проверьте их вручную')
        print(f'{name}: нужная ревизия уже установлена', flush=True)
        return
    print(f'{name}: {spec["url"]} @ {spec["commit"]}', flush=True)
    if plan:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-', dir=target.parent) as scratch:
        clone = Path(scratch) / 'repo'
        run('git', 'init', str(clone))
        run('git', '-C', str(clone), 'remote', 'add', 'origin', spec['url'])
        run('git', '-C', str(clone), 'fetch', '--depth=1', 'origin', spec['commit'])
        run('git', '-C', str(clone), 'checkout', '--detach', 'FETCH_HEAD')
        run('git', '-C', str(clone), 'submodule', 'update', '--init', '--recursive', '--depth=1')
        clone.rename(target)


def unpack(archive, destination):
    # Python 3.12's data filter also checks link targets and strips unsafe modes.
    with tarfile.open(archive, 'r:xz') as bundle:
        bundle.extractall(destination, filter='data')


def blender(root, spec, plan=False):
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise RuntimeError('Этот bootstrap поддерживает Blender только для Linux x86_64')
    tools = root / 'tools'
    target = tools / f'blender-{spec["version"]}-linux-x64'
    link = tools / 'blender'
    if link.exists():
        version = run(str(link), '--version').splitlines()[0]
        if version.split()[1] != spec['version']:
            raise RuntimeError(f'{link}: найдена другая версия: {version}')
        print(f'{version}: уже установлен', flush=True)
        return
    if link.is_symlink():
        raise RuntimeError(f'{link}: повреждённая ссылка; проверьте её вручную')
    print(f'Blender: {spec["url"]}, SHA256 {spec["sha256"]}', flush=True)
    if plan:
        return
    tools.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f'{target}: каталог уже существует без tools/blender; проверьте вручную')
    with tempfile.TemporaryDirectory(prefix='.blender-', dir=tools) as scratch:
        archive = Path(scratch) / 'blender.tar.xz'
        with urllib.request.urlopen(spec['url'], timeout=60) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != spec['sha256']:
            raise RuntimeError('SHA256 Blender не совпал; архив не распакован')
        unpack(archive, scratch)
        (Path(scratch) / target.name).rename(target)
    link.symlink_to(target.name + '/blender')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=['sources', 'blender', 'all'], default='all')
    parser.add_argument('--plan', action='store_true', help='Показать действия без скачивания и записи')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'configs/upstream-lock.json').read_text())
    if args.component in {'sources', 'all'}:
        for name in ['stable-fast-3d', 'TextDeformer']:
            source(ROOT, name, lock[name], args.plan)
    if args.component in {'blender', 'all'}:
        blender(ROOT, lock['Blender'], args.plan)


if __name__ == '__main__':
    main()
