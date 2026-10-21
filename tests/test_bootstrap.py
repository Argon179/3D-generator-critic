import io
from pathlib import Path
import subprocess
import tarfile

import pytest
from scripts.bootstrap import source, unpack, blender


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def test_sources_install_repeat_and_preserve_changes(tmp_path):
    upstream = tmp_path/'upstream'
    git('init', upstream)
    (upstream/'model.py').write_text('original')
    git('-C', upstream, 'add', '.')
    git('-C', upstream, '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-m', 'fixture')
    spec={'url': str(upstream), 'commit': git('-C', upstream, 'rev-parse', 'HEAD')}
    root=tmp_path/'project'
    source(root, 'example', spec, plan=True)
    assert not root.exists()
    source(root, 'example', spec)
    source(root, 'example', spec)
    model=root/'vendor/example/model.py'
    model.write_text('local changes')
    with pytest.raises(RuntimeError, match='локальные изменения'):
        source(root, 'example', spec)
    assert model.read_text()=='local changes'


def test_archive_rejects_escape(tmp_path):
    archive=tmp_path/'bad.tar.xz'
    with tarfile.open(archive, 'w:xz') as bundle:
        info=tarfile.TarInfo('../escaped');info.size=4
        bundle.addfile(info, io.BytesIO(b'bad!'))
    with pytest.raises(tarfile.TarError):
        unpack(archive, tmp_path/'output')
    assert not (tmp_path/'escaped').exists()


def test_blender_checksum_install_and_repeat(tmp_path):
    import hashlib
    archive=tmp_path/'fixture.tar.xz'
    payload=b'#!/bin/sh\nprintf "Blender 4.5.14 LTS\\n"\n'
    with tarfile.open(archive, 'w:xz') as bundle:
        info=tarfile.TarInfo('blender-4.5.14-linux-x64/blender')
        info.size=len(payload);info.mode=0o755
        bundle.addfile(info, io.BytesIO(payload))
    spec={'version':'4.5.14','url':archive.as_uri(),'sha256':'0'*64}
    root=tmp_path/'project'
    blender(root,spec,plan=True)
    assert not root.exists()
    with pytest.raises(RuntimeError, match='SHA256'):
        blender(root,spec)
    assert not (root/'tools/blender').exists()
    spec['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    blender(root,spec)
    assert (root/'tools/blender').is_symlink()
    blender(root,spec)
