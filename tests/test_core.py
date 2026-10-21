from pathlib import Path
from gai.core import local_path


def test_virtualenv_executable_symlink_is_not_dereferenced(tmp_path):
    base=tmp_path/'base-python';base.write_text('not a real interpreter')
    link=tmp_path/'venv-python';link.symlink_to(base)
    assert local_path(link)==link
    assert local_path(link)!=base
