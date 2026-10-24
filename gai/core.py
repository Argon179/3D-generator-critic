from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import shlex
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CONFIG = ROOT / "configs/project.json"
STATE_LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return json.loads(CONFIG.read_text())


def local_path(value):
    p = Path(value).expanduser()
    # Resolving the venv's python symlink would run the base interpreter instead,
    # losing all isolated packages. Absolutize without dereferencing symlinks.
    return Path(os.path.abspath(p if p.is_absolute() else ROOT / p))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_dir(run_id):
    if not re.fullmatch(r"[a-f0-9]{12}", run_id):
        raise ValueError("Некорректный идентификатор запуска")
    return RUNS / run_id


def read_run(run_id):
    with STATE_LOCK:
        return json.loads((run_dir(run_id) / "state.json").read_text())


def update_run(run_id, **fields):
    with STATE_LOCK:
        state = read_run(run_id)
        state.update(fields, updated_at=now())
        save_json(run_dir(run_id) / "state.json", state)
    return state


def event(run_id, message):
    with STATE_LOCK:
        state = read_run(run_id)
        state.setdefault("events", []).append({"time": now(), "message": message})
        save_json(run_dir(run_id) / "state.json", state)


def library():
    base = ROOT / "reference_images/multiview"
    groups = []
    labels = {"rubber_duck": ("Уточка", "a yellow rubber duck"), "lucky_cat": ("Кот", "a white maneki-neko cat figurine"), "frog": ("Лягушка", "a green frog figurine"), "leopard": ("Леопард", "a toy leopard"), "cad_spur_gear": ("Шестерня · 6 видов + изометрия", "a spur gear with a central bore"), "cad_bearing_pedestal": ("Корпус подшипника · 6 видов + изометрия", "a bearing pedestal housing"), "cad_ring_flange": ("Фланец · 6 видов + изометрия", "a ring flange"), "gear": ("Шестерня", "a blue spur gear"), "nut": ("Гайка", "a hexagonal nut")}
    manifest = {}
    if (base / "sources.json").exists():
        manifest = {r["file"]: r for r in json.loads((base / "sources.json").read_text())}
    for parent in [base / "figurines", base / "engineering", ROOT / "reference_images/uploads"]:
        if not parent.exists():
            continue
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir():
                continue
            images = []
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                rel = p.relative_to(ROOT).as_posix()
                m = manifest.get(p.relative_to(base).as_posix(), {}) if p.is_relative_to(base) else {}
                images.append({"path": rel, "azimuth_deg": m.get("azimuth_deg"), "view_label":m.get("view_label")})
            if images:
                label, prompt = labels.get(folder.name, (folder.name, ""))
                groups.append({"id": folder.relative_to(ROOT).as_posix(), "name": label, "prompt": prompt, "images": images, "front_azimuth_deg": {"rubber_duck":270,"lucky_cat":0}.get(folder.name), "warning": "128 × 128: только грубая форма" if folder.name in {"rubber_duck", "lucky_cat", "frog"} else "Нужно выделить деталь на платформе" if folder.name in {"gear", "nut"} else ""})
    return groups


def child_env():
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / ".cache/huggingface"), TORCH_HOME=str(ROOT / ".cache/torch"), TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/extensions"), U2NET_HOME=str(ROOT / ".cache/rembg"), HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", MAX_JOBS="2")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # NVCC 12.0 needs a compatible host compiler. Keep the wrapper path space-free
    # because torch/ninja use CXX inside shell commands.
    compiler = ROOT / 'tools/compiler/root/usr'
    if (compiler/'bin/x86_64-linux-gnu-g++-12').exists():
        tag = hashlib.sha256(str(ROOT).encode()).hexdigest()[:10]
        wrapper = Path('/tmp') / f'gai-cxx12-{os.getuid()}-{tag}'
        lib = compiler/'lib/gcc/x86_64-linux-gnu/12'
        args = [str(compiler/'bin/x86_64-linux-gnu-g++-12'), '-B'+str(lib)+'/', '-B/usr/lib/gcc/x86_64-linux-gnu/12/', '-isystem',str(compiler/'include/c++/12'), '-isystem',str(compiler/'include/x86_64-linux-gnu/c++/12'), '-isystem',str(compiler/'include/c++/12/backward')]
        script='#!/bin/sh\nexec '+shlex.join(args)+' "$@"\n'
        if not wrapper.exists() or wrapper.read_text()!=script:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',dir='/tmp',prefix='gai-cxx-',delete=False) as f:
                f.write(script);temporary=Path(f.name)
            temporary.chmod(0o700);temporary.replace(wrapper)
        env.update(CC=str(wrapper), CXX=str(wrapper), CUDAHOSTCXX=str(wrapper))
        env['TORCH_EXTENSIONS_DIR'] = f'/tmp/gai-extensions-{os.getuid()}-{tag}'
    return env


class Cancelled(Exception):
    pass


def command(run_id, args, cwd=None, timeout=3600):
    """One subprocess at a time; exiting releases all model VRAM."""
    import signal
    import time

    folder = run_dir(run_id)
    if (folder / "cancel").exists():
        raise Cancelled()
    started = time.monotonic()
    with (folder / "process.log").open("a", encoding="utf-8") as log:
        log.write("\n$ " + repr([str(x) for x in args]) + "\n")
        log.flush()
        env=child_env()
        env['PATH']=str(Path(args[0]).parent)+os.pathsep+env['PATH']
        proc = subprocess.Popen([str(x) for x in args], cwd=cwd or ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if (folder / "cancel").exists():
                    raise Cancelled()
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Превышено время этапа. Подробности в process.log")
                time.sleep(0.25)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if proc.returncode:
        raise RuntimeError(f"Этап завершился с кодом {proc.returncode}. Откройте process.log")


def weight_checks(cfg=None):
    cfg=cfg or config()
    checks=[]
    models=[('sf3d',1),('qwen',2)]+([('qwen_review',2)] if cfg.get('qwen_review_model') else [])
    for prefix,count in models:
        name=cfg[prefix+'_model'].replace('/','--')
        snapshot=ROOT/'.cache/huggingface/hub'/('models--'+name)/'snapshots'/cfg[prefix+'_revision']
        present=[p for p in snapshot.glob('*.safetensors') if p.is_file()]
        checks.append({'name':'Веса '+prefix.upper(),'ok':len(present)==count,'detail':f'{len(present)} из {count} файлов в локальном кэше'})
    return checks


def multiview_checks(backend, cfg=None):
    cfg=cfg or config()
    model=cfg[backend+'_model']
    from gai.multiview import cached_snapshot
    files=(['config.yaml','model.fp16.safetensors'] if backend=='hunyuan' else ['config.json','model.safetensors'])
    sub=cfg['hunyuan_subfolder'] if backend=='hunyuan' else ''
    ready=cached_snapshot(ROOT/'.cache/huggingface',model,[str(Path(sub)/f) for f in files],cfg.get(backend+'_revision')) is not None
    return [{'name':backend+' · окружение','ok':local_path(cfg[backend+'_python']).is_file() and (local_path(cfg[backend+'_python']).parent.parent/'ready.json').is_file(),'detail':'Изолированное окружение; готовность после проверки импортов'},
            {'name':backend+' · исходный код','ok':local_path(cfg[backend+'_repo']).is_dir(),'detail':cfg[backend+'_repo']},
            {'name':backend+' · веса','ok':ready,'detail':model}]


def doctor():
    cfg = config()
    checks = []
    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
        check("GPU", r.returncode == 0, r.stdout.strip() if r.returncode == 0 else "CUDA недоступна из этого процесса. Запустите приложение из обычного терминала.")
    except (OSError, subprocess.TimeoutExpired):
        check("GPU", False, "nvidia-smi недоступен")
    blender = local_path(cfg["blender"])
    check("Blender EEVEE", blender.is_file(), str(blender))
    probes = {"SF3D": ("sf3d_python", "sf3d_repo", "import sys; sys.path.insert(0, sys.argv[1]); import sf3d.system, rembg, texture_baker, uv_unwrapper; import torch; assert torch.cuda.is_available(); print('CUDA и импорты OK')"), "Qwen3-VL": ("qwen_python", None, "import torch, bitsandbytes; from transformers import Qwen3VLForConditionalGeneration; assert torch.cuda.is_available(); print('CUDA и импорты OK')"), "TextDeformer": ("textdeformer_python", "textdeformer_repo", "import clip, igl, cholespy, torch_sparse, torch_scatter, pymeshlab, nvdiffrast.torch; import torch; assert torch.cuda.is_available(); print('CUDA и импорты OK')")}
    for name, (py, repo, code) in probes.items():
        executable = local_path(cfg[py])
        if not executable.exists():
            check(name, False, "Не установлено окружение; см. README")
            continue
        try:
            args = [str(executable), "-c", code]
            if repo:
                args.append(str(local_path(cfg[repo])))
            env=child_env();env['PATH']=str(executable.parent)+os.pathsep+env['PATH']
            r = subprocess.run(args, capture_output=True, text=True, timeout=60, env=env)
            lines = (r.stdout + r.stderr).strip().splitlines()
            check(name, r.returncode == 0, "Импорты и CUDA доступны" if r.returncode == 0 else "\n".join(lines[-4:])[-900:])
        except (OSError, subprocess.TimeoutExpired) as e:
            check(name, False, str(e))
    optional={name:multiview_checks(name,cfg) for name in ('hunyuan','da3')}
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    check("Свободное место", free >= 12, f"{free:.1f} ГБ; веса скачиваются отдельно")
    checks.extend(weight_checks(cfg))
    return {"optional_generators":optional,"checks": checks, "ready": all(x["ok"] for x in checks), "note": "Импорты не подтверждают наличие весов. Первый запуск может скачать веса; SF3D требует доступа Hugging Face."}
