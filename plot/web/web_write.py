from __future__ import annotations

import json
import os


def _template_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "tamplate")


def _ensure_web_dir(out_dir: str) -> str:
    web_dir = out_dir
    os.makedirs(web_dir, exist_ok=True)
    os.makedirs(os.path.join(web_dir, "js"), exist_ok=True)
    return web_dir


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def write_file(out_dir: str, name: str) -> None:
    web_dir = _ensure_web_dir(out_dir)
    src = os.path.join(_template_dir(), name)
    dst = os.path.join(web_dir, name)
    _write_text(dst, _read_text(src))


def write_data_js(out_dir: str, data: dict) -> None:
    web_dir = _ensure_web_dir(out_dir)
    s = json.dumps(data, ensure_ascii=False)
    js = f"window.__THREE_VIEWER_DATA__ = {s};\n"
    _write_text(os.path.join(web_dir, "data.js"), js)