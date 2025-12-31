from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import re

INPUT_PATTERNS = [
    # input.int(14, "Length", minval=1, maxval=100)
    re.compile(r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*input\.(?P<typ>int|float|bool|string)\((?P<args>[^\)]*)\)"),
]

def _infer_id(p: Path) -> str:
    s = p.stem
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s.lower()

def harvest_pine_to_spec(pine_path: Path, indicator_id: Optional[str] = None) -> Dict[str, Any]:
    txt = pine_path.read_text(encoding="utf-8", errors="ignore")

    # basic metadata heuristics
    # //@version=5
    # indicator("Name", ...)
    name = indicator_id or _infer_id(pine_path)
    m = re.search(r'\bindicator\("([^"]+)"', txt)
    if m:
        name = m.group(1)

    # attempt to capture author/license from comments
    author = ""
    lic = ""
    for line in txt.splitlines()[:60]:
        if "author" in line.lower():
            author = author or line.strip().lstrip("/# ").replace("Author:", "").strip()
        if "license" in line.lower():
            lic = lic or line.strip().lstrip("/# ").replace("License:", "").strip()

    params = {}
    for pat in INPUT_PATTERNS:
        for m in pat.finditer(txt):
            var = m.group("var")
            typ = m.group("typ")
            args = m.group("args")
            # first arg is default value
            parts = [a.strip() for a in args.split(",") if a.strip()]
            default = parts[0] if parts else None
            minv = re.search(r"minval\s*=\s*([0-9\.\-]+)", args)
            maxv = re.search(r"maxval\s*=\s*([0-9\.\-]+)", args)
            params[var] = {
                "type": typ,
                "default": default,
            }
            if minv:
                params[var]["min"] = float(minv.group(1)) if typ == "float" else int(float(minv.group(1)))
            if maxv:
                params[var]["max"] = float(maxv.group(1)) if typ == "float" else int(float(maxv.group(1)))

    spec = {
        "indicator_id": indicator_id or _infer_id(pine_path),
        "name": name,
        "description": "DRAFT: harvested from Pine source; fill outputs/python_impl/pine_snippet.",
        "author": author,
        "license": lic,
        "source": str(pine_path),
        "timeframe": "trigger",
        "parameters": params,
        "outputs": {},  # must be filled by user
        "python_impl": "",
        "pine_snippet": "",
    }
    return spec
