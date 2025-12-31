from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from ga_trader.utils import save_json, ensure_dir

def export_trades_json(path: Path, trades: List[Dict[str, Any]], meta: Dict[str, Any] | None = None):
    ensure_dir(path.parent)
    save_json(path, {"meta": meta or {}, "trades": trades})
