from pathlib import Path
import pandas as pd

from ga_trader.parity.tv_csv import load_tv_trades_csv

def test_tv_csv_comma_headers(tmp_path: Path):
    p = tmp_path / "tv.csv"
    df = pd.DataFrame({
        "Entry time":["2025-12-01 09:00"],
        "Exit time":["2025-12-01 09:05"],
        "Direction":["Long"],
        "Qty":["1"],
        "Entry price":["100.0"],
        "Exit price":["101.0"],
        "Profit":["1.0"],
    })
    df.to_csv(p, index=False)
    out = load_tv_trades_csv(p)
    assert len(out) == 1
    assert out.loc[0, "direction"] == "long"
    assert out.loc[0, "pnl"] == 1.0
    assert out.loc[0, "entry_time_ms"] is not None
    assert out.loc[0, "exit_time_ms"] is not None

def test_tv_csv_semicolon_korean_headers(tmp_path: Path):
    p = tmp_path / "tv_kr.csv"
    p.write_text(
        "진입시간;청산시간;방향;수량;진입가;청산가;손익\n"
        "2025-12-01 09:00;2025-12-01 09:05;롱;1;100,0;101,0;1,0\n",
        encoding="utf-8"
    )
    out = load_tv_trades_csv(p)
    assert len(out) == 1
    assert out.loc[0, "direction"] == "long"
    assert out.loc[0, "pnl"] == 1.0
