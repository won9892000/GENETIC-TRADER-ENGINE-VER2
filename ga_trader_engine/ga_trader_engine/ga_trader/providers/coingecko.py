from __future__ import annotations
from typing import List
import requests

def fetch_top_marketcap_symbols(vs_currency: str = "usd", limit: int = 120) -> List[str]:
    '''
    CoinGecko markets endpoint returns market-cap ranked assets.
    No API key required for basic usage, but rate limits apply.
    '''
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": min(250, int(limit)),
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    syms = []
    for it in data:
        s = it.get("symbol")
        if s:
            syms.append(str(s).upper())
    return syms[:limit]
