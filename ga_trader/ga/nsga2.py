from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import math

@dataclass
class Individual:
    idx: int
    objectives: List[float]
    rank: int = 0
    crowding: float = 0.0

def dominates(a: List[float], b: List[float], maximize: List[bool]) -> bool:
    # a dominates b if a is >= b for all (or <= if minimize) and strictly better in at least one
    better_or_equal_all = True
    strictly_better = False
    for i, (av, bv) in enumerate(zip(a, b)):
        if maximize[i]:
            if av < bv:
                better_or_equal_all = False
                break
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal_all = False
                break
            if av < bv:
                strictly_better = True
    return better_or_equal_all and strictly_better

def fast_nondominated_sort(objs: List[List[float]], maximize: List[bool]) -> List[List[int]]:
    n = len(objs)
    S = [[] for _ in range(n)]
    n_dom = [0]*n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q], maximize):
                S[p].append(q)
            elif dominates(objs[q], objs[p], maximize):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front: List[int], objs: List[List[float]]) -> Dict[int, float]:
    if not front:
        return {}
    m = len(objs[0])
    dist = {i: 0.0 for i in front}
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: objs[i][k])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        minv = objs[front_sorted[0]][k]
        maxv = objs[front_sorted[-1]][k]
        if maxv == minv:
            continue
        for j in range(1, len(front_sorted)-1):
            prevv = objs[front_sorted[j-1]][k]
            nextv = objs[front_sorted[j+1]][k]
            dist[front_sorted[j]] += (nextv - prevv) / (maxv - minv)
    return dist

def nsga2_select(scores: List[Dict[str, float]], objectives: List[Tuple[str, str]], pop_size: int) -> List[int]:
    '''
    scores: list of dict metrics for each individual
    objectives: list of (metric_key, direction) where direction in {"max","min"}
    returns indices of selected individuals
    '''
    maximize = [d == "max" for _, d in objectives]
    objs = []
    for s in scores:
        o = []
        for k, d in objectives:
            v = float(s.get(k, 0.0))
            o.append(v)
        objs.append(o)

    fronts = fast_nondominated_sort(objs, maximize=maximize)

    selected = []
    for f_rank, front in enumerate(fronts):
        cd = crowding_distance(front, objs)
        front_sorted = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
        if len(selected) + len(front_sorted) <= pop_size:
            selected.extend(front_sorted)
        else:
            needed = pop_size - len(selected)
            selected.extend(front_sorted[:needed])
            break
    return selected
