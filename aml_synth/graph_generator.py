"""Synthetic AML transaction-graph generator (container: aml-synth-worker)  -  v0.2

Builds a directed multi-hop transaction network made of

  * benign background traffic (heavy-tailed activity, mostly domestic, realistic amounts),
  * BENIGN LOOK-ALIKES ("hard negatives") that resemble laundering typologies on the surface:
    merchants with many customers, payroll, cash-heavy businesses, rotating savings groups (chamas),
    remittance senders, cross-border traders, young businesses, supply chains, and
  * injected laundering patterns for 5 typologies:
      smurfing, scatter_gather, cyclic_loop, shell_company, cross_border_velocity
    Laundering accounts also carry ordinary traffic ("camouflage") and half of the mules are ordinary
    accounts that were recruited, so no single account looks guilty on its own. What gives them away is the
    NEIGHBOURHOOD, which is exactly what a graph neural network can use and a per-account model cannot.

All data is synthetic, so no personal data is involved. Everything is reproducible from `seed`.
All laundering patterns are placed so that every timestamp stays inside the requested window.

Run:
    python -m aml_synth.graph_generator --accounts 5000 --to csv
    python -m aml_synth.graph_generator --accounts 50000 --to parquet,json,cypher
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("aml_synth")
__version__ = "0.2.0"

TYPOLOGIES = ["smurfing", "scatter_gather", "cyclic_loop", "shell_company", "cross_border_velocity"]
_TYPO_CODE = {t: i + 1 for i, t in enumerate(TYPOLOGIES)}          # 0 = none

NORMAL_COUNTRIES = ["US", "GB", "DE", "FR", "KE", "NG", "ZA", "IN", "BR", "JP", "CA", "AE"]
COUNTRY_WEIGHTS = np.array([30, 10, 8, 7, 8, 6, 5, 8, 4, 4, 6, 4], dtype=float)
COUNTRY_WEIGHTS /= COUNTRY_WEIGHTS.sum()
OFFSHORE = ["VG", "KY", "PA", "SC", "BZ"]                           # synthetic "secrecy jurisdiction" set
COUNTRIES = NORMAL_COUNTRIES + OFFSHORE
OFFSHORE_SHARE = 0.03                                               # share of ordinary accounts registered offshore
PAYMENT_FORMATS = ["wire", "ach", "card", "cash", "crypto_ramp"]
_F = {n: i for i, n in enumerate(PAYMENT_FORMATS)}
ACCOUNT_TYPES = ["individual", "business", "shell"]

START_TS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
HOUR, DAY = 3600, 86400
AMT_MU, AMT_SIGMA = 6.95, 1.05          # benign base amounts (then scaled by account size): overall median about $1,000


@dataclass
class GeneratorConfig:
    n_accounts: int = 5000
    n_background_tx: int | None = None          # default: 8 per account
    n_patterns_per_typology: int | None = None  # default: scales with the graph (about 6-7% suspicious accounts)
    smurf_deposits: tuple = (1000, 2000)        # small deposits into the central account (brief 1.1)
    hard_negative_ratio: float = 2.0            # benign look-alike structures per laundering pattern
    camouflage: float = 1.5                     # ordinary traffic carried by laundering accounts (1.0-2.0 keeps them looking normal)
    days: int = 30
    reporting_threshold: float = 10_000.0
    seed: int = 42

    def resolved(self) -> dict:
        n = self.n_accounts
        bg = self.n_background_tx if self.n_background_tx is not None else 8 * n
        if self.n_patterns_per_typology is None:
            others, smurf = max(2, round(n / 450)), max(1, round(n / 5000))
        else:
            others = int(self.n_patterns_per_typology)
            smurf = max(1, others // 10) if others > 0 else 0
        return {"background_tx": int(bg), "smurfing": smurf, "scatter_gather": others, "cyclic_loop": others,
                "shell_company": others, "cross_border_velocity": others}


@dataclass
class SyntheticGraph:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    config: dict = field(default_factory=dict)

    def summary(self) -> dict:
        tx, ac = self.transactions, self.accounts
        return {
            "accounts": int(len(ac)),
            "suspicious_accounts": int(ac["is_suspicious"].sum()),
            "suspicious_share_pct": round(100 * float(ac["is_suspicious"].mean()), 2),
            "transactions": int(len(tx)),
            "laundering_transactions": int(tx["is_laundering"].sum()),
            "laundering_rate_pct": round(100 * float(tx["is_laundering"].mean()), 3),
            "by_typology": {k: int(v) for k, v in tx.loc[tx["is_laundering"] == 1, "typology"].value_counts().items()},
            "amount_median_usd": round(float(tx["amount"].median()), 2),
            "time_range_days": round(float((tx["timestamp"].max() - tx["timestamp"].min()) / DAY), 2),
        }

    def to_networkx(self):
        """Directed multigraph: one node per account, one edge per transaction (parallel edges allowed)."""
        import networkx as nx
        g = nx.MultiDiGraph(name="xai-amlbench synthetic transaction graph")
        g.add_nodes_from((r["account_id"], {k: v for k, v in r.items() if k != "account_id"})
                         for r in self.accounts.to_dict("records"))
        tx = self.transactions
        cols = ["tx_id", "amount", "currency", "payment_format", "timestamp", "cross_border", "is_laundering", "typology", "pattern_id"]
        g.add_edges_from((s, d, dict(zip(cols, vals))) for s, d, *vals in
                         zip(tx["src"], tx["dst"], *[tx[c] for c in cols]))
        return g


class AMLGraphGenerator:
    def __init__(self, cfg: GeneratorConfig | None = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.W = self.cfg.days * DAY
        self.counts = self.cfg.resolved()
        # account columns; python lists so patterns can append new accounts
        self.a_type: list[int] = []
        self.a_cty: list[int] = []
        self.a_age: list[float] = []
        self.a_risk: list[float] = []
        self.a_size: list[float] = []
        self.a_susp: list[int] = []
        self.a_typo: list[int] = []
        self.rows: list[tuple] = []          # pattern / structure transactions
        self._pid = 0
        self._last_place = (0.0, 0.0)

    # ---------------------------------------------------------------- public
    def generate(self) -> SyntheticGraph:
        self._make_pool()
        bg = self._background()
        builders = {"smurfing": self._smurfing, "scatter_gather": self._scatter_gather, "cyclic_loop": self._cyclic_loop,
                    "shell_company": self._shell_company, "cross_border_velocity": self._cross_border_velocity}
        n_patterns = 0
        for typ, fn in builders.items():
            for _ in range(self.counts[typ]):
                fn()
                self._pid += 1
                n_patterns += 1
        for _ in range(int(round(self.cfg.hard_negative_ratio * n_patterns))):
            self._hard_negative()
            self._pid += 1
        return self._assemble(bg)

    # --------------------------------------------------------------- accounts
    def _make_pool(self) -> None:
        n, rng = self.cfg.n_accounts, self.rng
        probs = np.concatenate([COUNTRY_WEIGHTS * (1 - OFFSHORE_SHARE), np.full(len(OFFSHORE), OFFSHORE_SHARE / len(OFFSHORE))])
        cty = rng.choice(len(COUNTRIES), size=n, p=probs)
        kind = rng.choice(3, size=n, p=[0.77, 0.15, 0.08])                       # individual / business / holding company
        age = np.where(kind == 0, rng.uniform(60, 3650, n), np.where(kind == 1, rng.uniform(30, 3650, n), rng.uniform(500, 4000, n)))
        young_ind = (kind == 0) & (rng.random(n) < 0.12)                          # new customers
        age[young_ind] = rng.uniform(5, 365, int(young_ind.sum()))
        young = (kind == 1) & (rng.random(n) < 0.20)                              # legitimate start-ups
        age[young] = rng.uniform(10, 365, int(young.sum()))
        size = np.exp(rng.normal(0, 0.8, n)) * np.where(kind == 1, 4.0, np.where(kind == 2, 2.0, 1.0))
        self.pool_size = size
        self.a_size = size.tolist()
        short = rng.random(n) < 0.25                                              # short-lived accounts (one-off relationships)
        a0 = np.where(short, rng.uniform(0, 0.95 * self.W, n), 0.0)
        a1 = np.where(short, np.minimum(a0 + rng.uniform(1, 7, n) * DAY, self.W), float(self.W))
        self.pool_a0, self.pool_a1 = a0, a1
        self.pool_kind, self.pool_cty, self.pool_age = kind, cty, age
        self.a_type = kind.tolist()
        self.a_cty = cty.tolist()
        self.a_age = age.tolist()
        self.a_risk = rng.beta(2, 9, n).tolist()
        self.a_susp = [0] * n
        self.a_typo = [0] * n
        w = rng.pareto(1.6, n) + 1.0
        self.pool_w = w / w.sum()
        self.by_cty = {}
        for c in range(len(COUNTRIES)):
            ix = np.where(cty == c)[0]
            if len(ix):
                p = self.pool_w[ix]
                self.by_cty[c] = (ix, p / p.sum())
        self.biz = np.where(kind == 1)[0]
        self.ind = np.where(kind == 0)[0]
        self.young_biz = np.where(young)[0]

    def _new_account(self, kind: str, country: str | None = None, age: float | None = None, risky: bool = True) -> int:
        rng = self.rng
        if country is None:
            country = str(rng.choice(NORMAL_COUNTRIES, p=COUNTRY_WEIGHTS))
        if age is None:                              # a quarter brand-new; the rest look like any ordinary account
            age = float(rng.uniform(20, 400) if rng.random() < 0.25 else rng.choice(self.pool_age))
        self.a_size.append(float(np.exp(rng.normal(0, 0.8)) * {"individual": 1.0, "business": 4.0, "shell": 2.0}[kind]))
        self.a_type.append(ACCOUNT_TYPES.index(kind))
        self.a_cty.append(COUNTRIES.index(country))
        self.a_age.append(float(age))
        self.a_risk.append(float(rng.beta(3, 6) if (risky and rng.random() < 0.55) else rng.beta(2, 9)))
        self.a_susp.append(0)
        self.a_typo.append(0)
        return len(self.a_type) - 1

    def _flag(self, i: int, typo: str) -> None:
        self.a_susp[i] = 1
        if self.a_typo[i] == 0:
            self.a_typo[i] = _TYPO_CODE[typo]

    def _pool_pick(self, k: int, among: np.ndarray | None = None, home: int | None = None) -> np.ndarray:
        src = among if among is not None and len(among) else np.arange(self.cfg.n_accounts)
        if home is None:
            return self.rng.choice(src, size=k, replace=k > len(src))
        local = src[self.pool_cty[src] == home]
        return np.array([int(self.rng.choice(local if (len(local) and self.rng.random() < 0.80) else src)) for _ in range(k)])

    def _home(self) -> int:
        """Home country of a laundering ring: rings mostly operate inside one region."""
        return int(self.rng.choice(len(NORMAL_COUNTRIES), p=COUNTRY_WEIGHTS))

    def _acct(self, kind: str, home: int, **kw) -> int:
        """A new account, in the ring's home country 85% of the time."""
        return self._new_account(kind, country=NORMAL_COUNTRIES[home] if self.rng.random() < 0.80 else None, **kw)

    def _partner(self, country_idx: int) -> int:
        """An ordinary counterparty: 75% same country, weighted by activity."""
        rng = self.rng
        if rng.random() < 0.75 and country_idx in self.by_cty:
            ix, p = self.by_cty[country_idx]
            return int(rng.choice(ix, p=p))
        return int(rng.choice(self.cfg.n_accounts, p=self.pool_w))

    def _amount(self, size=None):
        return self.rng.lognormal(AMT_MU, AMT_SIGMA, size)

    def _pamt(self, a: int, b: int, shift: float, sigma: float = 0.9) -> float:
        """A laundering amount defined RELATIVE to the two accounts' normal scale (shift = log-multiple of a typical payment),
        so it is only modestly elevated for these accounts whatever the global median is."""
        return float(self.rng.lognormal(AMT_MU + shift, sigma) * (self.a_size[a] * self.a_size[b]) ** 0.5)

    # ------------------------------------------------------------ background
    def _background(self):
        n, rng, nb = self.cfg.n_accounts, self.rng, self.counts["background_tx"]
        src = rng.choice(n, size=nb, p=self.pool_w)
        dst = rng.choice(n, size=nb, p=self.pool_w)
        domestic = rng.random(nb) < 0.75
        for c, (ix, p) in self.by_cty.items():
            m = domestic & (self.pool_cty[src] == c)
            k = int(m.sum())
            if k:
                dst[m] = rng.choice(ix, size=k, p=p)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        self.bg_deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)   # ordinary activity level per account
        amt = np.clip(self._amount(len(src)) * np.sqrt(self.pool_size[src] * self.pool_size[dst]), 1.0, 250_000.0)
        big = rng.random(len(src)) < 0.01
        amt[big] = rng.uniform(15_000, 120_000, int(big.sum()))
        a0, a1 = self.pool_a0, self.pool_a1                          # a transaction happens while both accounts are active
        lo, hi = np.maximum(a0[src], a0[dst]), np.minimum(a1[src], a1[dst])
        ok = (hi - lo) > 3600
        u = rng.random(len(src))
        ts = START_TS + np.where(ok, lo + u * (hi - lo), a0[src] + u * (a1[src] - a0[src]))
        fmt = rng.integers(0, len(PAYMENT_FORMATS), len(src))
        return src, dst, np.round(amt, 2), ts, fmt

    # ------------------------------------------------------------- placement
    def _emit(self, rows: list, typo: int, laundering: bool, anchored: bool = False) -> None:
        """rows: (src, dst, amount, offset_seconds, fmt_index). Places them wholly inside the time window."""
        if not rows:
            return
        off = np.array([r[3] for r in rows], dtype=float)
        span = float(off.max())
        k = 1.0
        if span > 0.98 * self.W:
            k = 0.98 * self.W / span
            span *= k
        t0 = 0.0 if anchored else float(self.rng.uniform(0, max(self.W - span, 0.0)))
        if laundering:
            self._last_place = (t0, span)
        for (s, d, a, o, f) in rows:
            self.rows.append((s, d, round(float(a), 2), START_TS + t0 + o * k, f, int(laundering), typo, self._pid if laundering else -1))

    def _camouflage(self, accounts, lam: float) -> None:
        """Give laundering accounts ordinary traffic too, so their own statistics look normal."""
        rng = self.rng
        t0, span = self._last_place
        for a in accounts:
            # ordinary traffic: as much as a randomly chosen ordinary account carries (matches the real degree distribution)
            k = int(round(self.bg_deg[int(rng.integers(len(self.bg_deg)))] * (lam / 8.0) * self.cfg.camouflage / 2.0))
            short = rng.random() < 0.25                               # short-lived, like many ordinary accounts
            lo, hi = (t0, min(t0 + span + 3 * DAY, self.W)) if short else (0.0, float(self.W))
            for _ in range(k):
                other = self._partner(self.a_cty[a])
                if other == a:
                    continue
                s, d = (a, other) if rng.random() < 0.5 else (other, a)
                amt = self._amount() * (self.a_size[a] * self.a_size[other]) ** 0.5
                self.rows.append((s, d, round(float(np.clip(amt, 1, 250_000)), 2), START_TS + float(rng.uniform(lo, hi)),
                                  int(rng.integers(0, len(PAYMENT_FORMATS))), 0, 0, -1))

    # ------------------------------------------------------- laundering patterns
    def _smurfing(self) -> None:
        """Many small deposits (each under the threshold) from a crowd of mules into one central account,
        then rapid disbursement onwards. Half of the mules are ordinary recruited accounts."""
        rng, T = self.rng, self.cfg.reporting_threshold
        name = "smurfing"
        home = self._home()
        central = self._acct("business", home)
        self._flag(central, name)
        k = min(int(rng.integers(80, 201)), max(20, int(0.04 * self.cfg.n_accounts)))    # small graphs get fewer mules
        n_pool = int(k * 0.4)
        mules = [int(x) for x in self._pool_pick(n_pool, self.ind, home)]
        mules += [self._acct("individual", home) for _ in range(k - n_pool)]
        for m in mules:
            self._flag(m, name)
        deposits = int(rng.integers(self.cfg.smurf_deposits[0], self.cfg.smurf_deposits[1] + 1))
        deposits = max(50, min(deposits, int(0.3 * self.counts["background_tx"])))        # never dwarf a small graph
        burst = rng.uniform(2, 6) * DAY
        rows, total = [], 0.0
        mw = rng.pareto(1.1, len(mules)) + 1.0                   # a few very busy mules, many that only deposit once or twice
        mw /= mw.sum()
        mule_of = rng.choice(len(mules), size=deposits, p=mw)
        for j in range(deposits):
            amt = float(np.clip(self._pamt(mules[int(mule_of[j])], central, 0.1), 200, 0.99 * T))
            rows.append((mules[int(mule_of[j])], central, amt, rng.uniform(0, burst), _F["cash"] if rng.random() < .7 else _F["wire"]))
            total += amt
        benes = [self._new_account("business", country=str(rng.choice(OFFSHORE)) if rng.random() < .6 else None)
                 for _ in range(int(rng.integers(1, 4)))]
        for b in benes:
            self._flag(b, name)
            rows.append((central, b, total * 0.94 / len(benes), burst + rng.uniform(1 * HOUR, 24 * HOUR), _F["wire"]))
        self._emit(rows, _TYPO_CODE[name], True)
        self._camouflage(mules + [central], lam=8)

    def _scatter_gather(self) -> None:
        """origin scatters to k intermediaries (unequal shares), who gather into one destination."""
        rng, name = self.rng, "scatter_gather"
        home = self._home()
        origin = int(self._pool_pick(1, None, home)[0])
        dest = self._acct("business", home)
        k = int(rng.integers(6, 15))
        mids = [self._acct("business" if rng.random() < .5 else "individual", home) for _ in range(k)]
        for a in [origin, dest, *mids]:
            self._flag(a, name)
        rows = []
        for m in mids:
            share = float(np.clip(self._pamt(origin, m, 0.4), 300, 400_000))
            t1 = rng.uniform(0, 36 * HOUR)
            rows.append((origin, m, share, t1, _F["wire"]))
            rows.append((m, dest, share * rng.uniform(0.96, 0.99), t1 + rng.uniform(4 * HOUR, 72 * HOUR), _F["wire"]))
        self._emit(rows, _TYPO_CODE[name], True)
        self._camouflage(mids + [dest], lam=8)

    def _cyclic_loop(self) -> None:
        """funds travel around a closed loop several times, shrinking a little on every hop."""
        rng, name = self.rng, "cyclic_loop"
        home = self._home()
        length = int(rng.integers(3, 8))
        nodes = []
        for _ in range(length):
            kind = "business" if rng.random() < 0.5 else "individual"
            nodes.append(int(self._pool_pick(1, self.biz if kind == "business" else self.ind, home)[0]) if rng.random() < 0.5 else self._acct(kind, home))
        nodes = list(dict.fromkeys(nodes))
        while len(nodes) < 3:
            nodes.append(self._acct("individual", home))
        for a in nodes:
            self._flag(a, name)
        amt = float(np.clip(self._pamt(nodes[0], nodes[1], 0.5, 1.0), 300, 200_000))
        t, rows = 0.0, []
        for _ in range(int(rng.integers(2, 5))):
            for i, s in enumerate(nodes):
                rows.append((s, nodes[(i + 1) % len(nodes)], amt, t, _F["wire"]))
                amt *= float(rng.uniform(0.98, 0.995))
                t += float(rng.uniform(2 * HOUR, 18 * HOUR))
        self._emit(rows, _TYPO_CODE[name], True)
        self._camouflage(nodes, lam=8)

    def _shell_company(self) -> None:
        """origin -> chain of freshly opened shell accounts (often offshore, then dormant) -> beneficiary."""
        rng, name = self.rng, "shell_company"
        home = self._home()
        origin = int(self._pool_pick(1, None, home)[0])
        beneficiary = self._acct("business", home)
        chain = [self._new_account("shell", country=str(rng.choice(OFFSHORE)) if rng.random() < .65 else None, age=rng.uniform(10, 150) if rng.random() < .7 else rng.uniform(150, 900))
                 for _ in range(int(rng.integers(3, 7)))]
        for a in [origin, beneficiary, *chain]:
            self._flag(a, name)
        amt = float(np.clip(self._pamt(origin, beneficiary, 1.6, 0.8), 3_000, 1_500_000))
        t, rows = 0.0, []
        path = [origin, *chain, beneficiary]
        for s, d in zip(path[:-1], path[1:]):
            rows.append((s, d, amt, t, _F["wire"]))
            amt *= float(rng.uniform(0.985, 0.999))
            t += float(rng.uniform(3 * HOUR, 3 * DAY))
        self._emit(rows, _TYPO_CODE[name], True)
        self._camouflage(chain, lam=5)                  # shells are quiet but not silent
        self._camouflage([origin, beneficiary], lam=8)

    def _cross_border_velocity(self) -> None:
        """one account moves many mid-size transfers to/from several countries within hours."""
        rng, name = self.rng, "cross_border_velocity"
        home = self._home()
        subject = self._acct("individual", home)
        self._flag(subject, name)
        n_tx = int(rng.integers(10, 25))
        cty_pool = [c for c in range(len(COUNTRIES)) if c != home and c in self.by_cty]
        chosen = list(rng.choice(cty_pool, size=min(int(rng.integers(3, 6)), len(cty_pool)), replace=False))
        span = rng.uniform(6 * HOUR, 96 * HOUR)
        rows = []
        for i in range(n_tx):
            c = int(chosen[int(rng.integers(len(chosen)))])
            ix, p = self.by_cty[c]
            cp = int(rng.choice(ix, p=p))
            amt = float(np.clip(self._pamt(cp, subject, 0.3, 0.6), 100, 9500))
            o = rng.uniform(0, span)
            rows.append((cp, subject, amt, o, _F["wire"]) if i % 2 == 0 else (subject, cp, amt * rng.uniform(0.9, 0.99), o, _F["crypto_ramp"]))
        self._emit(rows, _TYPO_CODE[name], True)
        self._camouflage([subject], lam=8)

    # ---------------------------------------------- benign look-alikes (hard negatives)
    def _hard_negative(self) -> None:
        rng = self.rng
        fam = rng.choice(["merchant", "cash_business", "payroll", "rosca", "remitter", "trader", "startup", "supply_chain", "collector",
                          "escrow_chain", "cash_pooling", "subscription"],
                         p=[.09, .08, .08, .09, .07, .10, .04, .07, .09, .10, .09, .10])
        W = self.W
        rows = []
        if fam == "merchant":                                        # fan-in: many customers pay one business
            m = int(self._pool_pick(1, self.biz)[0])
            for c in self._pool_pick(int(rng.integers(60, 250))):
                if c != m:
                    rows.append((int(c), m, float(np.clip(rng.lognormal(np.log(150), 0.9), 5, 8000)), rng.uniform(0, W), _F["card"]))
        elif fam == "collector":                                     # marketplace / wholesaler: hundreds to thousands of payers
            m = int(self._pool_pick(1, self.biz)[0])
            mu = float(rng.uniform(np.log(60), np.log(1200)))
            burst = rng.random() < 0.5                                # flash sale / payday: most payments within 1-3 days
            b0, bl = rng.uniform(0, 0.8 * W), rng.uniform(1, 3) * DAY
            for c in self._pool_pick(int(rng.integers(300, 2001))):
                if c != m:
                    o = b0 + rng.uniform(0, bl) if (burst and rng.random() < 0.85) else rng.uniform(0, W)
                    rows.append((int(c), m, float(np.clip(rng.lognormal(mu, 0.9), 5, 9800)), min(o, W), _F["card"] if rng.random() < .6 else _F["wire"]))
        elif fam == "subscription":                                  # utility / school / gym: many payers, each paying repeatedly
            m = int(self._pool_pick(1, self.biz)[0])
            payers = self._pool_pick(int(rng.integers(60, 301)))
            base = float(rng.uniform(np.log(30), np.log(900)))
            for c in payers:
                amt_c = float(np.clip(rng.lognormal(base, 0.5), 5, 9800))
                start = rng.uniform(0, 0.4 * W)
                for r in range(int(rng.integers(2, 11))):
                    if c != m:
                        rows.append((int(c), m, amt_c * float(rng.uniform(0.97, 1.03)), min(start + r * rng.uniform(2, 6) * DAY, W), _F["ach"]))
        elif fam == "escrow_chain":                                  # legitimate pass-through of a large sum (closing, escrow)
            home = int(self._home())
            chain = [int(x) for x in self._pool_pick(int(rng.integers(3, 7)), self.biz, home)]
            amt, t = float(np.clip(rng.lognormal(np.log(60_000), 0.8), 8_000, 1_000_000)), rng.uniform(0, 0.5 * W)
            for a, b in zip(chain[:-1], chain[1:]):
                if a != b:
                    rows.append((a, b, amt, t, _F["wire"]))
                amt *= float(rng.uniform(0.98, 0.999))
                t += float(rng.uniform(3 * HOUR, 3 * DAY))
        elif fam == "cash_pooling":                                  # legitimate treasury ring between a group's own companies
            home = int(self._home())
            g = list(dict.fromkeys(int(x) for x in self._pool_pick(int(rng.integers(3, 6)), self.biz, home)))
            if len(g) >= 3:
                amt, t = float(np.clip(rng.lognormal(np.log(20_000), 0.8), 2_000, 300_000)), rng.uniform(0, 0.3 * W)
                for _ in range(int(rng.integers(2, 5))):
                    for i, a in enumerate(g):
                        rows.append((a, g[(i + 1) % len(g)], amt, t, _F["wire"]))
                        amt *= float(rng.uniform(0.995, 1.0))
                        t += float(rng.uniform(2 * HOUR, 30 * HOUR))
        elif fam == "cash_business":                                 # large deposits just under the threshold, legitimately
            m = int(self._pool_pick(1, self.biz)[0])
            for c in self._pool_pick(int(rng.integers(8, 31))):
                if c != m:
                    rows.append((int(c), m, float(rng.uniform(6500, 9900)), rng.uniform(0, W), _F["cash"]))
        elif fam == "payroll":                                       # fan-out: employer pays many employees, twice
            e = int(self._pool_pick(1, self.biz)[0])
            staff = self._pool_pick(int(rng.integers(10, 61)), self.ind)
            base = rng.lognormal(np.log(1800), 0.3)
            for cycle in (rng.uniform(0, 3 * DAY), rng.uniform(14, 17) * DAY):
                for s in staff:
                    rows.append((e, int(s), float(base * rng.uniform(0.6, 1.5)), cycle + rng.uniform(0, 6 * HOUR), _F["ach"]))
        elif fam == "rosca":                                         # savings group (chama): ring of equal contributions
            g = self._pool_pick(int(rng.integers(4, 11)), self.ind)
            g = list(dict.fromkeys(int(x) for x in g))
            if len(g) >= 3:
                amt, gap = float(rng.uniform(100, 2500)), float(rng.uniform(5, 9) * DAY)
                for r in range(int(rng.integers(3, 7))):
                    for i, a in enumerate(g):
                        rows.append((a, g[(i + 1) % len(g)], amt * rng.uniform(0.97, 1.03), r * gap + rng.uniform(0, 8 * HOUR), _F["ach"]))
        elif fam == "remitter":                                      # regular transfers to family abroad
            s = int(self._pool_pick(1, self.ind)[0])
            for c in [c for c in rng.choice(len(COUNTRIES) - len(OFFSHORE), size=2, replace=False) if c != self.a_cty[s]][:2]:
                if c in self.by_cty:
                    ix, p = self.by_cty[int(c)]
                    dst = int(rng.choice(ix, p=p))
                    for _ in range(int(rng.integers(2, 6))):
                        rows.append((s, dst, float(np.clip(rng.lognormal(np.log(350), 0.5), 40, 3000)), rng.uniform(0, W), _F["wire"]))
        elif fam == "trader":                                        # e-commerce / freight burst across countries
            t = int(self._pool_pick(1, self.biz)[0])
            span = rng.uniform(6 * HOUR, 96 * HOUR)
            for i in range(int(rng.integers(8, 31))):
                c = int(rng.integers(0, len(COUNTRIES) - len(OFFSHORE)))
                if c in self.by_cty:
                    ix, p = self.by_cty[c]
                    cp = int(rng.choice(ix, p=p))
                    amt = float(rng.uniform(300, 5000))
                    o = rng.uniform(0, span)
                    rows.append((cp, t, amt, o, _F["wire"]) if i % 2 == 0 else (t, cp, amt, o, _F["wire"]))
        elif fam == "startup":                                       # young business ramping up
            pool = self.young_biz if len(self.young_biz) else self.biz
            m = int(self._pool_pick(1, pool)[0])
            for c in self._pool_pick(int(rng.integers(20, 81))):
                if c != m:
                    rows.append((int(c), m, float(np.clip(rng.lognormal(np.log(300), 0.9), 10, 9000)), rng.uniform(0, W), _F["card"]))
            for v in self._pool_pick(int(rng.integers(3, 10)), self.biz):
                if v != m:
                    rows.append((m, int(v), float(rng.uniform(500, 6000)), rng.uniform(0, W), _F["wire"]))
        else:                                                        # supply chain: buyer -> suppliers -> shared logistics firm
            buyer = int(self._pool_pick(1, self.biz)[0])
            sup = [int(x) for x in self._pool_pick(int(rng.integers(5, 13)), self.biz)]
            logi = int(self._pool_pick(1, self.biz)[0])
            for s in sup:
                if s in (buyer, logi):
                    continue
                t1 = rng.uniform(0, 0.6 * W)
                a = float(np.clip(rng.lognormal(np.log(8000), 0.8), 500, 120_000))
                rows.append((buyer, s, a, t1, _F["wire"]))
                rows.append((s, logi, a * rng.uniform(0.25, 0.6), t1 + rng.uniform(1, 12) * DAY, _F["wire"]))
        self._emit(rows, 0, False, anchored=True)

    # -------------------------------------------------------------- assemble
    def _assemble(self, bg) -> SyntheticGraph:
        bs, bd, ba, bt, bf = bg
        if self.rows:
            ps, pd_, pa, pt, pf, pl, pty, ppid = (np.array(c) for c in zip(*self.rows))
        else:
            ps = pd_ = pf = pl = pty = ppid = np.array([], dtype=np.int64)
            pa = pt = np.array([], dtype=float)
        src = np.concatenate([bs, ps]).astype(np.int64)
        dst = np.concatenate([bd, pd_]).astype(np.int64)
        amt = np.concatenate([ba, pa]).astype(float)
        ts = np.concatenate([bt, pt]).astype(float)
        fmt = np.concatenate([bf, pf]).astype(np.int64)
        lab = np.concatenate([np.zeros(len(bs), dtype=np.int64), pl]).astype(np.int64)
        typ = np.concatenate([np.zeros(len(bs), dtype=np.int64), pty]).astype(np.int64)
        pid = np.concatenate([np.full(len(bs), -1, dtype=np.int64), ppid]).astype(np.int64)

        end = START_TS + self.W
        ts = np.clip(ts, START_TS, end).astype(np.int64)            # patterns are placed inside; this is a safety net
        order = np.argsort(ts, kind="stable")
        src, dst, amt, ts, fmt, lab, typ, pid = (a[order] for a in (src, dst, amt, ts, fmt, lab, typ, pid))

        n_acc = len(self.a_type)
        ids = np.array([f"ACC{i:07d}" for i in range(n_acc)])
        cty = np.array(self.a_cty)
        typo_names = np.array(["none"] + TYPOLOGIES)
        accounts = pd.DataFrame({
            "account_id": ids,
            "account_type": np.array(ACCOUNT_TYPES)[np.array(self.a_type)],
            "country": np.array(COUNTRIES)[cty],
            "opened_ts": (START_TS - np.array(self.a_age) * DAY).astype(np.int64),
            "risk_score": np.round(np.array(self.a_risk), 4),
            "is_suspicious": np.array(self.a_susp, dtype=np.int64),
            "typology": typo_names[np.array(self.a_typo)],
        })
        tx = pd.DataFrame({
            "tx_id": [f"TX{i:09d}" for i in range(len(src))],
            "src": ids[src], "dst": ids[dst], "amount": np.round(amt, 2), "currency": "USD",
            "payment_format": np.array(PAYMENT_FORMATS)[fmt], "timestamp": ts,
            "cross_border": (cty[src] != cty[dst]).astype(np.int64),
            "is_laundering": lab, "typology": typo_names[typ], "pattern_id": pid,
        })
        cfg = asdict(self.cfg)
        cfg.update({"resolved": self.counts, "generator_version": __version__, "start_ts": START_TS})
        return SyntheticGraph(accounts=accounts, transactions=tx, config=cfg)


# -------------------------------------------------------------------- CLI
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Generate a synthetic AML transaction graph")
    p.add_argument("--accounts", type=int, default=int(os.getenv("SYNTH_ACCOUNTS", 5000)))
    p.add_argument("--background-tx", type=int, default=int(os.getenv("SYNTH_BACKGROUND_TX", 0)) or None,
                   help="benign transactions (default: 8 per account)")
    p.add_argument("--patterns", type=int, default=int(os.getenv("SYNTH_PATTERNS", 0)) or None,
                   help="laundering patterns per typology (default: scales with the graph size)")
    p.add_argument("--hard-negatives", type=float, default=2.0, help="benign look-alike structures per laundering pattern")
    p.add_argument("--camouflage", type=float, default=1.5, help="how much ordinary traffic laundering accounts carry (1.0-2.0 keeps them looking normal)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.getenv("SYNTH_OUT_DIR", "data"))
    p.add_argument("--to", default=os.getenv("EXPORT_TARGETS", "csv"), help="comma list of: csv,parquet,json,cypher,kafka,neo4j")
    a = p.parse_args()

    cfg = GeneratorConfig(n_accounts=a.accounts, n_background_tx=a.background_tx, n_patterns_per_typology=a.patterns,
                          hard_negative_ratio=a.hard_negatives, camouflage=a.camouflage, days=a.days, seed=a.seed)
    graph = AMLGraphGenerator(cfg).generate()
    log.info("generated: %s", graph.summary())

    from aml_synth import exporters  # lazy: csv-only runs need no kafka / neo4j / pyarrow

    targets = {t.strip() for t in a.to.split(",") if t.strip()}
    for name in ("csv", "parquet", "json", "cypher"):
        if name in targets:
            getattr(exporters, f"export_{name}")(graph, a.out)
    if "neo4j" in targets:
        exporters.export_neo4j(graph)
    if "kafka" in targets:
        exporters.export_kafka(graph)


if __name__ == "__main__":
    main()
