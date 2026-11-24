"""gnn_aml_core.anomaly_reference: a NORMAL-ONLY reference for Model 2 (the anomaly detector).

Why this exists. Model 2 used to learn "normal" from the live accounts Model 1 scored as low-risk. Those accounts
are exactly where a missed mule would hide, so a missed mule could be taught to the network as normal. This
module removes live accounts from the training loop entirely: Model 2 is trained ONCE on a large synthetic
population that is normal by construction (no laundering pattern is ever generated), and then FROZEN. Live
accounts are only ever scored against it. A live account is flagged when it does not look like anything in the
catalogue of normal behaviour below.

What "all normal patterns" means here, honestly: the catalogue below lists every kind of legitimate behaviour
we could name (including the awkward ones that merely LOOK suspicious -- cash-heavy markets, payroll bursts,
chama/rosca rotations, remittances, import/export, escrow pass-through, marketplaces, round-trip refunds ...),
and on top of that a "profile" generator that draws every behavioural knob independently over wide ranges, so
combinations nobody thought to name are covered as well. A legitimate pattern that is in none of those will
look unusual and be flagged; when an analyst clears it, add it here (a new archetype function) and rebuild.

    python -m gnn_aml_core.anomaly_reference --build --accounts 20000 --epochs 150 --models-dir models
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Self-contained on purpose: the gnn_aml_core image does not ship aml_synth, so these mirror its constants.
NORMAL_COUNTRIES = ["US", "GB", "DE", "FR", "KE", "NG", "ZA", "IN", "BR", "JP", "CA", "AE"]
COUNTRY_WEIGHTS = np.array([30, 10, 8, 7, 8, 6, 5, 8, 4, 4, 6, 4], dtype=float)
COUNTRY_WEIGHTS /= COUNTRY_WEIGHTS.sum()
OFFSHORE = ["VG", "KY", "PA", "SC", "BZ"]
COUNTRIES = NORMAL_COUNTRIES + OFFSHORE
PAYMENT_FORMATS = ["wire", "ach", "card", "cash", "crypto_ramp"]
START_TS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())

log = logging.getLogger("anomaly_reference")

DAY, HOUR = 86400, 3600
_F = {n: i for i, n in enumerate(PAYMENT_FORMATS)}
GENERIC_MEMOS = ["PAYMENT", "INVOICE PMT", "TRANSFER", "RENT", "SALARY", ""]
# legitimate memos that MANY unrelated senders share (school fees, chama contributions, tax ...) -- the shared-memo
# feature must not read these as coordination
SHARED_LEGIT_MEMOS = ["SCHOOL FEES T1", "SCHOOL FEES T3", "CHAMA CONTRIBUTION", "ELECTRICITY", "WATER BILL",
                      "RENT NOV", "TAX PAYMENT", "SACCO SHARES", "INSURANCE PREMIUM", "DONATION", "ORDER REF"]


class _Pop:
    """Accumulates accounts and transactions. Every account is normal; nothing here ever emits a laundering row."""

    def __init__(self, days: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.W = days * DAY
        self.kind: list[int] = []
        self.cty: list[int] = []
        self.age: list[float] = []
        self.risk: list[float] = []
        self.highrisk: list[bool] = []
        self.rows: list[tuple] = []     # src, dst, amount, ts, fmt, memo, disputed

    # ---- accounts
    def acct(self, kind: int, country: int | None = None, age: float | None = None, highrisk: bool = False) -> int:
        r = self.rng
        if country is None:
            if r.random() < 0.03:
                country = len(NORMAL_COUNTRIES) + int(r.integers(0, len(OFFSHORE)))   # legit offshore-registered
            else:
                country = int(r.integers(0, len(NORMAL_COUNTRIES)))
        if age is None:
            age = float(r.uniform(60, 3650) if kind == 0 else r.uniform(30, 3650) if kind == 1 else r.uniform(500, 4000))
            if r.random() < 0.12:
                age = float(r.uniform(3, 365))                       # new customers are normal
        self.kind.append(kind)
        self.cty.append(int(country))
        self.age.append(age)
        self.risk.append(float(r.beta(2, 9)))
        self.highrisk.append(bool(highrisk))
        return len(self.kind) - 1

    # ---- timing / amounts
    def ts(self, n: int, hours: str = "business", lo: float = 0.0, hi: float | None = None) -> np.ndarray:
        r, hi = self.rng, (self.W if hi is None else hi)
        day = np.floor(r.uniform(lo, hi, n) / DAY)
        if hours == "business":
            h = np.clip(r.normal(13, 3.2, n), 6, 21)
        elif hours == "morning":
            h = np.clip(r.normal(9, 1.5, n), 5, 13)
        elif hours == "evening":
            h = np.clip(r.normal(19, 2.0, n), 15, 23.5)
        elif hours == "night":
            h = (r.normal(1.5, 2.5, n)) % 24
        else:                                                         # "any": automated / always-on
            h = r.uniform(0, 24, n)
        return np.clip(day * DAY + h * HOUR + r.uniform(0, HOUR, n), 0, self.W - 1)

    def amt(self, n: int, median: float, sigma: float = 0.8, lo: float = 1.0, hi: float = 250_000.0,
            round_share: float = 0.0, decimals: bool = False) -> np.ndarray:
        a = np.clip(self.rng.lognormal(np.log(max(median, 1.0)), sigma, n), lo, hi)
        if round_share > 0:
            m = self.rng.random(n) < round_share
            step = np.where(a[m] < 100, 10.0, np.where(a[m] < 1000, 50.0, 100.0))   # round to a human-looking figure
            a[m] = np.maximum(np.round(a[m] / step), 1.0) * step
            a = np.clip(a, lo, hi)
        return np.round(a, 2 if decimals or round_share == 0 else 0)

    def fmt(self, n: int, mix: dict[str, float]) -> np.ndarray:
        names = list(mix)
        p = np.array([mix[k] for k in names], dtype=float)
        pick = self.rng.choice(len(names), size=n, p=p / p.sum())
        return np.array([_F[names[i]] for i in pick])

    def memo(self, n: int, shared: str | None = None, p_shared: float = 0.0) -> list[str]:
        r = self.rng
        out = []
        for _ in range(n):
            if shared is not None and r.random() < p_shared:
                out.append(shared)
            else:
                out.append(GENERIC_MEMOS[int(r.integers(0, len(GENERIC_MEMOS)))])
        return out

    def add(self, src, dst, amount, ts, fmt, memo=None, disputed_rate: float = 0.015) -> None:
        n = len(amount)
        if n == 0:
            return
        src = np.broadcast_to(np.asarray(src), (n,))
        dst = np.broadcast_to(np.asarray(dst), (n,))
        fmt = np.broadcast_to(np.asarray(fmt), (n,))
        memo = memo if memo is not None else self.memo(n)
        dis = self.rng.random(n) < disputed_rate
        keep = src != dst
        for i in np.where(keep)[0]:
            self.rows.append((int(src[i]), int(dst[i]), float(amount[i]), float(ts[i]), int(fmt[i]), memo[i], bool(dis[i])))

    def people(self, n: int, country: int | None = None) -> list[int]:
        return [self.acct(0, country) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------
# The catalogue. Each archetype builds one small, self-contained group of accounts behaving normally.
# ---------------------------------------------------------------------------------------------------------
def _salaried(p: _Pop) -> None:
    r = p.rng
    emp, c = p.acct(1), int(r.integers(0, len(NORMAL_COUNTRIES)))
    for _ in range(int(r.integers(3, 9))):
        w = p.acct(0, c)
        pay = float(r.lognormal(np.log(1200), 0.6)) if r.random() < 0.9 else float(r.uniform(6000, 9900))   # incl. near-threshold pay
        k = max(1, p.W // (30 * DAY))
        p.add(emp, w, [pay] * k, p.ts(k, "morning"), p.fmt(k, {"ach": 1}), ["SALARY"] * k)
        landlord, shop, util = p.acct(0, c), p.acct(1, c), p.acct(1, c)
        p.add(w, landlord, [pay * r.uniform(0.2, 0.4)] * k, p.ts(k, "business"), p.fmt(k, {"ach": 3, "wire": 1}), ["RENT"] * k)
        n = int(r.integers(4, 30))
        p.add(w, shop, p.amt(n, 40, 0.9), p.ts(n, "evening"), p.fmt(n, {"card": 8, "cash": 1}))
        p.add(w, util, p.amt(k, 60, 0.4), p.ts(k, "any"), p.fmt(k, {"ach": 1}), ["ELECTRICITY"] * k)


def _merchant_fan_in(p: _Pop) -> None:
    r = p.rng
    m = p.acct(1)
    n = int(r.integers(40, 400))
    cust = p.people(min(n, 120), p.cty[m] if r.random() < 0.85 else None)
    src = [cust[int(r.integers(0, len(cust)))] for _ in range(n)]
    p.add(src, m, p.amt(n, float(r.uniform(15, 400)), 0.9, hi=9000), p.ts(n, "business"), p.fmt(n, {"card": 7, "cash": 2, "ach": 1}),
          p.memo(n, "ORDER REF", 0.2))


def _cash_business(p: _Pop) -> None:      # market vendor, matatu sacco, mobile-money agent: cash, round amounts, high volume
    r = p.rng
    b = p.acct(1)
    n = int(r.integers(60, 500))
    cust = p.people(int(r.integers(20, 150)), p.cty[b])
    src = [cust[int(r.integers(0, len(cust)))] for _ in range(n)]
    p.add(src, b, p.amt(n, float(r.uniform(5, 120)), 0.8, round_share=float(r.uniform(0.3, 0.9))), p.ts(n, "business"),
          p.fmt(n, {"cash": 9, "card": 1}))
    sup = p.acct(1, p.cty[b])
    k = int(r.integers(2, 10))
    p.add(b, sup, p.amt(k, float(r.uniform(500, 6000)), 0.6, hi=9900), p.ts(k, "morning"), p.fmt(k, {"wire": 3, "ach": 2}))


def _payroll(p: _Pop) -> None:            # many outflows in a burst at month end -- looks like fan-out, is normal
    r = p.rng
    emp = p.acct(1)
    staff = p.people(int(r.integers(15, 300)), p.cty[emp])
    base = float(r.uniform(400, 3500))
    days = max(1, p.W // (30 * DAY))
    for d in range(days):
        t = p.ts(len(staff), "morning", lo=min(p.W - 3 * DAY, 26 * DAY + d * 30 * DAY) if p.W > 27 * DAY else 0, hi=p.W)
        p.add(emp, staff, p.amt(len(staff), base, 0.4, round_share=float(r.choice([0.0, 0.8]))), t, p.fmt(len(staff), {"ach": 1}),
              ["SALARY"] * len(staff))
    for s in staff[: max(1, len(staff) // 4)]:
        p.add(s, p.acct(1, p.cty[emp]), p.amt(3, base * 0.3, 0.5), p.ts(3, "evening"), p.fmt(3, {"card": 1}))


def _remitter(p: _Pop) -> None:           # diaspora worker sends monthly money home -- cross-border but ordinary
    r = p.rng
    abroad = int(r.integers(0, len(NORMAL_COUNTRIES)))
    home = int(r.integers(0, len(NORMAL_COUNTRIES)))
    sender = p.acct(0, abroad)
    fam = [p.acct(0, home) for _ in range(int(r.integers(1, 5)))]
    k = max(1, p.W // (30 * DAY)) + int(r.integers(0, 3))
    for f in fam:
        p.add(sender, f, p.amt(k, float(r.uniform(80, 900)), 0.3, round_share=float(r.uniform(0, 0.8))), p.ts(k, "evening"),
              p.fmt(k, {"wire": 3, "card": 1, "crypto_ramp": 0.2}), ["TRANSFER"] * k)


def _trader(p: _Pop) -> None:             # import/export: large cross-border wires to a few counterparties
    r = p.rng
    t = p.acct(1)
    sup = [p.acct(1, int(r.integers(0, len(NORMAL_COUNTRIES)))) for _ in range(int(r.integers(2, 7)))]
    cust = [p.acct(1, p.cty[t]) for _ in range(int(r.integers(2, 12)))]
    n = int(r.integers(4, 25))
    p.add([sup[int(r.integers(0, len(sup)))] for _ in range(n)], t, p.amt(n, float(r.uniform(3000, 40000)), 0.7, hi=150_000),
          p.ts(n, "business"), p.fmt(n, {"wire": 1}), p.memo(n, "INVOICE PMT", 0.7))
    m = int(r.integers(4, 25))
    p.add(t, [cust[int(r.integers(0, len(cust)))] for _ in range(m)], p.amt(m, float(r.uniform(3000, 40000)), 0.7, hi=150_000),
          p.ts(m, "business"), p.fmt(m, {"wire": 3, "ach": 1}))


def _startup(p: _Pop) -> None:            # young business with ramping volume
    r = p.rng
    b = p.acct(1, age=float(r.uniform(10, 120)))
    n = int(r.integers(10, 80))
    cust = p.people(int(r.integers(5, 40)))
    p.add([cust[int(r.integers(0, len(cust)))] for _ in range(n)], b, p.amt(n, float(r.uniform(30, 600)), 1.0),
          p.ts(n, "business", lo=0.3 * p.W), p.fmt(n, {"card": 4, "ach": 2, "wire": 1}))
    f = p.acct(0)
    p.add(f, b, p.amt(1, float(r.uniform(5000, 60000)), 0.4), p.ts(1, "business", hi=0.3 * p.W), p.fmt(1, {"wire": 1}))


def _supply_chain(p: _Pop) -> None:       # chain of businesses each passing goods money on with a margin
    r = p.rng
    chain = [p.acct(1) for _ in range(int(r.integers(3, 6)))]
    amount = float(r.uniform(2000, 30000))
    for i in range(len(chain) - 1):
        n = int(r.integers(3, 12))
        a = p.amt(n, amount * (1 + 0.12 * i), 0.3, hi=120_000)
        p.add(chain[i + 1], chain[i], a, p.ts(n, "business"), p.fmt(n, {"wire": 2, "ach": 3}), p.memo(n, "INVOICE PMT", 0.6))


def _subscription(p: _Pop) -> None:       # utility, school, gym: many payers, each paying repeatedly
    r = p.rng
    m = p.acct(1)
    payers = p.people(int(r.integers(40, 250)), p.cty[m])
    base = float(r.uniform(15, 600))
    memo = SHARED_LEGIT_MEMOS[int(r.integers(0, len(SHARED_LEGIT_MEMOS)))]
    for q in payers:
        k = int(r.integers(1, 6))
        p.add(q, m, [base * float(r.uniform(0.97, 1.03))] * k, p.ts(k, "any"), p.fmt(k, {"ach": 3, "card": 2}), [memo] * k)


def _chama(p: _Pop) -> None:              # rosca/chama/sacco: members contribute equally, one receives in rotation (a legit cycle)
    r = p.rng
    members = p.people(int(r.integers(5, 25)))
    amount = float(r.uniform(20, 800))
    rounds = int(r.integers(2, 6))
    for k in range(rounds):
        taker = members[k % len(members)]
        t0 = k * p.W / rounds
        for mbr in members:
            if mbr != taker:
                p.add(mbr, taker, [amount], p.ts(1, "evening", lo=t0, hi=t0 + 5 * DAY), p.fmt(1, {"ach": 2, "card": 1}), ["CHAMA CONTRIBUTION"])


def _escrow(p: _Pop) -> None:             # property closing / escrow: one big sum in and straight out
    r = p.rng
    buyer, esc, seller = p.acct(0), p.acct(1), p.acct(0)
    s = float(r.uniform(20_000, 120_000))
    t = float(r.uniform(0, 0.8 * p.W))
    p.add(buyer, esc, [s], [t], p.fmt(1, {"wire": 1}), ["PAYMENT"])
    p.add(esc, seller, [s * 0.985], [t + float(r.uniform(0.5, 3)) * DAY], p.fmt(1, {"wire": 1}), ["PAYMENT"])


def _marketplace(p: _Pop) -> None:        # platform collects from many buyers and pays many sellers
    r = p.rng
    plat = p.acct(1)
    sellers = [p.acct(1) for _ in range(int(r.integers(8, 60)))]
    buyers = p.people(int(r.integers(60, 300)))
    n = len(buyers) * 2
    p.add([buyers[int(r.integers(0, len(buyers)))] for _ in range(n)], plat, p.amt(n, float(r.uniform(20, 300)), 0.9, hi=5000),
          p.ts(n, "any"), p.fmt(n, {"card": 5, "ach": 1}))
    for s in sellers:
        k = int(r.integers(1, 5))
        p.add(plat, s, p.amt(k, float(r.uniform(300, 5000)), 0.6, hi=20000), p.ts(k, "morning"), p.fmt(k, {"ach": 1}))


def _freelancer(p: _Pop) -> None:
    r = p.rng
    f = p.acct(0)
    clients = [p.acct(1, int(r.integers(0, len(NORMAL_COUNTRIES)))) for _ in range(int(r.integers(1, 6)))]
    n = int(r.integers(1, 8))
    p.add([clients[int(r.integers(0, len(clients)))] for _ in range(n)], f, p.amt(n, float(r.uniform(300, 6000)), 0.8, hi=30000),
          p.ts(n, "business"), p.fmt(n, {"wire": 3, "crypto_ramp": 1, "ach": 1}), p.memo(n, "INVOICE PMT", 0.5))
    m = int(r.integers(3, 20))
    p.add(f, p.acct(1), p.amt(m, 45, 1.0), p.ts(m, "evening"), p.fmt(m, {"card": 1}))


def _student(p: _Pop) -> None:
    r = p.rng
    s, parent, school = p.acct(0, age=float(r.uniform(60, 900))), p.acct(0), p.acct(1)
    k = int(r.integers(1, 4))
    p.add(parent, s, p.amt(k, float(r.uniform(50, 400)), 0.4, round_share=0.7), p.ts(k, "evening"), p.fmt(k, {"ach": 1, "card": 1}))
    n = int(r.integers(5, 40))
    p.add(s, p.acct(1), p.amt(n, 8, 0.9, hi=200), p.ts(n, "evening"), p.fmt(n, {"card": 6, "cash": 1}))
    p.add(parent, school, p.amt(1, float(r.uniform(300, 3500)), 0.3), p.ts(1, "business"), p.fmt(1, {"ach": 1, "wire": 1}),
          [SHARED_LEGIT_MEMOS[int(r.integers(0, 2))]])


def _high_net_worth(p: _Pop) -> None:     # rare large legitimate transactions
    r = p.rng
    h = p.acct(0, age=float(r.uniform(900, 4000)))
    n = int(r.integers(1, 6))
    p.add(h, [p.acct(int(r.choice([0, 1, 2]))) for _ in range(n)], p.amt(n, float(r.uniform(15000, 80000)), 0.5, hi=250_000),
          p.ts(n, "business"), p.fmt(n, {"wire": 1}))
    m = int(r.integers(5, 30))
    p.add(h, p.acct(1), p.amt(m, 400, 1.0), p.ts(m, "evening"), p.fmt(m, {"card": 1}))


def _dormant_seasonal(p: _Pop) -> None:   # silent for most of the window, then a legitimate burst (harvest, school term)
    r = p.rng
    a = p.acct(int(r.choice([0, 1])), age=float(r.uniform(400, 3000)))
    start = float(r.uniform(0.6, 0.9)) * p.W
    n = int(r.integers(6, 40))
    cp = [p.acct(int(r.choice([0, 1]))) for _ in range(int(r.integers(2, 10)))]
    p.add([cp[int(r.integers(0, len(cp)))] for _ in range(n)], a, p.amt(n, float(r.uniform(100, 2500)), 0.8, hi=20000),
          p.ts(n, "business", lo=start), p.fmt(n, {"ach": 2, "wire": 1, "cash": 1}))


def _refund_round_trip(p: _Pop) -> None:  # pay a merchant, get (most of) it back
    r = p.rng
    m = p.acct(1)
    for c in p.people(int(r.integers(5, 40))):
        a = float(r.uniform(20, 900))
        t = float(r.uniform(0, 0.85 * p.W))
        p.add(c, m, [a], [t], p.fmt(1, {"card": 1}))
        if r.random() < 0.5:
            p.add(m, c, [a * float(r.uniform(0.9, 1.0))], [t + float(r.uniform(0.5, 6)) * DAY], p.fmt(1, {"card": 1}), disputed_rate=0.3)


def _gig_worker(p: _Pop) -> None:         # many small daily inflows from a platform, cash-out to own accounts
    r = p.rng
    w, plat, bank2 = p.acct(0), p.acct(1), p.acct(0)
    n = int(r.integers(20, 120))
    p.add(plat, w, p.amt(n, float(r.uniform(8, 60)), 0.7, hi=400), p.ts(n, "any"), p.fmt(n, {"ach": 1}))
    k = int(r.integers(2, 10))
    p.add(w, bank2, p.amt(k, float(r.uniform(100, 600)), 0.5), p.ts(k, "evening"), p.fmt(k, {"ach": 1, "cash": 1}))


def _gambling_or_crypto_user(p: _Pop) -> None:   # ordinary people sometimes use high-risk merchants
    r = p.rng
    house = p.acct(1, highrisk=True)
    for u in p.people(int(r.integers(3, 30))):
        n = int(r.integers(1, 12))
        p.add(u, house, p.amt(n, float(r.uniform(10, 250)), 0.9, hi=3000), p.ts(n, "night" if r.random() < 0.4 else "evening"),
              p.fmt(n, {"card": 4, "crypto_ramp": 2}))


def _charity(p: _Pop) -> None:
    r = p.rng
    ngo = p.acct(1)
    donors = p.people(int(r.integers(30, 220)))
    n = len(donors)
    p.add(donors, ngo, p.amt(n, float(r.uniform(10, 150)), 0.9, round_share=0.5), p.ts(n, "any"), p.fmt(n, {"card": 3, "ach": 1}),
          ["DONATION"] * n)
    k = int(r.integers(2, 12))
    p.add(ngo, [p.acct(1, int(r.integers(0, len(NORMAL_COUNTRIES)))) for _ in range(k)], p.amt(k, float(r.uniform(1000, 15000)), 0.6),
          p.ts(k, "business"), p.fmt(k, {"wire": 1}))


def _cash_pooling(p: _Pop) -> None:       # group treasury: sweeps between subsidiaries, including cycles
    r = p.rng
    holding = p.acct(2)
    subs = [p.acct(1) for _ in range(int(r.integers(3, 8)))]
    for s in subs:
        n = int(r.integers(4, 16))
        a = p.amt(n, float(r.uniform(5000, 50000)), 0.5, hi=200_000)
        p.add(s, holding, a, p.ts(n, "business"), p.fmt(n, {"wire": 1, "ach": 1}), ["TRANSFER"] * n)
        p.add(holding, s, a * float(r.uniform(0.6, 1.0)), p.ts(n, "business"), p.fmt(n, {"wire": 1, "ach": 1}), ["TRANSFER"] * n)


def _friends_split(p: _Pop) -> None:      # small circular P2P among friends
    r = p.rng
    g = p.people(int(r.integers(3, 9)))
    for _ in range(int(r.integers(6, 40))):
        a, b = r.choice(len(g), 2, replace=False)
        p.add([g[a]], [g[b]], p.amt(1, float(r.uniform(5, 80)), 0.7, decimals=True), p.ts(1, "evening"), p.fmt(1, {"ach": 2, "card": 1}))


def _taxpayer(p: _Pop) -> None:
    r = p.rng
    gov = p.acct(2, age=3500)
    for b in [p.acct(1) for _ in range(int(r.integers(5, 40)))]:
        k = int(r.integers(1, 4))
        p.add(b, gov, p.amt(k, float(r.uniform(300, 12000)), 0.8, hi=60000), p.ts(k, "business"), p.fmt(k, {"wire": 1, "ach": 2}), ["TAX PAYMENT"] * k)


def _night_shift_business(p: _Pop) -> None:     # 24h/night businesses: bars, clubs, 24h pharmacies, taxis
    r = p.rng
    b = p.acct(1)
    n = int(r.integers(40, 300))
    cust = p.people(int(r.integers(20, 100)), p.cty[b])
    p.add([cust[int(r.integers(0, len(cust)))] for _ in range(n)], b, p.amt(n, float(r.uniform(5, 90)), 0.8),
          p.ts(n, "night"), p.fmt(n, {"cash": 4, "card": 5}))


def _saver(p: _Pop) -> None:              # nearly inactive: a handful of transactions or none
    r = p.rng
    a = p.acct(0, age=float(r.uniform(200, 3600)))
    for _ in range(int(r.integers(0, 3))):
        p.add(a, p.acct(0), p.amt(1, 200, 0.8), p.ts(1, "business"), p.fmt(1, {"ach": 1}))


def _background_pool(p: _Pop) -> None:
    """Ordinary traffic at scale: a heavy-tailed population (a few very busy accounts, a long tail of quiet ones)
    paying random partners, mostly in their own country. This is what most normal accounts really look like, so it
    must be a large share of the reference."""
    r = p.rng
    m = int(r.integers(800, 1600))
    kinds = r.choice(3, size=m, p=[0.77, 0.15, 0.08])
    ctys = r.choice(len(NORMAL_COUNTRIES), size=m, p=COUNTRY_WEIGHTS)
    ids = [p.acct(int(k), int(c)) for k, c in zip(kinds, ctys)]
    ids_a = np.array(ids)
    size = np.exp(r.normal(0, 0.8, m)) * np.where(kinds == 1, 4.0, np.where(kinds == 2, 2.0, 1.0))
    w = r.pareto(1.6, m) + 1.0
    w /= w.sum()
    nb = int(r.uniform(5, 11) * m)
    src = r.choice(m, size=nb, p=w)
    dst = r.choice(m, size=nb, p=w)
    domestic = r.random(nb) < 0.75
    for c in np.unique(ctys):
        ix = np.where(ctys == c)[0]
        mk = domestic & (ctys[src] == c)
        if mk.any():
            dst[mk] = r.choice(ix, size=int(mk.sum()))
    am = np.clip(r.lognormal(6.95, 1.05, nb) * np.sqrt(size[src] * size[dst]), 1.0, 250_000.0)
    big = r.random(nb) < 0.01
    am[big] = r.uniform(15_000, 120_000, int(big.sum()))
    p.add(ids_a[src], ids_a[dst], np.round(am, 2), p.ts(nb, str(r.choice(["business", "any"], p=[0.7, 0.3]))),
          p.fmt(nb, {k: 1 for k in PAYMENT_FORMATS}))


def _offshore_dealer(p: _Pop) -> None:
    """Legitimate businesses that trade with, or are registered in, offshore/secrecy jurisdictions (shipping,
    holding structures, fund managers, shell-like holding companies that are real)."""
    r = p.rng
    off = [p.acct(int(r.choice([1, 2])), len(NORMAL_COUNTRIES) + int(r.integers(0, len(OFFSHORE)))) for _ in range(int(r.integers(1, 4)))]
    for _ in range(int(r.integers(1, 5))):
        b = p.acct(int(r.choice([1, 2])), None if r.random() < 0.7 else len(NORMAL_COUNTRIES) + int(r.integers(0, len(OFFSHORE))))
        for o_ in off:
            n = int(r.integers(2, 14))
            amount = p.amt(n, float(r.uniform(2000, 40000)), 0.6, hi=200_000, round_share=float(r.choice([0.0, 0.6])))
            if r.random() < 0.5:
                p.add(b, o_, amount, p.ts(n, "business"), p.fmt(n, {"wire": 1}), p.memo(n, "INVOICE PMT", 0.5))
            else:
                p.add(o_, b, amount, p.ts(n, "business"), p.fmt(n, {"wire": 1}), p.memo(n, "INVOICE PMT", 0.5))


def _cohort_onboarded(p: _Pop) -> None:
    """A group of accounts opened together because they belong together: a new-hire intake, a school year,
    an employer's staff, a co-operative's members. Opened the same day, then used together."""
    r = p.rng
    day_age = float(r.uniform(5, 700))
    org = p.acct(1)
    members = [p.acct(0, p.cty[org], age=day_age + float(r.uniform(0, 1.5))) for _ in range(int(r.integers(8, 60)))]
    base = float(r.uniform(150, 2500))
    for m in members:
        k = int(r.integers(1, 4))
        p.add(org, m, p.amt(k, base, 0.3, round_share=0.5), p.ts(k, "morning"), p.fmt(k, {"ach": 1}), ["SALARY"] * k)
    for _ in range(int(r.integers(3, 25))):
        a, b = r.choice(len(members), 2, replace=False)
        p.add([members[a]], [members[b]], p.amt(1, float(r.uniform(10, 200)), 0.7), p.ts(1, "evening"), p.fmt(1, {"ach": 1, "card": 1}))


def _forwarding_agent(p: _Pop) -> None:
    """Accountants, lawyers, freight forwarders and payment agents: legitimately receive a sum and pass it on within
    hours, keeping a fee, and sometimes pay in amounts just under a reporting line."""
    r = p.rng
    agent = p.acct(1)
    for _ in range(int(r.integers(3, 15))):
        c, v = p.acct(int(r.choice([0, 1]))), p.acct(1, int(r.integers(0, len(NORMAL_COUNTRIES))))
        amount = float(r.choice([r.uniform(1000, 9000), r.uniform(8000, 9900), r.uniform(10000, 90000)]))
        t = float(r.uniform(0, 0.9 * p.W))
        p.add(c, agent, [amount], [t], p.fmt(1, {"wire": 1}), ["PAYMENT"])
        p.add(agent, v, [amount * float(r.uniform(0.92, 0.99))], [t + float(r.uniform(0.05, 2)) * DAY], p.fmt(1, {"wire": 1}), ["PAYMENT"])


def _collector(p: _Pop) -> None:
    """Very large fan-in: a utility, ride-hail, telecom or school collecting from hundreds to thousands of payers,
    often in a short burst (payday, flash sale, fees deadline)."""
    r = p.rng
    m = p.acct(1)
    n = int(r.integers(300, 2000))
    payers = p.people(min(n, 700), p.cty[m] if r.random() < 0.9 else None)
    burst = r.random() < 0.5
    b0, bl = float(r.uniform(0, 0.8 * p.W)), float(r.uniform(1, 3) * DAY)
    t = p.ts(n, "any", lo=b0, hi=b0 + bl) if burst else p.ts(n, "any")
    p.add([payers[int(r.integers(0, len(payers)))] for _ in range(n)], m, p.amt(n, float(r.uniform(20, 1200)), 0.9, hi=9800),
          t, p.fmt(n, {"card": 6, "wire": 1, "ach": 2}))


def _profile_accounts(p: _Pop) -> None:
    """The catch-all: one account whose behavioural knobs are drawn INDEPENDENTLY over wide ranges (degree, amount
    scale, spread, cross-border share, payment mix, timing, regularity, round amounts), so combinations no archetype
    names are still represented."""
    r = p.rng
    a = p.acct(int(r.choice([0, 1, 2], p=[0.6, 0.35, 0.05])))
    cps = [p.acct(int(r.choice([0, 1]))) for _ in range(int(np.exp(r.uniform(0, np.log(60)))))]
    xb = float(r.choice([0.0, 0.0, r.uniform(0, 1)]))
    for direction in (0, 1):
        n = int(np.exp(r.uniform(0, np.log(300)))) if r.random() < 0.85 else 0
        if n == 0:
            continue
        pool = [c for c in cps if (p.cty[c] != p.cty[a]) == (r.random() < xb)] or cps
        other = [pool[int(r.integers(0, len(pool)))] for _ in range(n)]
        am = p.amt(n, float(np.exp(r.uniform(np.log(5), np.log(20000)))), float(r.uniform(0.2, 1.4)), hi=200_000,
                   round_share=float(r.choice([0.0, 0.0, r.uniform(0, 1)])))
        t = p.ts(n, str(r.choice(["business", "morning", "evening", "night", "any"])),
                 lo=float(r.choice([0.0, r.uniform(0, 0.7) * p.W])))
        mix = {k: float(r.random()) + 0.01 for k in PAYMENT_FORMATS}
        if direction == 0:
            p.add(other, a, am, t, p.fmt(n, mix))
        else:
            p.add(a, other, am, t, p.fmt(n, mix))


# (function, share of the accounts budget, rough accounts per call)
CATALOGUE = [
    (_salaried, 14, 22), (_merchant_fan_in, 6, 60), (_cash_business, 6, 85), (_payroll, 4, 160), (_remitter, 8, 4),
    (_trader, 6, 12), (_startup, 4, 20), (_supply_chain, 4, 4), (_subscription, 5, 140), (_chama, 4, 13),
    (_escrow, 3, 3), (_marketplace, 3, 190), (_freelancer, 5, 5), (_student, 4, 4), (_high_net_worth, 3, 8),
    (_dormant_seasonal, 4, 7), (_refund_round_trip, 3, 22), (_gig_worker, 4, 3), (_gambling_or_crypto_user, 3, 16),
    (_charity, 2, 110), (_cash_pooling, 2, 6), (_friends_split, 3, 6), (_taxpayer, 1, 22), (_night_shift_business, 3, 60),
    (_saver, 5, 2), (_profile_accounts, 14, 14), (_background_pool, 40, 1200),
    (_offshore_dealer, 3, 8), (_cohort_onboarded, 4, 35), (_forwarding_agent, 3, 20), (_collector, 3, 500),
]


def generate_normal_population(n_accounts: int = 20000, days: int = 30, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (accounts, transactions) frames in the same schema as aml_synth's output, with every account normal."""
    p = _Pop(days, seed)
    total_share = sum(s for _, s, _ in CATALOGUE)
    for fn, share, per in CATALOGUE:
        budget = n_accounts * share / total_share
        calls = max(3, int(round(budget / per)))          # at least 3 instances of EVERY archetype, however small the run
        for _ in range(calls):
            fn(p)
    n = len(p.kind)
    ids = np.array([f"REF{i:07d}" for i in range(n)])
    cty = np.array(p.cty)
    accounts = pd.DataFrame({
        "account_id": ids,
        "account_type": np.array(["individual", "business", "shell"])[np.array(p.kind)],
        "country": np.array(COUNTRIES)[cty],
        "opened_ts": (START_TS - np.array(p.age) * DAY).astype(np.int64),
        "risk_score": np.round(np.array(p.risk), 4),
        "is_suspicious": 0, "typology": "none",
    })
    src, dst, amt, ts, fmt, memo, dis = (list(c) for c in zip(*p.rows))
    src, dst = np.array(src), np.array(dst)
    order = np.argsort(np.array(ts), kind="stable")
    hr = np.array(p.highrisk, dtype=bool)
    tx = pd.DataFrame({
        "tx_id": [f"RTX{i:09d}" for i in range(len(src))],
        "src": ids[src][order], "dst": ids[dst][order], "amount": np.round(np.array(amt), 2)[order], "currency": "USD",
        "payment_format": np.array(PAYMENT_FORMATS)[np.array(fmt)][order],
        "timestamp": (START_TS + np.array(ts)).astype(np.int64)[order],
        "cross_border": (cty[src] != cty[dst]).astype(np.int64)[order],
        "is_laundering": 0, "typology": "none", "pattern_id": -1,
        "category": np.where(hr[dst], "high_risk_dest", "standard")[order],
        "memo": np.array(memo)[order], "disputed": np.array(dis)[order],
    })
    return accounts, tx


# ---------------------------------------------------------------------------------------------------------
# Build (train + calibrate + freeze) and score
# ---------------------------------------------------------------------------------------------------------
def _arrays_for(accounts, tx, mean, std):
    """Graph arrays for the reference population, standardized with Model 1's OWN mean/std -- the same scaling the
    live graph.pt uses, so a reference account and a live account with identical behaviour have identical features."""
    from gnn_aml_core.features import build_edge_features, build_node_features, make_bidirectional, standardize

    x_raw, names = build_node_features(accounts, tx)
    x, _, _ = standardize(x_raw, mean, std)
    idx = {a: i for i, a in enumerate(accounts["account_id"])}
    s = tx["src"].map(idx).to_numpy(dtype=np.int64)
    d = tx["dst"].map(idx).to_numpy(dtype=np.int64)
    attr, etype = build_edge_features(tx)
    s2, d2, attr2, _ = make_bidirectional(s, d, attr, etype)
    return {"x": x, "edge_index": np.stack([s2, d2]), "edge_attr": attr2, "feature_names": names}


def _components(model, arrays, feat_scale=None, centre=None, fit_idx=None):
    """Runs the frozen network and returns the three raw neural signals per node, plus the scale/centre used."""
    import torch

    from gnn_aml_core.anomaly import _structure_error

    dev = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(arrays["x"])).float().to(dev)
    ei = torch.from_numpy(np.asarray(arrays["edge_index"])).to(dev)
    ea = torch.from_numpy(np.asarray(arrays["edge_attr"])).float().to(dev)
    model.eval()
    with torch.no_grad():
        recon, z = model(x, ei, edge_attr=ea)
        sq = (recon - x) ** 2
        if feat_scale is None:
            sel = torch.from_numpy(np.asarray(fit_idx)).long().to(dev)
            feat_scale = sq[sel].mean(dim=0).clamp_min(1e-8).cpu()
            centre = z[sel].mean(dim=0, keepdim=True).cpu()
        feat = (sq / feat_scale.to(dev)).mean(dim=1)
        emb = ((z - centre.to(dev)) ** 2).sum(dim=1)
        struct = _structure_error(z, ei, x.shape[0])
        per_feature = (sq / feat_scale.to(dev)).cpu().numpy()
    return ({"feature": feat.cpu().numpy(), "embedding": emb.cpu().numpy(), "structure": struct.cpu().numpy(),
             "_per_feature": per_feature}, feat_scale, centre)


def _mad_stats(v):
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad < 1e-12:
        mad = float(np.std(v))
    return med, (mad if mad > 1e-12 else 1.0)


def combine(components: dict, calib: dict) -> np.ndarray:
    """Per-signal robust z against the REFERENCE's own held-out normal accounts, then half the strongest signal
    plus half the average (so one strong signal, or several moderate ones, both count)."""
    zs = np.stack([(components[k] - calib[k][0]) / calib[k][1] for k in ("feature", "embedding", "structure")])
    return np.clip(0.5 * zs.max(axis=0) + 0.5 * zs.mean(axis=0), -3.0, None) + 3.0


def build_reference(models_dir: str = "models", n_accounts: int = 20000, epochs: int = 150, seed: int = 7,
                    hidden: int = 64, heads: int = 4, layers: int = 2, lr: float = 0.001) -> dict:
    """Generates the normal-only population, trains the graph autoencoder on 85% of it, calibrates on the 15% it
    never saw, and writes models/anomaly_reference.pt. The OLD live-trained checkpoint is destroyed."""
    import torch

    from gnn_aml_core.anomaly import build_autoencoder, train_autoencoder

    out = Path(models_dir)
    ckpt = torch.load(out / "gnn_model.pt", map_location="cpu", weights_only=False)
    mean, std = ckpt["mean"].numpy(), ckpt["std"].numpy()
    t0 = time.time()
    accounts, tx = generate_normal_population(n_accounts, seed=seed)
    log.info("reference population: %d normal accounts, %d transactions", len(accounts), len(tx))
    arrays = _arrays_for(accounts, tx, mean, std)
    if arrays["feature_names"] != ckpt["feature_names"]:
        raise ValueError("feature set differs from Model 1's -- retrain Model 1 or rebuild after features change")

    n = len(accounts)
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(0.85 * n)
    train_idx, hold_idx = np.sort(perm[:cut]), np.sort(perm[cut:])
    res = train_autoencoder(arrays, train_idx, hidden=hidden, heads=heads, layers=layers, lr=lr, epochs=epochs,
                            mode="ensemble", trim=0.0, seed=seed)
    model = build_autoencoder(arrays["x"].shape[1], hidden=hidden, heads=heads, layers=layers)
    model.load_state_dict(res["state_dict"])
    comps, feat_scale, centre = _components(model, arrays, fit_idx=train_idx)
    calib = {k: _mad_stats(comps[k][hold_idx]) for k in ("feature", "embedding", "structure")}
    comb_hold = combine(comps, calib)[hold_idx]
    comb_med, comb_mad = _mad_stats(comb_hold)
    torch.save({"state_dict": res["state_dict"], "hparams": res["hparams"], "calib": calib,
                "feat_scale": feat_scale, "centre": centre, "combined_median": comb_med, "combined_mad": comb_mad,
                "feature_names": arrays["feature_names"], "n_accounts": n, "epochs": epochs, "ts": time.time(),
                "kind": "normal-only synthetic reference"}, out / "anomaly_reference.pt")
    for old in ("anomaly_autoencoder.pt", "anomaly_streaks.json"):       # destroy the live-trained model
        try:
            (out / old).unlink()
            log.info("removed old %s", old)
        except FileNotFoundError:
            pass
    log.info("reference model built in %.0fs (train loss %.5f)", time.time() - t0, res["train_loss"])
    return {"accounts": n, "transactions": int(len(tx)), "train_loss": res["train_loss"], "seconds": round(time.time() - t0)}


def load_reference(models_dir: str):
    import torch

    from gnn_aml_core.anomaly import build_autoencoder

    path = Path(models_dir) / "anomaly_reference.pt"
    if not path.exists():
        return None
    ref = torch.load(path, map_location="cpu", weights_only=False)
    hp = ref["hparams"]
    model = build_autoencoder(len(ref["feature_names"]), hidden=hp["hidden"], heads=hp["heads"], layers=hp["layers"],
                              dropout=hp.get("dropout", 0.2))
    model.load_state_dict(ref["state_dict"])
    model.eval()
    return model, ref


def score_live(models_dir: str, arrays: dict, k_std: float):
    """Scores a live graph against the frozen reference. Returns (scores, threshold, extras) or None if no reference exists.
    Nothing here trains or changes the network."""
    loaded = load_reference(models_dir)
    if loaded is None:
        return None
    model, ref = loaded
    comps, _, _ = _components(model, arrays, feat_scale=ref["feat_scale"], centre=ref["centre"])
    scores = combine(comps, ref["calib"])
    threshold = ref["combined_median"] + k_std * ref["combined_mad"]
    z = {k: (comps[k] - ref["calib"][k][0]) / ref["calib"][k][1] for k in ("feature", "embedding", "structure")}
    extras = {"z": z, "per_feature": comps["_per_feature"], "feature_names": ref["feature_names"]}
    return scores, float(threshold), extras


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="generate the normal-only population, train, calibrate, freeze")
    ap.add_argument("--models-dir", default=os.getenv("MODELS_DIR", "models"))
    ap.add_argument("--accounts", type=int, default=int(os.getenv("ANOMALY_REF_ACCOUNTS", 20000)))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("ANOMALY_REF_EPOCHS", 150)))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.build:
        print(build_reference(a.models_dir, a.accounts, a.epochs, a.seed))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
