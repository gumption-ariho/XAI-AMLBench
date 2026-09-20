"""Synthetic AML transaction-graph generator (container: aml-synth-worker).

Builds a directed multi-hop transaction network made of
  * benign background traffic (heavy-tailed, preferential-attachment style), and
  * injected laundering patterns for 5 typologies:
      smurfing, scatter_gather, cyclic_loop, shell_company, cross_border_velocity

All data is synthetic, so no personal data is involved.

Run:
    python -m aml_synth.graph_generator --accounts 5000 --patterns 15 --to csv
    python -m aml_synth.graph_generator --to csv,kafka,neo4j
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("aml_synth")

TYPOLOGIES = [
    "smurfing",
    "scatter_gather",
    "cyclic_loop",
    "shell_company",
    "cross_border_velocity",
]

NORMAL_COUNTRIES = ["US", "GB", "DE", "FR", "KE", "NG", "ZA", "IN", "BR", "JP", "CA", "AE"]
COUNTRY_WEIGHTS = np.array([30, 10, 8, 7, 8, 6, 5, 8, 4, 4, 6, 4], dtype=float)
COUNTRY_WEIGHTS /= COUNTRY_WEIGHTS.sum()
OFFSHORE = ["VG", "KY", "PA", "SC", "BZ"]  # synthetic "secrecy jurisdiction" set
PAYMENT_FORMATS = ["wire", "ach", "card", "cash", "crypto_ramp"]

START_TS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
HOUR, DAY = 3600, 86400


@dataclass
class GeneratorConfig:
    n_accounts: int = 5000
    n_background_tx: int = 40000
    n_patterns_per_typology: int = 15
    days: int = 30
    reporting_threshold: float = 10_000.0
    seed: int = 42


@dataclass
class SyntheticGraph:
    accounts: pd.DataFrame
    transactions: pd.DataFrame

    def summary(self) -> dict:
        tx, ac = self.transactions, self.accounts
        return {
            "accounts": int(len(ac)),
            "suspicious_accounts": int(ac["is_suspicious"].sum()),
            "transactions": int(len(tx)),
            "laundering_transactions": int(tx["is_laundering"].sum()),
            "laundering_rate_pct": round(100 * float(tx["is_laundering"].mean()), 3),
            "by_typology": tx.loc[tx["is_laundering"] == 1, "typology"].value_counts().to_dict(),
        }


class AMLGraphGenerator:
    def __init__(self, cfg: GeneratorConfig | None = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._accounts: dict[str, dict] = {}
        self._tx: list[dict] = []
        self._pool: list[str] = []          # benign accounts that exist from the start
        self._pattern_id = 0

    # ------------------------------------------------------------------ public
    def generate(self) -> SyntheticGraph:
        self._make_pool()
        self._make_background()
        injectors = {
            "smurfing": self._smurfing,
            "scatter_gather": self._scatter_gather,
            "cyclic_loop": self._cyclic_loop,
            "shell_company": self._shell_company,
            "cross_border_velocity": self._cross_border_velocity,
        }
        for typology, fn in injectors.items():
            for _ in range(self.cfg.n_patterns_per_typology):
                fn()
                self._pattern_id += 1

        tx = pd.DataFrame(self._tx).sort_values("timestamp", kind="stable").reset_index(drop=True)
        tx.insert(0, "tx_id", [f"TX{i:09d}" for i in range(len(tx))])
        accounts = pd.DataFrame(self._accounts.values())
        accounts["is_suspicious"] = accounts["is_suspicious"].astype(int)
        return SyntheticGraph(accounts=accounts, transactions=tx)

    # --------------------------------------------------------------- accounts
    def _new_account(self, kind: str, country: str | None = None, age_days: float | None = None) -> str:
        acc_id = f"ACC{len(self._accounts):07d}"
        if country is None:
            country = str(self.rng.choice(OFFSHORE)) if kind == "shell" else str(
                self.rng.choice(NORMAL_COUNTRIES, p=COUNTRY_WEIGHTS)
            )
        if age_days is None:
            age_days = float(self.rng.uniform(5, 40)) if kind == "shell" else float(self.rng.uniform(60, 3650))
        self._accounts[acc_id] = {
            "account_id": acc_id,
            "account_type": kind,
            "country": country,
            "opened_ts": int(START_TS - age_days * DAY),
            "is_suspicious": 0,
            "typology": "none",
        }
        return acc_id

    def _flag(self, acc_id: str, typology: str) -> None:
        self._accounts[acc_id]["is_suspicious"] = 1
        self._accounts[acc_id]["typology"] = typology

    def _pick(self, exclude: set[str] | None = None, country_not: str | None = None) -> str:
        exclude = exclude or set()
        for _ in range(200):
            a = self._pool[int(self.rng.integers(len(self._pool)))]
            if a in exclude:
                continue
            if country_not and self._accounts[a]["country"] == country_not:
                continue
            return a
        raise RuntimeError("could not pick a pool account")

    def _make_pool(self) -> None:
        for _ in range(self.cfg.n_accounts):
            kind = "business" if self.rng.random() < 0.15 else "individual"
            self._pool.append(self._new_account(kind))

    # ------------------------------------------------------------ transactions
    def _add_tx(self, src, dst, amount, ts, typology="none", laundering=False, fmt=None):
        cross = int(self._accounts[src]["country"] != self._accounts[dst]["country"])
        self._tx.append({
            "src": src,
            "dst": dst,
            "amount": round(float(amount), 2),
            "currency": "USD",
            "payment_format": fmt or str(self.rng.choice(PAYMENT_FORMATS)),
            "timestamp": int(ts),
            "cross_border": cross,
            "is_laundering": int(laundering),
            "typology": typology if laundering else "none",
            "pattern_id": self._pattern_id if laundering else -1,
        })

    def _make_background(self) -> None:
        n = self.cfg.n_background_tx
        w = self.rng.pareto(1.6, size=len(self._pool)) + 1.0      # heavy-tailed activity
        w /= w.sum()
        # group accounts by country so most benign traffic stays domestic (~75%)
        groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        countries = np.array([self._accounts[a]["country"] for a in self._pool])
        for c in np.unique(countries):
            ix = np.where(countries == c)[0]
            groups[c] = (ix, w[ix] / w[ix].sum())
        src = self.rng.choice(len(self._pool), size=n, p=w)
        dst_any = self.rng.choice(len(self._pool), size=n, p=w)
        domestic = self.rng.random(n) < 0.75
        amounts = np.clip(self.rng.lognormal(6.0, 1.2, size=n), 1, 250_000)
        # a small share of legitimate large transfers (payroll, property, etc.)
        big = self.rng.random(n) < 0.01
        amounts[big] = self.rng.uniform(15_000, 120_000, size=big.sum())
        ts = START_TS + self.rng.uniform(0, self.cfg.days * DAY, size=n)
        for i in range(n):
            s = src[i]
            d = dst_any[i]
            if domestic[i]:
                ix, pr = groups[countries[s]]
                d = ix[self.rng.choice(len(ix), p=pr)]
            if s == d:
                continue
            self._add_tx(self._pool[s], self._pool[d], amounts[i], ts[i])

    # ------------------------------------------------------------- typologies
    def _t0(self) -> float:
        return START_TS + float(self.rng.uniform(0, max(self.cfg.days - 4, 1) * DAY))

    def _smurfing(self) -> None:
        """origin -> many mules (each just below the reporting threshold) -> collector"""
        T, rng, name = self.cfg.reporting_threshold, self.rng, "smurfing"
        origin = self._pick()
        collector = self._new_account("business")
        self._flag(origin, name); self._flag(collector, name)
        t0 = self._t0()
        for _ in range(int(rng.integers(8, 20))):
            mule = self._new_account("individual", age_days=float(rng.uniform(20, 300)))
            self._flag(mule, name)
            amt = float(rng.uniform(0.80, 0.99) * T)
            t1 = t0 + float(rng.uniform(0, 20 * HOUR))
            self._add_tx(origin, mule, amt, t1, name, True, fmt="cash")
            self._add_tx(mule, collector, amt * float(rng.uniform(0.95, 0.99)),
                         t1 + float(rng.uniform(2 * HOUR, 24 * HOUR)), name, True, fmt="wire")

    def _scatter_gather(self) -> None:
        """origin scatters to k intermediaries, who gather into one destination"""
        rng, name = self.rng, "scatter_gather"
        origin = self._pick()
        dest = self._new_account("business")
        self._flag(origin, name); self._flag(dest, name)
        k = int(rng.integers(5, 13))
        total = float(rng.uniform(60_000, 500_000))
        shares = rng.dirichlet(np.ones(k)) * total
        t0 = self._t0()
        for share in shares:
            mid = self._new_account("individual" if rng.random() < .6 else "business")
            self._flag(mid, name)
            t1 = t0 + float(rng.uniform(0, 10 * HOUR))
            self._add_tx(origin, mid, share, t1, name, True, fmt="wire")
            self._add_tx(mid, dest, share * float(rng.uniform(0.97, 0.995)),
                         t1 + float(rng.uniform(6 * HOUR, 36 * HOUR)), name, True, fmt="wire")

    def _cyclic_loop(self) -> None:
        """funds travel around a closed loop several times, shrinking slightly each hop"""
        rng, name = self.rng, "cyclic_loop"
        length = int(rng.integers(3, 7))
        nodes = [self._pick() if rng.random() < 0.4 else self._new_account("business") for _ in range(length)]
        nodes = list(dict.fromkeys(nodes))          # unique, keep order
        if len(nodes) < 3:
            nodes += [self._new_account("business") for _ in range(3 - len(nodes))]
        for n in nodes:
            self._flag(n, name)
        amt = float(rng.uniform(25_000, 180_000))
        t = self._t0()
        for _ in range(int(rng.integers(2, 5))):
            for i, s in enumerate(nodes):
                d = nodes[(i + 1) % len(nodes)]
                self._add_tx(s, d, amt, t, name, True, fmt="wire")
                amt *= float(rng.uniform(0.985, 0.998))
                t += float(rng.uniform(1 * HOUR, 14 * HOUR))

    def _shell_company(self) -> None:
        """origin -> chain of freshly-opened offshore shell accounts -> beneficiary"""
        rng, name = self.rng, "shell_company"
        origin = self._pick()
        beneficiary = self._new_account("business")
        chain = [self._new_account("shell") for _ in range(int(rng.integers(3, 6)))]
        for n in [origin, beneficiary, *chain]:
            self._flag(n, name)
        amt = float(rng.uniform(120_000, 1_200_000))
        t = self._t0()
        path = [origin, *chain, beneficiary]
        for s, d in zip(path[:-1], path[1:]):
            self._add_tx(s, d, amt, t, name, True, fmt="wire")
            amt *= float(rng.uniform(0.985, 0.999))
            t += float(rng.uniform(4 * HOUR, 2 * DAY))

    def _cross_border_velocity(self) -> None:
        """one account moves many mid-size transfers to/from many countries within hours"""
        rng, name = self.rng, "cross_border_velocity"
        subject = self._new_account("individual", age_days=float(rng.uniform(30, 400)))
        self._flag(subject, name)
        home = self._accounts[subject]["country"]
        t0 = self._t0()
        used: set[str] = set()
        for i in range(int(rng.integers(15, 40))):
            cp = self._pick(exclude=used, country_not=home)
            used.add(cp)
            t = t0 + float(rng.uniform(0, 6 * HOUR))
            amt = float(rng.uniform(1_000, 9_500))
            if i % 2 == 0:
                self._add_tx(cp, subject, amt, t, name, True, fmt="wire")
            else:
                self._add_tx(subject, cp, amt * float(rng.uniform(0.9, 0.99)), t, name, True, fmt="crypto_ramp")


# -------------------------------------------------------------------- CLI
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Generate a synthetic AML transaction graph")
    p.add_argument("--accounts", type=int, default=int(os.getenv("SYNTH_ACCOUNTS", 5000)))
    p.add_argument("--background-tx", type=int, default=int(os.getenv("SYNTH_BACKGROUND_TX", 40000)))
    p.add_argument("--patterns", type=int, default=int(os.getenv("SYNTH_PATTERNS", 15)),
                   help="laundering patterns injected per typology")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.getenv("SYNTH_OUT_DIR", "data"))
    p.add_argument("--to", default=os.getenv("EXPORT_TARGETS", "csv"),
                   help="comma list of: csv,kafka,neo4j")
    args = p.parse_args()

    cfg = GeneratorConfig(
        n_accounts=args.accounts, n_background_tx=args.background_tx,
        n_patterns_per_typology=args.patterns, days=args.days, seed=args.seed,
    )
    graph = AMLGraphGenerator(cfg).generate()
    log.info("generated: %s", graph.summary())

    from aml_synth import exporters  # imported lazily so csv-only runs need no kafka/neo4j libs

    targets = {t.strip() for t in args.to.split(",") if t.strip()}
    if "csv" in targets:
        exporters.export_csv(graph, args.out)
    if "neo4j" in targets:
        exporters.export_neo4j(graph)
    if "kafka" in targets:
        exporters.export_kafka(graph)


if __name__ == "__main__":
    main()
