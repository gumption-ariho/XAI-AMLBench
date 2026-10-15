#!/usr/bin/env python3
"""
track_metrics.py  -  records real, public adoption metrics for the petition evidence packet (brief section 4.5).

Reads ONLY public counts (GitHub API, PyPI's public JSON API, Hugging Face's public API). Nothing here can
inflate a number -- it just fetches what is already public and appends one dated row per run to a CSV, so a
history builds up automatically as you run this weekly. It never modifies GitHub, PyPI, Zenodo or Hugging Face.

    python3 track_metrics.py --repo OWNER/xai-amlbench --pypi xai-amlbench --hf-dataset OWNER/xai-amlbench
    python3 track_metrics.py --repo OWNER/xai-amlbench --pypi xai-amlbench --zenodo-doi 10.5281/zenodo.XXXXXXX

Zenodo has no simple public "download count" API endpoint the way GitHub/PyPI/HF do; --zenodo-doi just records
the DOI as a reminder to note its download/citation count by hand from the record's own page. Run this weekly
(a plain cron job or a scheduled GitHub Action is enough; see .github/workflows/ci.yml for the pattern) and keep
metrics.csv in the repository so its history is itself part of the public evidence trail.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

FIELDS = ["date", "github_stars", "github_forks", "github_watchers", "github_open_issues",
          "pypi_downloads_last_month", "pypi_version", "hf_downloads", "hf_likes", "zenodo_doi", "notes"]


def get_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": "xai-amlbench-metrics-tracker", **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def github_metrics(repo: str) -> dict:
    try:
        d = get_json(f"https://api.github.com/repos/{repo}")
        return {"github_stars": d.get("stargazers_count"), "github_forks": d.get("forks_count"),
                "github_watchers": d.get("subscribers_count", d.get("watchers_count")),
                "github_open_issues": d.get("open_issues_count")}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        print(f"  warning: could not fetch GitHub metrics for {repo}: {exc}", file=sys.stderr)
        return {}


def pypi_metrics(package: str) -> dict:
    out = {}
    try:
        d = get_json(f"https://pypi.org/pypi/{package}/json")
        out["pypi_version"] = d.get("info", {}).get("version")
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        print(f"  warning: could not fetch PyPI metadata for {package}: {exc}", file=sys.stderr)
    try:
        # pypistats.org mirrors the official BigQuery download stats; a few days' lag is normal.
        d = get_json(f"https://pypistats.org/api/packages/{package}/recent")
        out["pypi_downloads_last_month"] = d.get("data", {}).get("last_month")
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        print(f"  warning: could not fetch PyPI download stats for {package}: {exc}", file=sys.stderr)
    return out


def huggingface_metrics(dataset: str) -> dict:
    try:
        d = get_json(f"https://huggingface.co/api/datasets/{dataset}")
        return {"hf_downloads": d.get("downloads"), "hf_likes": d.get("likes")}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        print(f"  warning: could not fetch Hugging Face metrics for {dataset}: {exc}", file=sys.stderr)
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="GitHub repo, e.g. OWNER/xai-amlbench")
    ap.add_argument("--pypi", help="PyPI package name, e.g. xai-amlbench")
    ap.add_argument("--hf-dataset", help="Hugging Face dataset id, e.g. OWNER/xai-amlbench")
    ap.add_argument("--zenodo-doi", default="", help="Zenodo DOI, once minted (recorded as-is; check its download/citation count by hand)")
    ap.add_argument("--notes", default="", help="free-text note for this week's row, e.g. a beta-tester milestone")
    ap.add_argument("--out", default=str(Path(__file__).with_name("metrics.csv")))
    a = ap.parse_args()

    if not any((a.repo, a.pypi, a.hf_dataset, a.zenodo_doi)):
        ap.error("pass at least one of --repo, --pypi, --hf-dataset, --zenodo-doi")

    row = {"date": datetime.date.today().isoformat(), "notes": a.notes, "zenodo_doi": a.zenodo_doi}
    if a.repo:
        print(f"fetching GitHub metrics for {a.repo} ...")
        row.update(github_metrics(a.repo))
    if a.pypi:
        print(f"fetching PyPI metrics for {a.pypi} ...")
        row.update(pypi_metrics(a.pypi))
    if a.hf_dataset:
        print(f"fetching Hugging Face metrics for {a.hf_dataset} ...")
        row.update(huggingface_metrics(a.hf_dataset))

    out = Path(a.out)
    is_new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"\nappended one row to {out}:")
    for k in FIELDS:
        if row.get(k) not in (None, ""):
            print(f"  {k}: {row[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
