#!/usr/bin/env python3
"""Makes a small, realistic bank-transaction sample for the Intake tab.
Run from the project folder:  python tester-pack/sample-data/make_sample.py
Writes three files next to this script:
  sample_transactions.csv   paste or upload this into the Intake tab (src,dst,amount,timestamp,cross_border)
  answer_key_accounts.csv   which accounts are really suspicious (for the owner only, never given to testers)
  sample_summary.txt        what is in the sample
All data is synthetic, so no personal data is involved."""
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1])); sys.path.insert(0, str(Path.cwd()))
from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig
ACC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
G = AMLGraphGenerator(GeneratorConfig(n_accounts=ACC, n_background_tx=4 * ACC, n_patterns_per_typology=1)).generate()
tx = G.transactions.sort_values("timestamp")
tx[["src", "dst", "amount", "timestamp", "cross_border"]].to_csv(here / "sample_transactions.csv", index=False)
G.accounts[["account_id", "is_suspicious", "typology", "account_type", "country"]].to_csv(here / "answer_key_accounts.csv", index=False)
sus = G.accounts[G.accounts.is_suspicious == 1]
lines = [f"Accounts: {len(G.accounts)}", f"Transactions: {len(tx)}", f"Suspicious accounts (hidden answer key): {len(sus)}",
         "Suspicious by pattern: " + ", ".join(f"{k} {v}" for k, v in sus.typology.value_counts().items()),
         f"Amount median {tx.amount.median():.0f}, 95th percentile {tx.amount.quantile(.95):.0f}, largest {tx.amount.max():.0f}",
         f"Cross-border share: {tx.cross_border.mean()*100:.0f} percent",
         "Also contains look-alike honest activity: payroll, merchants with many customers, cash-heavy businesses, savings groups, remittances."]
(here / "sample_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
