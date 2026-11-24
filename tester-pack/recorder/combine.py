#!/usr/bin/env python3
"""Owner tool: combines every tester's zip into one tracking workbook and a filled-in client report.
Usage:  python combine.py  folder-with-the-zips  [output-folder]
Needs:  pip install openpyxl python-docx
"""
import json, sys, zipfile, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from thresholds import RULES, CAPS, SRC
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter as L
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

src = Path(sys.argv[1]); out = Path(sys.argv[2] if len(sys.argv) > 2 else "combined"); out.mkdir(exist_ok=True, parents=True)
runs = []
for z in sorted(src.glob("*.zip")):
    with zipfile.ZipFile(z) as f:
        n = [x for x in f.namelist() if x.endswith("results.json")]
        if n: runs.append(json.loads(f.read(n[0])))
if not runs: sys.exit("No results found in " + str(src))
N = len(runs)

def pct(vals, q):
    v = sorted(vals)
    if not v: return None
    return v[min(len(v) - 1, max(0, int(round(q * len(v) + 0.4999)) - 1))]

rule = lambda r, i: next((x for x in r["rules"] if x["id"] == i), {"status": "Not measured", "measured": "", "detail": ""})
cap = lambda r, i: next((x for x in r["capabilities"] if x["id"] == i), {"status": "Not exercised", "calls": 0, "detail": ""})
per = {}
for rid, *_ in RULES:
    st = [rule(r, rid)["status"] for r in runs]
    per[rid] = {k: st.count(k) for k in ("Pass", "Fail", "Not measured", "Not exercised", "N/A")}
tp = sum(v["Pass"] for v in per.values()); tf = sum(v["Fail"] for v in per.values())
rate = tp / (tp + tf) * 100 if tp + tf else 0
capper = {cid: {k: sum(1 for r in runs if cap(r, cid)["status"] == k) for k in ("Pass", "Fail", "Needs review", "Not exercised")} for cid, *_ in CAPS}
cap_used = sum(1 for v in capper.values() if v["Pass"] + v["Fail"] + v["Needs review"] > 0)
api = [x for r in runs for x in r["measurements"]["api_ms"]]; scan = [x for r in runs for x in r["measurements"]["scan_ms"]]
vit = {k: [v.get(k) for r in runs for v in r["measurements"]["vitals"].values() if v.get(k) is not None] for k in ("ttfb", "lcp", "inp", "cls")}
n_api = sum(r["measurements"]["n_api"] for r in runs); n_5 = sum(r["measurements"]["n_5xx"] for r in runs)
perf = [("Server calls measured", n_api), ("Server errors (5xx)", n_5),
        ("Server response time, median", f"{pct(api,.5)} ms" if api else "n/a"), ("Server response time, 95th percentile", f"{pct(api,.95)} ms" if api else "n/a"),
        ("Account scoring time, 95th percentile", f"{round(pct(scan,.95),1)} ms" if scan else "n/a"),
        ("Page time to first byte, 75th percentile", f"{pct(vit['ttfb'],.75)} ms" if vit["ttfb"] else "n/a"),
        ("Largest Contentful Paint, 75th percentile", f"{pct(vit['lcp'],.75)} ms" if vit["lcp"] else "n/a"),
        ("Interaction to Next Paint, 75th percentile", f"{pct(vit['inp'],.75)} ms" if vit["inp"] else "n/a"),
        ("Layout shift, 75th percentile", f"{round(pct(vit['cls'],.75),3)}" if vit["cls"] else "n/a")]
findings = []
for rid, cat, title, thr, srcs in RULES:
    fails = [(r["tester"], rule(r, rid)) for r in runs if rule(r, rid)["status"] == "Fail"]
    if fails:
        det = sorted({x["detail"] for _, x in fails if x["detail"]})[:3]
        sug = next((r2["suggestion"] for r in runs for r2 in r["rules"] if r2["id"] == rid and r2.get("suggestion")), "")
        findings.append({"id": rid, "title": title, "n": len(fails), "detail": " | ".join(det), "suggestion": sug, "source": srcs})
findings.sort(key=lambda f: -f["n"])
scores = [float(r["opinion"]["score"]) for r in runs if r.get("opinion", {}).get("score")]
avg_score = sum(scores) / len(scores) if scores else None
help_yes = sum(1 for r in runs if r.get("opinion", {}).get("help") == "Yes")

# ---------- workbook ----------
F = Font(name="Times New Roman", size=11); FB = Font(name="Times New Roman", size=11, bold=True)
thin = Side(style="thin"); BR = Border(left=thin, right=thin, top=thin, bottom=thin)
def sheet(wb, title, header, rows, widths):
    ws = wb.create_sheet(title); ws.append(header)
    for r in rows: ws.append(r)
    for row in ws.iter_rows():
        for c in row: c.font = FB if c.row == 1 else F; c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = BR
    for i, w in enumerate(widths, 1): ws.column_dimensions[L(i)].width = w
    ws.freeze_panes = "A2"; return ws
wb = Workbook(); wb.remove(wb.active)
ws = sheet(wb, "Technical results", ["ID", "Category", "Check", "Standard"] + [r["tester"] for r in runs] + ["Passed", "Failed", "Pass rate"],
           [[rid, cat, title, thr] + [rule(r, rid)["status"] for r in runs] + [None, None, None] for rid, cat, title, thr, _ in RULES], [6, 14, 40, 44] + [12] * N + [9, 9, 10])
for rr in range(2, 2 + len(RULES)):
    rng = f"{L(5)}{rr}:{L(4+N)}{rr}"
    ws.cell(rr, 5 + N, f'=COUNTIF({rng},"Pass")'); ws.cell(rr, 6 + N, f'=COUNTIF({rng},"Fail")')
    ws.cell(rr, 7 + N, f'=IF({L(5+N)}{rr}+{L(6+N)}{rr}=0,"",{L(5+N)}{rr}/({L(5+N)}{rr}+{L(6+N)}{rr}))').number_format = "0%"
sheet(wb, "Measured values", ["ID", "Check"] + [r["tester"] for r in runs], [[rid, title] + [rule(r, rid)["measured"] for r in runs] for rid, _, title, *_ in RULES], [6, 40] + [26] * N)
sheet(wb, "Features used", ["ID", "Feature"] + [r["tester"] for r in runs] + ["Worked", "Failed", "Needs review", "Not used"],
      [[cid, t] + [cap(r, cid)["status"] for r in runs] + [capper[cid]["Pass"], capper[cid]["Fail"], capper[cid]["Needs review"], capper[cid]["Not exercised"]] for cid, t, _ in CAPS], [6, 44] + [12] * N + [8, 8, 12, 9])
sheet(wb, "Performance", ["Measure", "Value (all testers pooled)"], [[a, b] for a, b in perf], [48, 26])
sheet(wb, "Findings", ["No.", "Check", "What failed", "Testers affected", "Detail recorded", "Suggested improvement", "Standard", "Seriousness", "Fix made", "Fixed in", "Retested"],
      [[f"B-{i:02d}", f"{f['id']} {f['title']}", f["title"], f"{f['n']} of {N}", f["detail"], f["suggestion"], f["source"], "", "", "", ""] for i, f in enumerate(findings, 1)], [8, 34, 34, 10, 46, 46, 46, 12, 22, 10, 10])
sheet(wb, "Opinions", ["Tester", "Setup without help", "Overall impression (1 to 5)", "Overall opinion", "Suggestions for improvement"],
      [[r["tester"], r.get("opinion", {}).get("help", ""), r.get("opinion", {}).get("score", ""), r.get("opinion", {}).get("opinion", ""), r.get("opinion", {}).get("suggestions", "")] for r in runs], [14, 14, 14, 70, 70])
wb.save(out / "XAI-AMLBench-Test-Tracker.xlsx")

# ---------- report ----------
FONT = "Times New Roman"; BLK = RGBColor(0, 0, 0)
d = Document()
for s in d.sections: s.top_margin = s.bottom_margin = Inches(1); s.left_margin = s.right_margin = Inches(1.1)
def sf(st, sz, b=False):
    st.font.name = FONT; st.font.size = Pt(sz); st.font.bold = b; st.font.color.rgb = BLK
    rp = st.element.get_or_add_rPr(); rf = rp.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rp.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
sf(d.styles["Normal"], 12); d.styles["Normal"].paragraph_format.space_after = Pt(8); d.styles["Normal"].paragraph_format.line_spacing = 1.15
for n, z in (("Heading 1", 16), ("Heading 2", 13.5)): sf(d.styles[n], z, True); d.styles[n].paragraph_format.keep_with_next = True
sf(d.styles["List Bullet"], 12)
def P(t, b=False, al=None, sz=None, after=None):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = b
    if sz: r.font.size = Pt(sz)
    if al == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if after is not None: p.paragraph_format.space_after = Pt(after)
def B(items):
    for t in items: d.add_paragraph(t, style="List Bullet").paragraph_format.space_after = Pt(3)
def T(h, rows, w=None, sz=10):
    t = d.add_table(rows=1, cols=len(h)); t.autofit = False; bd = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "6"); x.set(qn("w:space"), "0"); x.set(qn("w:color"), "000000"); bd.append(x)
    t._tbl.tblPr.append(bd)
    for i, x in enumerate(h):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(x); r.bold = True; r.font.size = Pt(sz)
    for row in rows:
        cs = t.add_row().cells
        for i, x in enumerate(row): cs[i].text = ""; cs[i].paragraphs[0].add_run(str(x)).font.size = Pt(sz)
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); t.rows[0]._tr.get_or_add_trPr().append(th)
    if w:
        for r in t.rows:
            for i, x in enumerate(w): r.cells[i].width = Inches(x)
    d.add_paragraph().paragraph_format.space_after = Pt(4)
p = d.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("XAI-AMLBench Test Report    Page "); r.font.size = Pt(10); r.font.name = FONT
r = p.add_run()
for t, k in (("begin", None), (None, "PAGE"), ("end", None)):
    if t: e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
    else: e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = k
    r._r.append(e)
r.font.size = Pt(10); r.font.name = FONT
for _ in range(7): P("")
P("XAI-AMLBench", True, "c", 28, 6); P("Test Report", True, "c", 20, 4); P(f"Version 0.1 tested by {N} independent testers", False, "c", 14, 36)
P(f"Date of report: {datetime.date.today().strftime('%d %B %Y')}", al="c", after=2); P("Prepared for: [client name]", al="c", after=2); P("Prepared by: [your name]", al="c", after=2)
d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
d.add_heading("1. Purpose of the test", 1)
P("This report records a structured test of XAI-AMLBench version 0.1 by people other than its builders. The aim was to confirm that the whole platform works as designed in real use, and to measure it against published engineering standards and Kenyan law, so that the result does not rest on anyone's opinion.")
d.add_heading("2. How the test was done", 1)
B([f"{N} testers each ran their own copy of version 0.1 and used it normally in a recording browser. Testers used made-up banks, people and accounts only.",
   "While they worked, the recorder measured the system silently: page speed, server response times, accessibility, security headers, protection against access without sign-in, separation between banks, error handling, and whether secrets or raw ID numbers were ever sent to the browser.",
   "Each measurement is judged against a published standard, listed in the table below. A measurement is marked Pass, Fail, Not exercised (the tester did not use that part), Not measured (no data) or N/A (not applicable on a local test, for example HTTPS rules).",
   "Testers did not mark anything. They gave only an overall opinion at the end, which is reported separately in section 7.",
   "The recorder never stored typed text or passwords. Session tokens were held in memory only."])
d.add_heading("2.1 The standards used", 2)
T(["ID", "Check", "Threshold", "Source"], [[rid, title, thr, s] for rid, _, title, thr, s in RULES], [0.45, 1.6, 2.0, 2.3], 9)
d.add_heading("2.2 The testers", 2)
T(["No.", "Tester", "Active time (minutes)", "Pages visited", "Server calls"], [[i, r["tester"], r["active_minutes"], len(r["pages"]), r["measurements"]["n_api"]] for i, r in enumerate(runs, 1)], [0.5, 1.7, 1.6, 1.4, 1.2])
d.add_heading("3. Results of the technical checks", 1)
P(f"Each cell gives the number of testers, out of {N}, whose recording passed or failed that check. A check that a tester did not exercise is not counted against the system.")
cats = []
for rid, cat, *_ in RULES:
    if cat not in cats: cats.append(cat)
for i, cat in enumerate(cats, 1):
    d.add_heading(f"3.{i} {cat}", 2)
    T(["ID", "Check", "Passed", "Failed", "Not exercised or not measured", "Not applicable"],
      [[rid, title, per[rid]["Pass"], per[rid]["Fail"], per[rid]["Not exercised"] + per[rid]["Not measured"], per[rid]["N/A"]] for rid, c, title, *_ in RULES if c == cat], [0.5, 2.7, 0.7, 0.7, 1.3, 0.9], 10)
d.add_heading("4. Summary of results", 1)
T(["Measure", "Result"], [["Technical checks defined", len(RULES)], ["Results recorded as Pass", tp], ["Results recorded as Fail", tf], ["Pass rate (Pass out of Pass plus Fail)", f"{rate:.0f} %"],
   ["Checks that passed for every tester who exercised them", sum(1 for v in per.values() if v["Pass"] > 0 and v["Fail"] == 0)],
   ["Features of the platform used during the test", f"{cap_used} of {len(CAPS)}"]], [3.8, 2.4], 11)
P(f"Across {N} testers, {tp} technical results passed and {tf} failed, a pass rate of {rate:.0f} percent. [Add one or two sentences on which areas were strongest.]")
d.add_heading("5. Performance measured", 1)
T(["Measure", "Value (all testers pooled)"], [[a, b] for a, b in perf], [3.8, 2.4], 11)
d.add_heading("6. Features used and findings", 1)
d.add_heading("6.1 Features of the platform that testers used", 2)
T(["ID", "Feature", "Worked", "Failed", "Not used"], [[cid, t, capper[cid]["Pass"], capper[cid]["Fail"] + capper[cid]["Needs review"], capper[cid]["Not exercised"]] for cid, t, _ in CAPS], [0.5, 3.6, 0.7, 0.7, 0.7], 9.5)
d.add_heading("6.2 Findings and suggested improvements", 2)
if findings:
    P("Each failed check is listed once, with the number of testers it affected, what was recorded, and the suggested fix.")
    T(["No.", "Check", "Testers", "What was recorded", "Suggested improvement", "Fixed in"], [[f"B-{i:02d}", f"{f['id']} {f['title']}", f"{f['n']} of {N}", f["detail"], f["suggestion"], "[ ]"] for i, f in enumerate(findings, 1)], [0.5, 1.4, 0.6, 1.7, 1.7, 0.6], 9)
else: P("No technical check failed for any tester.")
d.add_heading("7. Testers' overall overviews", 1)
P("The following are the testers' own opinions, written at the end of their session. They are separate from the measurements above and are not counted in any pass rate.")
T(["Measure", "Result"], [["Testers who completed setup without help", f"{help_yes} of {N}"], ["Average overall impression (1 to 5)", f"{avg_score:.1f}" if avg_score else "not given"]], [3.8, 2.4], 11)
for r in runs:
    o = r.get("opinion", {}) or {}
    if o.get("opinion") or o.get("suggestions"):
        P(f"{r['tester']}: {o.get('opinion','')}", after=2)
        if o.get("suggestions"): P(f"Suggestions: {o['suggestions']}")
d.add_heading("8. Conclusion and next steps", 1)
P(f"[State the result of the test in one paragraph.] Version 0.1 has been tested by {N} people, with every result measured automatically against published standards. The next stage is version 1.0, where the platform is placed on a rented server with its own domain name and used by one or two real banks.")
B(["Fix and retest every finding listed in section 6.2.", "Release the corrected version with a new version number.", "Prepare the pilot server, domain name, HTTPS with HSTS and backups for version 1.0."])
d.add_heading("Appendix. Records kept", 1)
P("For each tester the following are kept: the machine-readable results, the readable report, screenshots, a video of the session, the pages visited and the computer details. They are available on request.")
d.save(out / "XAI-AMLBench-Test-Report.docx")
print(f"Combined {N} testers. Pass {tp}, Fail {tf}, pass rate {rate:.0f}%. Files in {out.resolve()}")
