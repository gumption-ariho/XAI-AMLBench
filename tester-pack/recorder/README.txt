XAI-AMLBENCH TEST RECORDER (version 2)
=======================================

WHAT IT DOES
You use the system normally in the browser this tool opens. You do not mark anything and there is no checklist.
While you work, the tool measures the system silently against published standards and Kenyan law, and records
what you did. At the end you press "Finish test" (small button at the bottom left of the page), and write your
overall opinion and suggestions. That is the only place your opinion is asked for, and it is kept separate from the
measurements.

WHAT IT MEASURES (each one against a named standard, listed in the report)
  Performance: page speed (Google Core Web Vitals), server response time (Nielsen Norman 1 second limit),
               time to score an account (project brief, 100 ms), server errors (99.9 percent SLO)
  Accessibility: WCAG 2.2 AA, using the axe-core engine
  Security: sign-in required (OWASP API Top 10), security headers (OWASP Secure Headers Project), CORS,
            error messages, cookie flags, separation between banks, secrets never sent to the browser (OWASP ASVS)
  Kenyan law: Data Protection Act 2019 (data minimisation s.25(d) and s.41(3), encryption s.41(4), records s.43(8)),
              Proceeds of Crime and Anti-Money Laundering Regulations reg. 37 (record keeping)
It also records which parts of the platform you used, and whether each worked.
It never stores what you type or any password. Session tokens are kept in memory only.

FOR EACH TESTER
1. Start the project and wait until it opens in a normal browser.
2. One-time install (Python 3.9 or newer):
       pip install playwright
       playwright install chromium
3. From the project folder, start the recorder (your own name, no spaces):
       python tester-pack/recorder/record_session.py --tester amina
   Add  --url https://your-address  if the address is not https://xai.localhost
   The first run downloads the accessibility checker (needs internet).
4. Use the whole system: register a bank, approve it as the owner, sign in as the bank, register and approve an
   analyst, scan accounts, load transactions, review alerts, use the anomaly screen, the owner's governing
   console, reminders, identity check. For the "one bank cannot see another's data" test, register and sign in as
   TWO different banks during the session. Use made-up data only.
5. Press "Finish test", write your overview and suggestions, press "Save and finish".
   (If the button is not reachable, press Ctrl+C in the terminal, or close the browser window. The report is still built.)
6. Email to the project owner, subject "XAI test - YOUR NAME - v0.1":
       - the zip file named in the terminal (test-results/xai-test-YOURNAME.zip)
   Inside it is REPORT.md and report.html: the report, the summary of results and the suggestions for improvement.

FOR THE OWNER
1. Put all the zip files you receive into one folder, for example "zips".
2. One-time install:  pip install openpyxl python-docx
3. Run:  python tester-pack/recorder/combine.py zips combined
4. In "combined" you get the tracking workbook and the client report, all numbers filled in. Fill the few [ ] spaces.

LIMITS TO KNOW
- Only calls made with the browser's fetch are measured (the platform's pages use it).
- "Not exercised" means the tester did not use that part. It is not counted as a failure.
- HTTPS/HSTS rules show N/A on a local test address and are enforced on the public domain.
- The video and screenshots can show anything on screen. Use made-up data only.
