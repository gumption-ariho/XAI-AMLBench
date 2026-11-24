# Testing XAI-AMLBench

You will run your own copy of the platform, use it normally in a recording browser, and email back one file. You do
not mark anything pass or fail. While you work, the recorder measures the system against published standards
(speed, accessibility, security, data-protection basics) and records which parts you used. At the end you write an
overall opinion and any suggestions.

Use made-up banks, people and accounts only. No real customer data.

## What you need
- Docker with Compose, about 8 GB of free memory, 15 GB of free disk
- Python 3.9 or newer
- About 2 hours, including the first start

## 1. Get your own copy
On the project's GitHub page press **Fork**, then:

    git clone https://github.com/YOUR-USERNAME/XAI-AMLBench.git
    cd XAI-AMLBench

## 2. Start the platform
    cp .env.example .env
Open `.env` and replace every `change_me` value with any text you like (remember the `ADMIN_PASSKEY`, you need it
at `/owner`). Then:

    docker compose --profile app --profile ml --profile obs up -d --build

The first start takes 10 to 20 minutes. When it finishes, open https://xai.localhost in your browser. A certificate
warning is expected on a local test address. If you see a plain "404 page not found", wait two minutes and refresh.

The written-explanation service needs an NVIDIA graphics card, so on most computers it will show as unreachable on
the Connect tab. That is expected. Everything else works without it.

## 3. Install the recorder (once)
    python3 -m venv ~/xai-recorder-env
    source ~/xai-recorder-env/bin/activate
    pip install playwright
    playwright install chromium

## 4. Start recording (use your own name, no spaces)
    python tester-pack/recorder/record_session.py --tester amina

The first run downloads the accessibility checker, so it needs internet. A browser window opens. A small
"Recording" tag sits at the bottom left of the page.

## 5. Use the whole system
Work through it in this order. Nothing needs marking.

1. Open `/request-access` and submit a made-up bank.
2. Open `/owner`, sign in with your `ADMIN_PASSKEY`, approve the bank, and note the passkey.
3. Sign in as the bank on the main page with that passkey.
4. Open `/analyst-signup`, register a made-up analyst. Try `/signin` before approval (it should refuse), then
   approve the analyst at `/owner` and sign in.
5. Generate the sample data, then in the **Intake** tab paste or upload `sample_transactions.csv`:

       python tester-pack/sample-data/make_sample.py

6. Scan a few accounts, open some alerts, confirm one and dismiss one, open **Investigate** and press Agree on a
   likely-normal anomaly, open **Network** and **Activity**.
7. Open `/?view=governance` as the owner, press **Alert bank now**, then sign in as the bank and acknowledge the
   reminder.
8. **Register and sign in as a second bank** and confirm it sees none of the first bank's data. This is how the
   recorder tests that banks are kept apart.
9. Try the identity check, and the two-step sign-in at `/security` if you like.

## 6. Finish
Press **Finish test** at the bottom left of the page. Say whether setup needed help, give an overall impression
from 1 to 5, and write your overall opinion and suggestions. Press **Save and finish**.

If the button is out of reach, press **Ctrl+C** in the terminal, or close the browser window. The report is still
built.

## 7. Send your results
Email the zip file the terminal prints (`test-results/xai-test-YOURNAME.zip`) to the project owner with the subject
`XAI test - YOUR NAME - v0.1`. Inside is a readable report with a summary of results and suggestions for
improvement, plus screenshots and a video of your session.

## What is recorded
The recorder stores pages you visited, buttons you pressed, which fields you used (never what you typed, never
passwords), server response times and errors, screenshots, a video, and your computer and Docker details. The video
and screenshots can show anything on screen, which is why only made-up data should be used.

## For the project owner
Put all received zip files in one folder, then:

    pip install openpyxl python-docx
    python tester-pack/recorder/combine.py zips combined

That produces the tracking workbook and the client report with the numbers filled in.
