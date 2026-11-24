(() => {
  if (window.top !== window || window.__xaiBadge) return;
  window.__xaiBadge = true;
  const boot = () => {
    if (!document.body) return setTimeout(boot, 50);
    const host = document.createElement("div");
    host.id = "xai-root";
    host.style.cssText = "position:fixed;left:10px;bottom:10px;z-index:2147483647;";
    document.body.appendChild(host);
    const root = host.attachShadow({ mode: "open" });
    const css = `*{box-sizing:border-box;font-family:"Times New Roman",serif;font-size:13px;color:#000}
      .pill{display:flex;gap:8px;align-items:center;background:#fff;border:1px solid #000;padding:3px 8px;opacity:.9}
      button{border:1px solid #000;background:#fff;padding:3px 8px;cursor:pointer}
      .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#000}
      .wrap{position:fixed;inset:0;background:rgba(255,255,255,.96);overflow:auto;padding:24px}
      .card{max-width:620px;margin:0 auto;border:2px solid #000;background:#fff;padding:18px}
      h2{margin:0 0 8px;font-size:20px} p{margin:6px 0;line-height:1.35;font-size:14px}
      textarea,select{width:100%;border:1px solid #000;padding:6px;margin:4px 0 10px;font-size:14px}
      textarea{height:90px}label{display:block;font-size:14px;font-weight:bold;margin-top:6px}
      .row{display:flex;gap:8px;margin-top:8px}.row button{padding:7px 14px;font-size:14px}`;
    let open = false;
    const draw = () => {
      root.innerHTML = "<style>" + css + "</style>";
      const pill = document.createElement("div");
      pill.className = "pill";
      pill.innerHTML = '<span class="dot"></span>Recording <button id="fin">Finish test</button>';
      root.appendChild(pill);
      pill.querySelector("#fin").onclick = () => { open = true; draw(); };
      if (open) {
        const w = document.createElement("div"); w.className = "wrap";
        w.innerHTML = `<div class="card"><h2>Your overall overview</h2>
          <p>The technical results were measured automatically while you used the system. This last page is only for your own opinion. It is kept separately and is not part of the measured results.</p>
          <label>Did you complete the setup without anyone's help?</label>
          <select id="help"><option value="">choose</option><option>Yes</option><option>No</option></select>
          <label>Overall impression, 1 (poor) to 5 (excellent)</label>
          <select id="score"><option value="">choose</option>${[1,2,3,4,5].map(n=>`<option>${n}</option>`).join("")}</select>
          <label>Overall opinion: what stood out, good or bad?</label><textarea id="op"></textarea>
          <label>Suggestions for improvement</label><textarea id="sg"></textarea>
          <div class="row"><button id="save">Save and finish</button><button id="back">Go back to testing</button></div>
          <p id="msg"></p></div>`;
        root.appendChild(w);
        w.querySelector("#back").onclick = () => { open = false; draw(); };
        w.querySelector("#save").onclick = async () => {
          await window.xai_finish({ help: w.querySelector("#help").value, score: w.querySelector("#score").value,
            opinion: w.querySelector("#op").value, suggestions: w.querySelector("#sg").value });
          w.querySelector("#msg").textContent = "Saved. Your report is being built. You can close this window.";
        };
      }
    };
    draw();
  };
  boot();
})();
