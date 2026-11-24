"""Render the demo library's documents and screenshots (all invented content:
the café, airline, band and people do not exist).

    uv run --no-project --with playwright scripts/demo/make_docs.py ~/zetadb-demo/docs [CHROME]

CHROME is an optional Chromium binary; without it Playwright's own is used
(`playwright install chromium`).
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]).expanduser()
CHROME = sys.argv[2] if len(sys.argv) > 2 else None
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
         '&family=Caveat:wght@500;700&family=JetBrains+Mono:wght@400;600'
         '&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap" rel="stylesheet">')
BASE = "*{box-sizing:border-box;margin:0;padding:0} body{font-family:Inter,sans-serif;-webkit-font-smoothing:antialiased}"

DOCS: dict[str, tuple[int, int, str]] = {}

DOCS["receipt_harbor_and_bean_cafe"] = (560, 860, """
<style>body{background:#e9e4da;display:flex;justify-content:center;align-items:center;height:100vh}
.r{width:400px;background:#fffdf8;padding:34px 30px;font-family:'JetBrains Mono',monospace;font-size:14px;color:#222;
box-shadow:0 18px 40px rgba(0,0,0,.18);transform:rotate(-1.5deg)}
h1{font-family:Inter;font-size:22px;text-align:center;letter-spacing:.08em} .c{text-align:center;color:#666;font-size:12px;margin:4px 0 18px}
.row{display:flex;justify-content:space-between;margin:6px 0} hr{border:0;border-top:1px dashed #999;margin:14px 0}
.t{font-weight:600;font-size:17px}</style>
<div class=r><h1>HARBOR &amp; BEAN</h1><div class=c>Café · 214 Pier Street, Portland<br>Tue 14 Oct 2026 · 08:42 · Table 6</div>
<div class=row><span>Oat flat white</span><span>5.20</span></div><div class=row><span>Cardamom bun</span><span>4.75</span></div>
<div class=row><span>Avocado toast</span><span>12.50</span></div><div class=row><span>Cold brew (L)</span><span>5.80</span></div>
<hr><div class=row><span>Subtotal</span><span>28.25</span></div><div class=row><span>Tax 8.5%</span><span>2.40</span></div>
<div class=row><span>Tip</span><span>5.00</span></div><hr><div class="row t"><span>TOTAL</span><span>$35.65</span></div>
<div class=row><span>VISA •••• 4417</span><span>APPROVED</span></div><hr><div class=c>Thank you! Free wifi: harborbean-guest</div></div>""")

DOCS["boarding_pass_sfo_tokyo"] = (1100, 460, """
<style>body{background:linear-gradient(135deg,#dfe7f2,#f4f1ec);display:flex;justify-content:center;align-items:center;height:100vh}
.b{width:960px;height:330px;background:#fff;border-radius:22px;display:flex;overflow:hidden;box-shadow:0 20px 50px rgba(20,40,80,.18)}
.m{flex:1;padding:30px 36px} .s{width:250px;border-left:2px dashed #cfd6e0;padding:30px 26px;background:#f8fafc}
.top{display:flex;justify-content:space-between;color:#1d4ed8;font-weight:700;letter-spacing:.12em;font-size:13px}
.route{display:flex;align-items:center;gap:28px;margin:26px 0} .code{font-size:58px;font-weight:700;color:#0f172a}
.city{color:#64748b;font-size:14px} .plane{flex:1;border-top:2px dotted #94a3b8;position:relative}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px} .k{color:#94a3b8;font-size:11px;letter-spacing:.1em}
.v{font-size:19px;font-weight:600;color:#0f172a;margin-top:3px}</style>
<div class=b><div class=m><div class=top><span>PACIFICA AIR · BOARDING PASS</span><span>ECONOMY PLUS</span></div>
<div class=route><div><div class=code>SFO</div><div class=city>San Francisco</div></div><div class=plane></div>
<div><div class=code>NRT</div><div class=city>Tokyo Narita</div></div></div>
<div class=grid><div><div class=k>PASSENGER</div><div class=v>A. RIVERA</div></div><div><div class=k>FLIGHT</div><div class=v>PA 838</div></div>
<div><div class=k>DATE</div><div class=v>03 APR 2027</div></div><div><div class=k>GATE</div><div class=v>G97</div></div>
<div><div class=k>SEAT</div><div class=v>32A</div></div></div></div>
<div class=s><div class=k>BOARDING</div><div class=v style="font-size:30px">11:05</div><div class=k style="margin-top:16px">GROUP</div>
<div class=v>3</div><div class=k style="margin-top:16px">CONFIRMATION</div><div class=v>K7QX2M</div></div></div>""")

DOCS["whiteboard_pricing_meeting"] = (1280, 820, """
<style>body{background:#8a8f96;display:flex;justify-content:center;align-items:center;height:100vh}
.w{width:1180px;height:720px;background:radial-gradient(circle at 30% 20%,#ffffff,#eef0f1 70%);border:16px solid #c9ccd0;border-radius:8px;
padding:40px 56px;font-family:Caveat,cursive;color:#1f2937;position:relative;box-shadow:inset 0 0 60px rgba(0,0,0,.06)}
h1{font-size:64px;color:#1d4ed8} .cols{display:flex;gap:40px;margin-top:26px} .tier{flex:1;border:4px solid #374151;border-radius:18px;padding:18px 24px;font-size:34px}
.tier b{font-size:44px;display:block} .red{color:#dc2626} .grn{color:#15803d} .note{position:absolute;bottom:44px;left:56px;font-size:36px}
.arrow{color:#dc2626;font-size:40px}</style>
<div class=w><h1>Pricing v2 — Q3 launch</h1><div class=cols>
<div class=tier><b>Starter</b>$9 / mo<br>1 workspace<br>5 GB</div>
<div class=tier style="border-color:#1d4ed8"><b class=grn>Pro ★</b>$24 / mo<br>unlimited projects<br>100 GB · API</div>
<div class=tier><b>Team</b>$18 / seat<br>SSO + audit log<br>min 5 seats</div></div>
<div class=note><span class=red>TODO:</span> annual = 2 months free? &nbsp;<span class=arrow>→</span> ask Priya by Fri 17th<br>
<span class=red>Risk:</span> Starter cannibalises Pro?</div></div>""")

DOCS["chart_monthly_active_users"] = (1200, 720, """
<style>body{background:#0b1020;color:#e5e7eb;padding:46px 56px}
h1{font-size:30px;font-weight:600} .sub{color:#9ca3af;margin:6px 0 30px} .chart{display:flex;align-items:flex-end;gap:22px;height:430px;border-bottom:1px solid #374151;padding:0 10px}
.bar{flex:1;border-radius:10px 10px 0 0;background:linear-gradient(180deg,#818cf8,#4f46e5);position:relative}
.bar span{position:absolute;top:-26px;width:100%;text-align:center;font-size:14px;color:#c7d2fe}
.lab{display:flex;gap:22px;padding:10px 10px 0;color:#9ca3af;font-size:14px} .lab div{flex:1;text-align:center}
.kpi{position:absolute;right:60px;top:46px;text-align:right} .kpi b{font-size:40px;color:#34d399;display:block}</style>
<h1>Monthly active users — 2026</h1><div class=sub>Product analytics · all platforms</div>
<div class=kpi><b>+38%</b>year over year</div>
<div class=chart>""" + "".join(f'<div class=bar style="height:{h}%"><span>{v}k</span></div>' for h, v in
                              zip([42, 45, 49, 52, 58, 61, 66, 71, 74, 80, 88, 96], [21, 23, 25, 26, 29, 31, 33, 36, 37, 40, 44, 48])) +
    '</div><div class=lab>' + "".join(f"<div>{m}</div>" for m in "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()) + "</div>")

DOCS["terminal_connection_refused"] = (1200, 640, """
<style>body{background:#1e1e2e;padding:0;font-family:'JetBrains Mono',monospace;font-size:16px;color:#cdd6f4}
.bar{background:#181825;padding:12px 16px;display:flex;gap:8px;align-items:center} .d{width:13px;height:13px;border-radius:50%}
.t{color:#6c7086;margin-left:14px;font-family:Inter;font-size:14px} pre{padding:22px 26px;line-height:1.6;white-space:pre-wrap}
.p{color:#a6e3a1} .e{color:#f38ba8} .y{color:#f9e2af} .m{color:#6c7086}</style>
<div class=bar><div class=d style="background:#f38ba8"></div><div class=d style="background:#f9e2af"></div><div class=d style="background:#a6e3a1"></div><span class=t>~/projects/orbit — zsh</span></div>
<pre><span class=p>➜ orbit</span> python -m orbit.worker --queue emails
<span class=m>2026-10-21 23:14:07 INFO  starting worker pid=48213</span>
Traceback (most recent call last):
  File "/srv/orbit/worker.py", line 88, in connect
    self.conn = redis.Redis(host=cfg.REDIS_HOST, port=6379).ping()
  File "/usr/lib/python3.12/socket.py", line 852, in create_connection
    raise exceptions[0]
<span class=e>ConnectionRefusedError: [Errno 111] Connection refused</span>
<span class=y>orbit.errors.QueueUnavailable: redis at 10.0.4.12:6379 did not answer after 5 retries</span>
<span class=p>➜ orbit</span> <span class=m>█</span></pre>""")

DOCS["recipe_card_shakshuka"] = (900, 1100, """
<style>body{background:#f3ead8;padding:60px;font-family:Inter;color:#3b2f2f}
.c{background:#fffaf0;border-radius:18px;padding:46px 50px;box-shadow:0 14px 40px rgba(80,50,20,.15)}
h1{font-family:Fraunces,serif;font-size:52px;color:#b4442b} .meta{color:#8a6d5b;margin:8px 0 26px;font-size:16px}
h2{font-family:Fraunces,serif;font-size:24px;margin:22px 0 10px} li{margin:7px 0 7px 22px;font-size:18px;line-height:1.4}</style>
<div class=c><h1>Weekend Shakshuka</h1><div class=meta>Serves 2 · 25 minutes · from Grandma Leila's notebook</div>
<h2>Ingredients</h2><ul><li>4 eggs</li><li>1 can crushed tomatoes (400 g)</li><li>1 red bell pepper, sliced</li><li>1 onion, 3 cloves garlic</li>
<li>1 tsp cumin, 1 tsp smoked paprika, pinch of chilli</li><li>Feta, parsley, crusty bread</li></ul>
<h2>Method</h2><ol><li>Soften onion and pepper in olive oil, 6 min.</li><li>Add garlic and spices, then tomatoes; simmer 10 min.</li>
<li>Make four wells, crack in the eggs, cover for 6 min.</li><li>Crumble feta, scatter parsley, serve with bread.</li></ol></div>""")

DOCS["concert_ticket_jazz"] = (1100, 440, """
<style>body{background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
.t{width:980px;height:320px;border-radius:20px;display:flex;overflow:hidden;background:linear-gradient(120deg,#3b0764,#9d174d 60%,#f59e0b)}
.m{flex:1;padding:34px 40px;color:#fff} .s{width:230px;background:#fff7ed;padding:30px;color:#431407;border-left:3px dashed #fdba74}
h1{font-family:Fraunces,serif;font-size:54px;line-height:1} .k{font-size:12px;letter-spacing:.14em;opacity:.75;margin-top:18px} .v{font-size:22px;font-weight:600}</style>
<div class=t><div class=m><div class=k>LIVE AT THE BLUE LANTERN</div><h1>Midnight Quartet</h1>
<div class=k>DATE</div><div class=v>Saturday 22 Nov 2026 · doors 8:30 pm</div><div class=k>VENUE</div><div class=v>The Blue Lantern, 61 Canal St, New Orleans</div></div>
<div class=s><div class=k>ADMIT ONE</div><div class=v style="font-size:30px">Row C · 14</div><div class=k>PRICE</div><div class=v>$42.00</div>
<div class=k>ORDER</div><div class=v>BL-55102</div></div></div>""")

DOCS["slide_roadmap_2027"] = (1280, 720, """
<style>body{background:#fafafa;padding:64px 80px;color:#111827}
h1{font-size:48px;font-weight:700;letter-spacing:-.02em} .sub{color:#6b7280;font-size:20px;margin:10px 0 44px}
.q{display:grid;grid-template-columns:repeat(4,1fr);gap:22px} .c{background:#fff;border:1px solid #e5e7eb;border-radius:18px;padding:24px;min-height:300px}
.c h3{font-size:15px;color:#6366f1;letter-spacing:.12em} .c p{font-size:19px;margin-top:14px;line-height:1.45}</style>
<h1>Roadmap 2027</h1><div class=sub>Offline-first sync, then the mobile app</div><div class=q>
<div class=c><h3>Q1</h3><p>Offline sync engine<br>Conflict-free merges<br>Beta to 200 users</p></div>
<div class=c><h3>Q2</h3><p>iOS + Android app<br>Share sheet capture<br>Voice notes</p></div>
<div class=c><h3>Q3</h3><p>Team spaces<br>Pricing v2 launch<br>SSO</p></div>
<div class=c><h3>Q4</h3><p>Public API<br>Zapier integration<br>EU data residency</p></div></div>""")

DOCS["code_snippet_debounce"] = (1100, 600, """
<style>body{background:linear-gradient(135deg,#a78bfa,#60a5fa);display:flex;justify-content:center;align-items:center;height:100vh}
.w{width:900px;background:#0d1117;border-radius:16px;box-shadow:0 30px 60px rgba(0,0,0,.35);overflow:hidden}
.bar{padding:14px 18px;display:flex;gap:8px} .d{width:13px;height:13px;border-radius:50%}
pre{padding:8px 30px 30px;font-family:'JetBrains Mono',monospace;font-size:18px;line-height:1.65;color:#e6edf3}
.k{color:#ff7b72} .f{color:#d2a8ff} .s{color:#a5d6ff} .c{color:#8b949e}</style>
<div class=w><div class=bar><div class=d style="background:#ff5f57"></div><div class=d style="background:#febc2e"></div><div class=d style="background:#28c840"></div></div>
<pre><span class=c>// debounce: wait until the user stops typing</span>
<span class=k>export function</span> <span class=f>debounce</span>(fn, ms = <span class=s>250</span>) {
  <span class=k>let</span> timer;
  <span class=k>return</span> (...args) =&gt; {
    clearTimeout(timer);
    timer = setTimeout(() =&gt; fn(...args), ms);
  };
}

<span class=k>const</span> search = <span class=f>debounce</span>(q =&gt; api.<span class=f>search</span>(q), <span class=s>300</span>);</pre></div>""")

DOCS["handwritten_grocery_list"] = (800, 1000, """
<style>body{background:#d6c7a8;display:flex;justify-content:center;align-items:center;height:100vh}
.p{width:560px;height:820px;background:repeating-linear-gradient(#fffef6 0 46px,#bcd4f0 46px 48px);padding:70px 50px 40px 90px;position:relative;
font-family:Caveat,cursive;font-size:40px;line-height:48px;color:#1e3a8a;transform:rotate(1.2deg);box-shadow:0 16px 36px rgba(60,40,10,.25)}
.p:before{content:'';position:absolute;left:66px;top:0;bottom:0;border-left:2px solid #f87171} s{color:#64748b}</style>
<div class=p><b>Saturday market</b><br>sourdough loaf<br>2 lemons + basil<br><s>oat milk</s><br>heirloom tomatoes<br>feta (the good one)<br>
coffee beans — Ethiopia<br>flowers for Mom 🌷<br>batteries AA ×4</div>""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        for name, (w, h, body) in DOCS.items():
            page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.set_content(f"<!doctype html><meta charset=utf-8>{FONTS}<style>{BASE}</style>{body}")
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            page.screenshot(path=str(OUT / f"{name}.png"))
            page.close()
            print("rendered", name)
        b.close()


if __name__ == "__main__":
    main()
