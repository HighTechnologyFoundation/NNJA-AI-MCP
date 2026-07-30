# Demo prep — NNJA-AI-MCP

Working prep doc for the upcoming demo. Companion cue card: [`demo-runsheet.md`](demo-runsheet.md)
(the one-screen thing to keep visible during the call). This file is the full prep; the
runsheet is what you glance at live.

> **The situation:** Microsoft Teams call, you present via screen share. Other business
> happens first; your **boss introduces you** and hands you a slot of **uncertain, possibly
> short length**. Audience: **NOAA employees/affiliates on the nested-EAGLE project** —
> technical vs non-technical mix **unknown**.

> **THE POINT OF THE MEETING (why you're really there):** to explore **expanding this work to
> build AI agents that interact with an eventual nested-EAGLE MCP server.** So your NNJA-AI-MCP
> project is the **proof of concept / template** — living evidence that you can wrap NOAA
> science data/systems in an MCP server and have LLM agents drive it in natural language. The
> demo's job is to make that pattern *feel real, feasible, and worth expanding* to nested-EAGLE.
> **The transferable thing is the ARCHITECTURE (MCP server + agents), not the NNJA data itself.**

> **Who they actually are (researched):** nested-EAGLE = NOAA/OAR's *Experimental AI Global
> and Limited-area Ensemble* forecast system (EPIC + PSL + GSL + EMC) — an **AI/ML weather
> forecasting** effort (Anemoi/GraphCast-based), fuelled by NOAA obs/reanalysis training data,
> with a roadmap onto **Microsoft Azure** (NOAA–Microsoft CRADA). So: earth-science / ML
> forecasting people. A *nested-EAGLE MCP server* would expose forecast/model things (run a
> forecast, fetch output fields, ensemble stats, verification metrics) — the same **shape** as
> your obs tools, different domain. Your NNJA server shows the pattern; they'd apply it to their
> model.

---

## 0. Confirm with your boss BEFORE the call (highest leverage)

You're guessing at several things that change the whole framing. A 2-minute check with your
boss removes the guesswork:

- [ ] **How long is my slot, realistically?** (Prepare for the *shortest* plausible — see the tiered script in §4.)
- [ ] **Technical or non-technical audience?** (Determines vocabulary — see §5.)
- [ ] **What does my boss want this demo to *achieve*?** Given the meeting's purpose — is the ask "prove this is feasible," "propose we build a nested-EAGLE MCP server + agents," "explore a collaboration," or just visibility? Tailor the closing line to that.
- [ ] **Does an "eventual nested-EAGLE MCP server" already have a design/owner, or is this greenfield?** Changes whether you pitch "here's the template to copy" vs "here's how agents would plug into what you're planning." Don't assume the server exists — the word was "eventual."
- [ ] **Am I the one proposing to build the nested-EAGLE side, or just showing what's possible?** Know your lane before you offer to do work.
- [ ] **"Planetary Computer" vs just "Azure"?** Public evidence ties nested-EAGLE to **Azure**, *not* the Planetary Computer product specifically. Don't say "Planetary Computer" on the call unless your boss confirms it — say "Azure / Microsoft cloud." (If they *are* targeting Planetary Computer Pro, see the §6 note — Microsoft already has MCP tooling there.)
- [ ] **Is the NNJA archive actually part of nested-EAGLE's training pipeline?** If **yes**, that's your single strongest line ("the obs archive that feeds your training is what this makes queryable in plain English"). If unconfirmed, phrase it as "the *same class* of observational data."
- [ ] **Am I sharing my own screen, or is someone driving?** (Confirm you'll have share permissions in their tenant — external guests are sometimes blocked from sharing in Teams.)

---

## 1. Technical readiness — do the day before AND ~30 min before

**The #1 rule: the server must be already running and its catalog already loaded before you
share your screen.** The catalog load takes several seconds (~3.6s observed) and logs to
stderr; you do **not** want a cold start, a spinner, or a stack trace on screen.

Day before:
- [ ] **Rebuild + verify the container** per `HANDOFF.md` (carries the merged server fixes B3/B6/MSLP/D1). Run `testing/integration_all.py` and `testing/integration_aliases.py` against it — green.
- [ ] **Confirm `GEMINI_API_KEY` is set** and the model is valid (`verify_model` fails fast on a bad model — you want that failure *now*, not live).
- [ ] **Dry-run every query you plan to show, end to end**, against the actual server you'll use. Confirm each returns sensible output *and resolves to the right variables* (see the B4 caveat in §4). Save the good outputs.
- [ ] **Capture a fallback recording / screenshots** of a full successful `--chat` run. If the network or Gemini flakes live, you show the recording and narrate it. This single item de-risks the whole demo.

~30 min before:
- [ ] **Start the server persistently in HTTP mode** and leave it up:
  `MCP_TRANSPORT=http uv run server.py` (PowerShell: `$env:MCP_TRANSPORT="http"; uv run server.py`) — or the Docker container. Wait for `DataCatalog ready in N seconds.`
- [ ] **Connect the client once and run one warm-up query** so caches are warm and you've confirmed connectivity: `uv run mcp-client --chat http://localhost:8000/mcp`.
- [ ] Leave that chat session open and ready, OR have the exact connect command ready to paste.

---

## 2. Screen-share & Teams logistics

- [ ] **Bump the terminal font way up** (and the browser zoom if you show one). Teams compresses shared video; small monospace text turns to mush. Aim for something readable on a laptop from across a room. Test by sharing to yourself.
- [ ] **Use a high-contrast/light terminal theme.** Dark-grey-on-black reads terribly over compressed screen share.
- [ ] **Share the specific window, not the whole screen.** Protects you from notification leaks and stray tabs, and keeps focus. (In Teams: "Window" not "Screen.")
- [ ] **Kill notifications & secrets:** close Slack/Teams chat popups, email, calendar toasts; turn on Focus/Do-Not-Disturb; make sure no `.env`, API keys, tokens, or unrelated client work are visible in the shared window or terminal scrollback. Clear the terminal before sharing (`clear` / Ctrl-L).
- [ ] **Don't type queries live — paste them.** Keep your queries in a scratch file (or the runsheet) and paste each one. Live typing = typos, autocomplete surprises, and dead air. Pasting also lets you keep queries phrased exactly as you tested them.
- [ ] **Second monitor for your notes/runsheet** (not shared), if you have one. If not, keep the runsheet on your phone.
- [ ] **Network:** wired if possible; close bandwidth hogs (cloud sync, downloads, other video). A dropped call mid-query is why the §1 recording exists.
- [ ] **Join early**, test share + audio in the actual meeting (or a test meeting in their tenant if you can), and then wait. Since other business runs first, be ready to un-mute and share on ~10 seconds' notice when your boss hands over.

---

## 3. The narrative / positioning (what to say)

**Lead with the PATTERN, demonstrated live, then map it to nested-EAGLE.** The demo is the
evidence; the point is "this shape works and you can build it for your model." Use NNJA as the
concrete, working example — don't let the obs domain become the whole story.

**30-second pitch (adapt to what your boss confirmed):**
> "This is a working example of the pattern you're here to talk about: an **MCP server** that
> wraps NOAA science data, and **AI agents that drive it in plain English** — find the right
> dataset, subset it in space and time, pull variables by plain-English name, run domain
> analyses, no code and no memorizing IDs. I built it over the NNJA *observational* archive,
> but the architecture is the transferable part: **swap in forecast tools and this is what an
> agent talking to a nested-EAGLE MCP server looks like** — and one agent can talk to both at
> once. It's an early proof of concept, but it's real and it runs today."

**The three beats that matter for THIS audience:**
1. **The pattern is real and runs today.** MCP server (tools/resources/prompts) + an LLM agent (the chat client) that consumes them. You'll show both sides — the server's capabilities *and* an agent using them — because **agents are what they want to build.** And you can *see* it happen: the CLI now prints each tool call live as `[tool] name(args)` with its completion time (`↳ done in N.Ns`), so the audience literally watches the agent invoke the server's tools and get data back. **That visible round-trip is your single strongest piece of evidence for "an agent driving an MCP server" — point at it when it scrolls by.**
2. **It maps directly to nested-EAGLE.** Your `calculate_*` / subsetting / aliasing tools are the obs-domain instance of a general shape; a nested-EAGLE MCP server would expose *forecast* tools (run/fetch/verify) the same way, and the same agent code drives it. **Multi-server composability is the forward hook:** an agent that pulls obs from NNJA *and* forecasts from a nested-EAGLE server in one conversation.
3. **You can own it and fit it to your domain.** It's **cloud-neutral** (GCS Parquet today; a container that deploys to Azure), **domain-specific** (English-name aliasing, subsetting, built-in science analyses), and self-built — not dependent on a raster-first platform's model. (See §6 for the "why not just use Microsoft's MPC MCP tools" question — answer it head-on.)

**Be honest that it's a prototype/PoC.** Don't oversell it as production. "This is an early
proof of concept" buys credibility and makes the working demo *more* impressive, and it fits
the meeting's exploratory purpose (feasibility, not a finished product).

---

## 4. The demo script — tiered by time (rehearse all three)

Because your slot length is uncertain, have three versions ready and pick on the fly. **Lead
with the punchline** in every version — assume you could be cut off.

**✅ The exact queries are verified against the live server — see the runsheet.** They return
real data, fast. Timings and the ground-truth tool calls are in `demo-runsheet.md`. The
findings that drove those choices:
- **Use SEVIRI + `2023-07-01` for the wildfire/cloud-phase hero** (~2–4s). **SEVIRI has NO data before 2022-07-01** — the obvious "2021-01-01" returns *nothing* for SEVIRI, which would be an ugly empty result on screen.
- **GOES ABI / Himawari** cover 2020-01-01+ (so they'd work for 2021), but each spectral query takes **15–25s** — too slow for a live screen. Avoid them live; SEVIRI is 5–8× faster.
- **Obs side** (surface stats, lapse rate) works great for **`2021-01-01`** (~0.5–4s).
- **Surface dataset name:** "adpsfc" / "conv-adpsfc" resolves; "surface synoptic observations" does *not* — so **ask "what datasets are available?" first** and let the model pick up the real IDs.

**⚠️ Query-safety caveat (still applies):** the fuzzy variable matcher can mis-resolve a
*channel-by-number* request like "brightness temperature 2" (the deferred **B4** issue; the
project's own `client.py` example uses that risky phrasing). **Avoid channel-by-number live.**
Lean on **aliased plain-English names** (`temperature`, `pressure`) and **the science tools**
(`calculate_spectral_index`, `calculate_lapse_rate`), which use exact hard-coded channel IDs
internally and can't mis-resolve. Keep regions/dates modest so nothing trips the ~500 MB prompt.

---

### 2-minute version (assume you're rushed)
One query that shows the whole magic: an agent driving an MCP server — NL in → real NOAA data + a domain analysis out.
1. `uv run mcp-client --chat http://localhost:8000/mcp`
2. Paste ONE hero query, e.g. a wildfire-risk or lapse-rate analysis over a small region/date (see runsheet for exact wording, pre-tested).
3. **As the `[tool]` line appears, point at it:** "see that — the agent is calling the server's tool live, and there's the round-trip time." Then narrate the result: "it picked the dataset, subset it to my region and date, resolved the variables, and computed a domain index — all from that English sentence." Then **land the point in one line**: "That's an AI agent talking to an MCP server over NOAA data — the same pattern would let agents drive a nested-EAGLE MCP server." Done.

### 5-minute version
1. **(20s) `--members`** — one screen showing the **server side**: the tool/resource catalog an MCP server exposes. "This is the contract an agent sees — for nested-EAGLE it'd be forecast tools instead of these obs ones." (Also good for non-technical: it looks capable without explanation.)
2. **(90s) The agent side, discovery query** — switch to `--chat`: "What datasets are available?" then "Summarize surface temperature and pressure over [small region] on [date]." **Point at the `[tool] name(...)` lines as they scroll — "that's the agent calling the server's tools live, with the timing."** Show the agent resolving English names → real data → a summary. Emphasize: this — an agent driving the server — is exactly the piece they want to build.
3. **(2min) The hero analysis** — the wildfire-risk or lapse-rate query. The "wow": a domain-specific derived product from one plain-English request.
4. **(30s) Land the framing + the mapping** — the §3 pitch: server + agents, transferable to nested-EAGLE, agents can span both servers. Close with your boss's intended ask.

### 10-minute version (if you get the room)
- All of the 5-min version, plus:
- **A second analysis** (e.g. lapse-rate stability on radiosonde data, or `compare_datasets` across two obs sources) to show it's not a one-trick query.
- **One "under the hood" beat for a technical crowd** — show the tool call / variable resolution, or briefly open `--members` and note the English-alias layer and subsetting params. Tie it to build effort: "the server is essentially one file of tool definitions — a nested-EAGLE equivalent is tractable." Keep it short; don't rathole.
- **The nested-EAGLE mapping, explicitly** — "these obs tools become forecast tools; the agent code is unchanged; and one agent can hold a conversation across *both* an NNJA server and a nested-EAGLE server." This is the vision beat — the reason they're in the room.
- Invite a **live question-as-query** from the audience ("what would you want to ask it?") — high-impact *only if* you're confident; risky if the phrasing trips B4/large loads. Have a graceful "let me rephrase that for it" ready.

---

## 5. Audience adaptation (flip based on §0 answer, or read the room)

**If technical (likely, given EPIC/PSL ML engineers):**
- Name the pieces: MCP, tool schemas exposed to the LLM, the fuzzy + alias variable resolution, spatial/temporal subsetting, the science tools.
- Emphasize **extensibility** (adding datasets/tools), **cloud-neutrality** (Azure-deployable), and how it complements STAC/raster stacks rather than duplicating them.
- Be ready to concede limits crisply (PoC, fuzzy-resolution edge cases, no auth yet).

**If non-technical:**
- No jargon. "You ask a question in plain English; it fetches the actual NOAA data and does the analysis for you — no code, no knowing the exact variable names."
- Emphasize **time saved** and **accessibility** (a scientist or a manager can both use it).
- The `--members` screen and a clean summarized answer sell it without explanation.

**Mixed / unknown:** default to plain-English framing, then offer *one* "for the engineers in
the room…" aside. Watch faces on the call; if they nod at detail, go deeper.

---

## 6. Q&A prep — anticipated questions + honest answers

- **"Doesn't Microsoft's Planetary Computer already have MCP tools / Earth Copilot for this?"**
  Yes — and answer it head-on. Microsoft ships an MCP server for Planetary Computer Pro and an
  open-source MPC↔Claude reference app. But those are **raster/STAC-first** (satellite imagery,
  gridded products). This is **domain-specific to point/atmospheric observations** — surface
  obs, radiosondes, satellite radiances as *tabular* data — which those platforms serve poorly
  (NWP grids on Azure aren't even STAC-indexed; the one NOAA point-obs set is blob-only). Same
  *idea* (NL over data via MCP), different, underserved *data niche*.
- **"Can it run on Azure / the Planetary Computer?"**
  It's cloud-neutral — data's on GCS Parquet today, and the server is a container; it deploys
  anywhere. Note (if it comes up): Parquet/tabular is only a *preview* format on Planetary
  Computer Pro, which is another reason a dedicated obs tool has a place.
- **"Does the LLM hallucinate / how do I trust the numbers?"**
  The LLM doesn't invent data — it calls tools that return **real values from the archive**;
  the analyses (lapse rate, spectral indices) are deterministic NumPy on the raw data. Honest
  caveat: *variable resolution* from loose names can occasionally pick the wrong channel (a
  known limitation we're aware of), which is why precision analyses use exact channels.
- **"What datasets / variables?"**
  NNJA observational archive — surface synoptic/METAR, radiosonde/upper-air, microwave
  sounders (ATMS/AMSU), and geostationary IR imagers (SEVIRI, GOES ABI, Himawari AHI).
- **"Is it production-ready? Security?"**
  It's a proof of concept. Honest: today it binds locally without auth (fine for a local
  demo); a real deployment would sit behind auth / a reverse proxy on Azure.
- **"What would a nested-EAGLE MCP server actually expose?"** (the core question of the meeting)
  Map it to what you're showing: instead of obs tools, forecast/model tools — e.g. "run a
  forecast for this domain/time," "fetch these output fields," "give ensemble spread," "return
  verification metrics vs analysis," "compare two model runs." Same tool/resource shape as your
  `calculate_*` and subsetting tools; the agent doesn't care whether the data behind a tool is
  obs or forecast. Be honest you don't know their internals — frame it as "*something like this*."
- **"How much work is it to build one?"**
  Honest and encouraging: the server is essentially one file of tool definitions over a data
  library — tractable for a focused effort. The hard parts are domain design (which tools,
  which parameters) and the data access, not the MCP plumbing. Don't quote a timeline you can't back.
- **"Can one agent talk to more than one MCP server?"**
  Yes — that's a big part of the appeal. An agent can connect to multiple MCP servers at once,
  so a single conversation could pull **obs from an NNJA server and forecasts from a
  nested-EAGLE server** and reason across both. That composability is the forward vision.
- **"Why build our own MCP server vs use Microsoft's MPC MCP tools / Earth Copilot?"**
  Not either/or — but you'd build your own for **domain fit and control**: your models, your
  data shapes (obs/tabular, which MPC serves poorly — Parquet is only preview there), your
  deployment. Microsoft's tools prove the *pattern* works at scale; a nested-EAGLE server owns
  the *domain*. They can coexist (an agent can use both).
- **"How does this connect to nested-EAGLE's data?"**
  It exposes the *same class* of NOAA observational data that feeds ML training. (Strengthen to
  "the exact archive that feeds your training" only if your boss confirms NNJA is in the pipeline.)
- **Anything about their project internals you don't know:** "Great question — I'd want to
  check with [boss] before answering that precisely." **Don't bluff about nested-EAGLE
  internals** to a room that knows them far better than you. Curiosity reads better than a guess.

---

## 7. Contingencies — when something breaks

- **Server not responding / catalog not loaded:** you pre-started it (§1). If it's dead, fall
  back to the **recording** immediately — don't debug live. ("Let me show you a run I captured.")
- **Live query errors / Gemini rate-limit or timeout:** have a **second, simpler pre-tested
  query** ready; if that also fails, go to the recording. Never stare at a stack trace.
- **The ~500 MB confirmation prompt appears:** it asks y/N before a big load. Either decline
  and rephrase smaller, or — if you planned for it — narrate it as a safety feature ("it's
  warning me this is a large pull").
- **Network drops:** recording. This is the whole reason it exists.
- **Cut short by time:** jump straight to the 2-min hero query, or just narrate the recording.
  Have the one-sentence pitch + ask memorized so you land the point even with zero live demo.
- **Awkward silence / other business runs long:** stay muted and ready; don't interrupt. When
  handed over, open with a crisp one-liner, not "can you see my screen?" (confirm sharing
  *before* you're introduced if you can).

---

## 8. Delivery reminders

- **Assume the shortest slot.** Punchline first, details if time allows.
- **Narrate what's happening** as queries run — and let the live `[tool]` lines carry it
  ("there it goes, calling the server… back in 3 seconds with data"). They double as a
  progress indicator, so a multi-second tool call reads as *working*, not frozen — and they
  put the agent↔server interaction on screen, which is the whole point.
- **You know this tool better than anyone on the call; you do NOT know nested-EAGLE better
  than they do.** Be confident about the demo, humble and curious about their world. Asking
  "would this be useful for how you build training sets?" is better than claiming it is.
- **End with the ask** your boss wants (feedback / collaboration / next step), not just "…and
  that's it."
