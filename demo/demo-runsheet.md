# Demo run sheet — glance at this DURING the call

Full prep: [`demo-prep.md`](demo-prep.md). This is the cue card only.

## 30-second pitch (say this)
> "This is a working example of exactly what we're here to discuss: an **MCP server** over NOAA
> science data, and an **AI agent driving it in plain English** — finds the dataset, subsets in
> space/time, pulls variables by plain-English name, runs domain analyses, no code. I built it
> over the NNJA *observational* archive, but **the architecture is the transferable part: swap
> in forecast tools and this is an agent talking to a nested-EAGLE MCP server** — and one agent
> can talk to both at once. Early proof of concept, but it runs today."

## Show these 3 things (in order, stop whenever time runs out)
1. **`--members`** = the **server side** → "the tool contract an agent sees; for nested-EAGLE it'd be forecast tools." (one screen)
2. **`--chat` discovery query** = the **agent side** → English in, real data + summary out. "This agent piece is what you'd build."
3. **Hero analysis** (wildfire risk OR lapse rate) → domain product from one sentence. Then the **mapping line**: "same pattern → nested-EAGLE; one agent spans both servers."

> **As every query runs, point at the `[tool] name(...)` / `↳ done in N.Ns` lines** — "that's the agent calling the server's tools live." Best visual proof of the pattern; don't let it scroll by unremarked.

## Commands (PASTE — don't type)
```
# already running before the call:  $env:MCP_TRANSPORT="http"; uv run server.py   (wait for "DataCatalog ready")
uv run mcp-client --members http://localhost:8000/mcp
uv run mcp-client --chat http://localhost:8000/mcp
```

## Queries — ✅ VERIFIED against the live server (timings noted). Paste in order.
```
1. What NNJA datasets are available?
   -> lists 14 datasets. Instant. (Ask this FIRST so the model learns the exact IDs.)

2. Summarize air temperature and sea-level pressure from the adpsfc surface
   observations for 2021-01-01 over the continental US (24 to 50 N, 125 to 66 W).
   -> ~0.5s. (temps come back in KELVIN; pressure is mean-sea-level)

3. HERO -- Assess wildfire risk from the SEVIRI satellite dataset for 2023-07-01.
   -> ~4s. Returns a risk-category distribution (No/Low/Medium risk).
   ⚠️ SEVIRI has NO data before 2022-07-01. Use 2023-07-01. Do NOT say 2021 with SEVIRI.

4. Now classify the cloud phases from the same SEVIRI data using the
   cloud-cooling index for 2023-07-01.
   -> ~1.5s. 6 cloud categories (clear / warm water / cirrus / mixed / deep convective / supercooled). Great 2nd beat.

5. (obs alt) Calculate the atmospheric lapse rate between 1000 and 500 hPa
   from the radiosonde data for 2021-01-01.
   -> ~2s. Stability distribution, dominant "Stable", mean ~4.5 K/km.
```
**Ground truth** (if the model's NL translation misfires, these exact tool calls work):
- `calculate_spectral_index(dataset="SEVIRI", index_name="wildfire_risk" | "cloud_cooling", time="2023-07-01")`
- `calculate_lapse_rate(time="2021-01-01", level1_hpa=1000, level2_hpa=500)` — takes NO dataset arg
- `descriptive_stats_dataset(dataset="adpsfc", variables=["temperature","pressure"], time="2021-01-01", lat_bounds=[24,50], lon_bounds=[-125,-66])`

**Rules:** English variable names + science tools only. **NO channel-by-number** ("brightness
temp 2" — mis-resolves, B4). **SEVIRI date ≥ 2022-07-01** (use 2023-07-01). GOES/Himawari have
2020-01-01+ data but are **SLOW live (15–25s)** — avoid on screen. Surface dataset name:
"adpsfc" or "conv-adpsfc" resolves; "surface synoptic observations" does NOT.

## If it breaks
- Query errors / Gemini flakes → paste **query #2** (simpler). Still broken → **play the recording.**
- Server dead / no response → **play the recording**, narrate it. Don't debug live.
- ~500 MB prompt pops up → decline & rephrase smaller, or narrate it as a safety feature.
- Cut short → skip to the **hero query**, or just say the 30-sec pitch + the ask.

## Top questions → one-line answers
- *"What would a nested-EAGLE MCP server expose?"* → Same shape, forecast tools: run a forecast, fetch fields, ensemble spread, verification metrics. Agent code unchanged. (Say "*something like this* — you know the internals.")
- *"How much work to build one?"* → Essentially one file of tool defs over a data library; tractable. Hard part is domain design, not MCP plumbing. (Don't quote a timeline.)
- *"Can one agent use multiple MCP servers?"* → Yes — an agent can pull **obs from NNJA + forecasts from nested-EAGLE** in one conversation. That's the vision.
- *"Why build our own vs Microsoft's MPC MCP / Earth Copilot?"* → Domain fit + control (your models, obs/tabular data MPC serves poorly — Parquet only *preview* there). They prove the pattern; you own the domain. Can coexist.
- *"Does the LLM make up numbers?"* → No — tools return **real archive values**; analyses are deterministic. (Caveat: loose variable names can occasionally mis-resolve.)
- *"Production-ready?"* → It's a PoC; a real deploy adds auth behind Azure.
- *Anything about nested-EAGLE internals* → "I'd confirm that with [boss]." **Don't bluff.**

## Don't forget
- Punchline first (assume short slot). Narrate while queries run.
- **Point at the `[tool]` lines** as they appear — visible agent↔server calls are the whole pitch.
- It's about the **pattern (server + agents)**, not the NNJA data — always land the nested-EAGLE mapping.
- Say **"Azure / Microsoft cloud"**, not "Planetary Computer" (unless boss confirmed).
- Confident about the demo; humble/curious about their project & model.
- **End with the ask** your boss wants (feasibility / build proposal / collaboration / next step).
```
