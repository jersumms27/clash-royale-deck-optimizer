/* =====================================================================
   Clash Royale Deck Optimizer — front-end logic
   Talks to the local server (server.py). No frameworks, no build step.

   Fitness comes in two flavours, told apart by `kind`:
     "winrate" — the learned matchup model: expected win rate in [0, 1],
                 shown as a percentage ("58.3%")
     "score"   — the hand-tuned heuristic, shown as a plain number ("0.5830")
   Every place a fitness is printed goes through fmtFit() so the page adapts
   to whichever scorer is selected.
   ===================================================================== */

"use strict";

const ART_BASE = "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards";

// A few names whose CDN slug differs from the simple rule below.
const ART_OVERRIDES = {
  "P.E.K.K.A": "pekka",
  "Mini P.E.K.K.A": "mini-pekka",
  "X-Bow": "x-bow",
};

const RARITY_ORDER = { common: 0, rare: 1, epic: 2, legendary: 3, champion: 4 };

// nice human labels for the card-attribute columns
const STAT_LABELS = {
  hitpoints: "Hitpoints",
  damage: "Damage",
  damage_per_second: "Damage / sec",
  attack_period: "Hit speed (s)",
  range: "Range",
  radius: "Radius",
  lifetime: "Lifetime (s)",
  crown_tower_damage: "Tower damage",
  special_damage: "Special damage",
};

const SCORER_ICON = { learned: "🧠", heuristic: "📐" };
const LS_SCORER = "cr.scorer";
const LS_BUILDER = "cr.builder";

const state = {
  cards: [],
  byId: new Map(),
  config: null,
  scorers: [],
  scorerId: null,          // the scorer picked for optimizer runs
  scorerInfo: {},          // scorer id -> info() dict (or {error})
  eventSource: null,
  run: { kind: "score", scorerId: null, total: 0, lastGen: 0, lastDeck: null },
  best: null,              // final best deck payload of the last run
  builder: { a: emptyDeck(), b: emptyDeck() },
  meta: { scorerId: null, decks: null, error: null, loading: false },
  h2h: { timer: null, seq: 0 },
  picker: { side: null, index: -1 },
};

/* ----------------------------- helpers ------------------------------- */
function $(sel) { return document.querySelector(sel); }
function el(tag, cls) { const n = document.createElement(tag); if (cls) n.className = cls; return n; }

function artUrl(name) {
  const slug = ART_OVERRIDES[name] || name
    .toLowerCase()
    .replace(/\./g, "")
    .replace(/'/g, "")
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${ART_BASE}/${slug}.png`;
}

function initials(name) {
  const words = name.replace(/[.']/g, "").split(/[\s-]+/).filter(Boolean);
  return (words.length >= 2 ? words[0][0] + words[1][0] : name.slice(0, 2)).toUpperCase();
}

function isNum(x) { return typeof x === "number" && isFinite(x); }
function pct(x, digits = 1) { return isNum(x) ? (x * 100).toFixed(digits) + "%" : "—"; }

// One formatter for every fitness on the page. `kind` defaults to the
// current run's scorer so old code paths keep working.
function fmtFit(x, kind) {
  if (!isNum(x)) return "—";
  return (kind || state.run.kind) === "winrate" ? pct(x) : x.toFixed(4);
}
function fitLabel(kind) {
  return (kind || state.run.kind) === "winrate" ? "exp. win rate" : "fitness";
}

function scorerById(id) { return state.scorers.find((s) => s.id === id) || null; }
function currentScorer() { return scorerById(state.scorerId); }

// The scorer that answers P(A beats B): the selected one if it can, else any
// available one that can (the heuristic never can).
function predictScorer() {
  const sel = currentScorer();
  if (sel && sel.available && sel.supports.predict) return sel;
  return state.scorers.find((s) => s.available && s.supports.predict) || null;
}
function metaScorer() {
  const sel = currentScorer();
  if (sel && sel.available && sel.supports.meta) return sel;
  return state.scorers.find((s) => s.available && s.supports.meta) || null;
}

async function getJSON(url) {
  const res = await fetch(url);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
  return body;
}

function lsGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
function lsSet(key, val) { try { localStorage.setItem(key, val); } catch (_) { /* private mode */ } }

/* ----------------------- shared card renderer ------------------------ */
// opts: { evolved, hero, champion, evoAvailable, heroAvailable, clickable, onClick, dim, title }
function cardEl(card, opts = {}) {
  const node = el("div", "card" + (opts.clickable ? " clickable" : "") + (opts.dim ? " dim" : ""));
  node.dataset.rarity = card.rarity;
  if (opts.title) node.title = opts.title;

  const elixir = el("div", "elixir");
  elixir.textContent = card.elixir;
  node.appendChild(elixir);

  // badges. A card takes one form: champions show the crown, other heroes show
  // HERO, evolutions show EVO. In the pool we also hint which forms are available.
  const badges = el("div", "badges");
  if (opts.champion) badges.appendChild(makeBadge("champ", "👑"));
  else if (opts.evolved) badges.appendChild(makeBadge("evo", "EVO"));
  else if (opts.hero) badges.appendChild(makeBadge("hero", "HERO"));
  if (opts.evoAvailable && !opts.evolved && !opts.champion)
    badges.appendChild(makeBadge("evo-avail", "EVO?"));
  if (opts.heroAvailable && !opts.hero && !opts.champion)
    badges.appendChild(makeBadge("hero-avail", "HERO?"));
  if (badges.children.length) node.appendChild(badges);

  // art (with graceful fallback to initials over a tinted backdrop)
  const wrap = el("div", "art-wrap");
  const ini = el("div", "initials");
  ini.textContent = initials(card.name);
  wrap.appendChild(ini);

  const img = el("img", "art");
  img.alt = card.name;
  img.loading = "lazy";
  img.src = artUrl(card.name);
  img.addEventListener("error", () => { wrap.classList.add("no-art"); img.remove(); });
  wrap.appendChild(img);
  node.appendChild(wrap);

  const name = el("div", "name");
  name.textContent = card.name;
  node.appendChild(name);

  const strip = el("div", "rarity-strip");
  strip.textContent = card.rarity;
  node.appendChild(strip);

  if (opts.clickable && opts.onClick) node.addEventListener("click", () => opts.onClick(card));
  return node;
}

function makeBadge(kind, text) {
  const b = el("span", "badge " + kind);
  b.textContent = text;
  return b;
}

// A compact 8-tile strip for a deck payload (meta decks, matchup rows).
function miniDeck(deck) {
  const strip = el("div", "mini-deck");
  for (const c of deck.cards) {
    const t = el("div", "mini-card");
    t.dataset.rarity = c.rarity;
    t.title = `${c.name} · ${c.elixir} elixir` + (c.form !== "base" ? ` · ${c.form.toUpperCase()}` : "");
    const ini = el("span", "mini-ini");
    ini.textContent = initials(c.name);
    t.appendChild(ini);
    const img = el("img");
    img.alt = c.name;
    img.loading = "lazy";
    img.src = artUrl(c.name);
    img.addEventListener("error", () => { t.classList.add("no-art"); img.remove(); });
    t.appendChild(img);
    if (c.is_champion) t.appendChild(makeBadge("champ mini", "👑"));
    else if (c.form === "evo") t.appendChild(makeBadge("evo mini", "E"));
    else if (c.form === "hero") t.appendChild(makeBadge("hero mini", "H"));
    strip.appendChild(t);
  }
  return strip;
}

/* ------------------------------ tabs --------------------------------- */
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("is-active", p.id === "tab-" + name));
  if (name === "matchups") {
    ensureMeta();
    renderMatchupAvailability();
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => showTab(tab.dataset.tab))
  );
}

/* ----------------------------- sliders ------------------------------- */
function fillRange(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.background =
    `linear-gradient(90deg, var(--gold) 0%, var(--gold) ${pct}%, rgba(255,255,255,0.12) ${pct}%)`;
}

function setupSlider(id) {
  const input = $("#" + id);
  const out = $("#" + id + "-val");
  const sync = () => { out.textContent = input.value; fillRange(input); };
  input.addEventListener("input", sync);
  sync();
}

/* -------------------------- initial config --------------------------- */
async function loadConfig() {
  try {
    state.config = await getJSON("/api/config");
  } catch (_) { return; }
  const c = state.config;
  applyLimit("population", c.limits.population, c.defaults.population);
  applyLimit("generations", c.limits.generations, c.defaults.generations);
  applyScorers(c);
}

function applyLimit(id, [lo, hi], def) {
  const input = $("#" + id);
  input.min = lo;
  input.max = hi;
  input.value = def;
  $("#" + id + "-val").textContent = def;
  fillRange(input);
}

/* --------------------------- scorer picker --------------------------- */
function applyScorers(payload) {
  state.scorers = payload.scorers || [];
  const remembered = lsGet(LS_SCORER);
  const pick = scorerById(remembered);
  state.scorerId = (pick && pick.available) ? pick.id : payload.default_scorer;
  renderScorerPicker();
  renderMatchupAvailability();
  renderH2H();
}

function renderScorerPicker() {
  const box = $("#scorer-picker");
  box.innerHTML = "";
  for (const s of state.scorers) {
    const opt = el("label", "scorer-opt" + (s.available ? "" : " off") + (s.id === state.scorerId ? " on" : ""));
    const input = el("input");
    input.type = "radio";
    input.name = "scorer";
    input.value = s.id;
    input.disabled = !s.available;
    input.checked = s.id === state.scorerId;
    input.addEventListener("change", () => {
      state.scorerId = s.id;
      lsSet(LS_SCORER, s.id);
      renderScorerPicker();
      renderMatchupAvailability();
      state.meta.decks = null; // meta set belongs to a scorer
      if ($("#tab-matchups").classList.contains("is-active")) ensureMeta();
      scheduleH2H();
    });

    const body = el("div", "scorer-body");
    const title = el("div", "scorer-title");
    title.textContent = `${SCORER_ICON[s.id] || "•"} ${s.label}`;
    const kind = el("span", "scorer-kind " + s.kind);
    kind.textContent = s.kind === "winrate" ? "win rate" : "score";
    title.appendChild(kind);
    if (!s.available) {
      const off = el("span", "scorer-kind off");
      off.textContent = "unavailable";
      title.appendChild(off);
    }
    const desc = el("div", "scorer-desc");
    desc.textContent = s.available ? s.description : s.reason;
    body.append(title, desc);
    if (s.available && s.supports.info) {
      const info = el("div", "scorer-info");
      info.id = "scorer-info-" + s.id;
      renderScorerInfo(info, s.id);
      body.appendChild(info);
    }
    opt.append(input, body);
    box.appendChild(opt);
  }
  if (!state.scorers.length) {
    const p = el("div", "scorer-desc");
    p.textContent = "No scorers reported by the server.";
    box.appendChild(p);
  }
  const sel = currentScorer();
  const runnable = !!(sel && sel.available);
  $("#run-btn").disabled = !runnable || !!state.eventSource;
  $("#run-btn").title = runnable ? "" : "Pick an available scorer first";
}

// The model's own info(): device, meta size, validation metrics, and whether
// hero form was in the training data (if not, the hero slot can't move fitness).
const INFO_LABELS = {
  device: (v) => `${v}`.startsWith("cuda") ? "🟢 GPU" : `💻 ${v}`,
  meta_decks: (v) => `${v} meta decks`,
  has_hero_data: (v) => (v ? "hero form learned" : "⚠ hero form ignored (no hero data)"),
};
const INFO_SKIP = new Set(["device", "meta_decks", "has_hero_data"]);

function renderScorerInfo(node, id) {
  const info = state.scorerInfo[id];
  node.innerHTML = "";
  if (info === undefined) {
    node.textContent = "loading model info…";
    fetchScorerInfo(id);
    return;
  }
  if (info && info.error) { node.textContent = "ⓘ " + info.error; node.classList.add("warn"); return; }
  const chips = [];
  for (const k of Object.keys(INFO_LABELS)) if (k in info) chips.push(INFO_LABELS[k](info[k]));
  for (const [k, v] of Object.entries(info)) {
    if (INFO_SKIP.has(k) || v === null || typeof v === "object") continue;
    if (chips.length >= 8) break;
    const val = typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : String(v);
    chips.push(`${k.replace(/_/g, " ")} ${val}`);
  }
  chips.forEach((c) => {
    const span = el("span", "info-chip" + (c.startsWith("⚠") ? " warn" : ""));
    span.textContent = c;
    node.appendChild(span);
  });
}

async function fetchScorerInfo(id) {
  if (state.scorerInfo[id] !== undefined) return;
  state.scorerInfo[id] = null; // in flight
  try {
    const r = await getJSON("/api/scorer_info?scorer=" + encodeURIComponent(id));
    state.scorerInfo[id] = r.info || {};
  } catch (err) {
    state.scorerInfo[id] = { error: err.message };
  }
  const node = $("#scorer-info-" + id);
  if (node) renderScorerInfo(node, id);
}

async function rescanScorers() {
  const btn = $("#scorer-rescan");
  btn.disabled = true;
  btn.textContent = "↻ scanning…";
  try {
    state.scorerInfo = {};
    applyScorers(await getJSON("/api/scorers?rescan=1"));
  } catch (err) {
    alert("Rescan failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "↻ rescan";
  }
}

/* ----------------------------- optimize ------------------------------ */
function setupOptimize() {
  $("#run-btn").addEventListener("click", runOptimize);
  $("#stop-btn").addEventListener("click", stopOptimize);
  $("#scorer-rescan").addEventListener("click", rescanScorers);
  $("#send-a-btn").addEventListener("click", () => {
    if (!state.best) return;
    loadDeckInto("a", state.best);
    showTab("matchups");
  });
  $("#copy-deck-btn").addEventListener("click", copyBestDeck);
}

function runOptimize() {
  const scorer = currentScorer();
  if (!scorer || !scorer.available) return;
  if (state.eventSource) state.eventSource.close();

  const population = $("#population").value;
  const generations = $("#generations").value;
  const seed = $("#seed").value.trim();

  const params = new URLSearchParams({ population, generations, scorer: scorer.id });
  if (seed !== "") params.set("seed", seed);

  state.run = { kind: scorer.kind, scorerId: scorer.id, total: Number(generations), lastGen: 0, lastDeck: null };
  state.best = null;

  setRunning(true);
  resetProgress(Number(generations));
  resetLive(Number(generations));
  resetResult();

  const es = new EventSource("/api/optimize?" + params.toString());
  state.eventSource = es;
  let finished = false;

  es.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    if (d.scorer && d.scorer.kind) state.run.kind = d.scorer.kind;
    updateLegend();
  });

  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    state.run.lastGen = d.gen;
    updateProgress(d);
    pushLive(d);
    if (d.deck) {
      state.run.lastDeck = d.deck;
      renderDeck(d.deck, { live: true });
    }
  });

  es.addEventListener("done", (e) => {
    finished = true;
    es.close();
    state.eventSource = null;
    finishRun(JSON.parse(e.data), { stopped: false });
    setRunning(false);
  });

  es.addEventListener("failed", (e) => {
    finished = true;
    showRunError(JSON.parse(e.data).message || "The optimizer failed.");
    $("#live-pill").hidden = true;
    es.close();
    state.eventSource = null;
    setRunning(false);
  });

  es.onerror = () => {
    if (finished) return;
    showRunError("Lost the connection to the optimizer. Is the server still running?");
    $("#live-pill").hidden = true;
    es.close();
    state.eventSource = null;
    setRunning(false);
  };
}

// Closing the EventSource is the signal: the server notices on its next
// progress write and abandons the run, so the GPU isn't left evolving a deck
// nobody is watching.
function stopOptimize() {
  const es = state.eventSource;
  if (!es) return;
  es.close();
  state.eventSource = null;
  setRunning(false);
  $("#live-pill").hidden = true;
  const deck = state.run.lastDeck;
  if (deck) finishRun(deck, { stopped: true });
  else showRunError("Stopped before the first generation finished.");
}

async function finishRun(deck, { stopped }) {
  state.best = deck;
  renderDeck(deck, { live: false });
  $("#live-pill").hidden = true;
  $("#result-actions").hidden = false;
  showResultNote(deck, stopped);

  if (deck.matchups) {
    renderMatchups(deck);
    return;
  }
  // A stopped run only has the last streamed deck; ask the server for its
  // breakdown when the scorer can produce one.
  const scorer = scorerById(state.run.scorerId);
  if (stopped && scorer && scorer.supports.matchups) {
    try {
      const full = await getJSON("/api/evaluate?" + deckParams(deckToSlots(deck), "cards", scorer.id));
      if (state.best === deck) {
        state.best = { ...deck, matchups: full.matchups };
        renderMatchups(state.best);
      }
    } catch (_) { $("#matchups-panel").hidden = true; }
  } else {
    $("#matchups-panel").hidden = true;
  }
}

/* --------------------- live evolution visuals ------------------------ */
const chartState = { best: [], avg: [], total: 0 };
let chartQueued = false;

function resetLive(total) {
  chartState.best = [];
  chartState.avg = [];
  chartState.total = Math.max(total, 2);
  $("#gen-log").innerHTML = "";
  $("#pop-stats").innerHTML = "";
  $("#live-panel").hidden = false;
  $("#live-pill").hidden = false;
  updateLegend();
  drawChart();
}

function updateLegend() {
  $("#legend-best").textContent = state.run.kind === "winrate" ? "Best expected win rate" : "Best fitness";
}

function pushLive(d) {
  chartState.best.push(d.best_fitness);
  chartState.avg.push(d.avg_fitness);
  if (d.total) chartState.total = d.total;
  scheduleChart();
  updatePopStats(d);
  logGen(d);
}

function updatePopStats(d) {
  const div = $("#pop-stats");
  div.innerHTML = "";
  div.appendChild(popChip(fmtFit(d.best_fitness), "best"));
  div.appendChild(popChip(fmtFit(d.avg_fitness), "avg"));
  div.appendChild(popChip(fmtFit(d.worst_fitness), "worst"));
  div.appendChild(popChip(`${d.diversity}/${d.pop_size}`, "unique"));
  div.appendChild(popChip(d.best_avg_elixir, "elixir"));
}

function popChip(val, label) {
  const c = el("div", "pop-chip");
  const v = el("span", "pc-val"); v.textContent = val;
  const l = el("span", "pc-lbl"); l.textContent = label;
  c.append(v, l);
  return c;
}

function logGen(d) {
  const log = $("#gen-log");
  const line = el("div", "log-line");
  line.innerHTML =
    `<span class="lg-gen">gen ${String(d.gen).padStart(3, "0")}</span> ` +
    `best <b>${fmtFit(d.best_fitness)}</b> · avg ${fmtFit(d.avg_fitness)} · ` +
    `unique ${d.diversity}`;
  log.insertBefore(line, log.firstChild);
  while (log.children.length > 120) log.removeChild(log.lastChild);
}

function scheduleChart() {
  if (chartQueued) return;
  chartQueued = true;
  requestAnimationFrame(() => { chartQueued = false; drawChart(); });
}

function drawChart() {
  const canvas = $("#fitness-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = 220;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const winrate = state.run.kind === "winrate";
  const pad = { l: winrate ? 44 : 48, r: 14, t: 12, b: 22 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const best = chartState.best.filter(isNum);
  const avg = chartState.avg.filter(isNum);
  const total = Math.max(chartState.total, 2);

  // y-range across both series (handles the flat-line case)
  let lo = Infinity, hi = -Infinity;
  for (const v of best) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const v of avg) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= winrate ? 0.05 : 1; hi += winrate ? 0.05 : 1; }
  const margin = (hi - lo) * 0.08;
  lo -= margin; hi += margin;
  if (winrate) { lo = Math.max(0, lo); hi = Math.min(1, hi); }

  const X = (i) => pad.l + (total <= 1 ? 0 : (i / (total - 1)) * w);
  const Y = (v) => pad.t + h - ((v - lo) / (hi - lo)) * h;
  const tick = (v) => (winrate ? (v * 100).toFixed(1) + "%" : v.toFixed(2));

  // gridlines + y labels
  ctx.strokeStyle = "rgba(160,190,240,0.14)";
  ctx.fillStyle = "rgba(200,215,245,0.7)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = pad.t + (h / 4) * g;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + w, yy); ctx.stroke();
    ctx.fillText(tick(hi - ((hi - lo) / 4) * g), 4, yy + 3);
  }
  // the coin-flip line is the natural reference for a win-rate fitness
  if (winrate && lo < 0.5 && hi > 0.5) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0.5)); ctx.lineTo(pad.l + w, Y(0.5)); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillText("50%", pad.l + w - 26, Y(0.5) - 4);
    ctx.fillStyle = "rgba(200,215,245,0.7)";
  }
  ctx.fillText("gen 1", pad.l, cssH - 6);
  ctx.fillText(String(total), pad.l + w - 18, cssH - 6);

  const drawLine = (data, color, width) => {
    if (!data.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
  };
  drawLine(avg, "#b14cff", 2);
  drawLine(best, "#ffd23f", 2.5);

  // marker on the latest best point
  if (best.length) {
    const i = best.length - 1;
    ctx.fillStyle = "#ffd23f";
    ctx.beginPath(); ctx.arc(X(i), Y(best[i]), 3.5, 0, Math.PI * 2); ctx.fill();
  }
}

window.addEventListener("resize", () => {
  if (!$("#live-panel").hidden) scheduleChart();
});

function setRunning(running) {
  const btn = $("#run-btn");
  btn.disabled = running || !(currentScorer() && currentScorer().available);
  btn.querySelector(".run-btn-label").textContent = running ? "⏳ Evolving…" : "⚡ Optimize Deck";
  $("#stop-btn").hidden = !running;
  document.querySelectorAll("#scorer-picker input").forEach((i) => {
    const s = scorerById(i.value);
    i.disabled = running || !(s && s.available);
  });
  if (running) $("#progress").hidden = false; // stays visible after finishing too
}

function resetProgress(total) {
  $("#progress-bar").style.width = "0%";
  $("#progress-gen").textContent = `Generation 0 / ${total}`;
  $("#progress-fitness").textContent = "best —";
}

function updateProgress(d) {
  const pct = d.total ? (d.gen / d.total) * 100 : 0;
  $("#progress-bar").style.width = pct + "%";
  $("#progress-gen").textContent = `Generation ${d.gen} / ${d.total}`;
  $("#progress-fitness").textContent = "best " + fmtFit(d.best_fitness);
}

/* --------------------------- render deck ----------------------------- */
// Lay the deck out by slot: evolution (upper-left), then hero, then the wild
// slot (the 2nd evolution OR 2nd hero, per config.py's slot model), then the
// remaining cards by elixir. Mirrors the engine's slot rules. Forms come from
// the card's `form` field ("evo"/"hero"/"base"); heroes include champions.
function orderDeckForDisplay(cards) {
  const evolved = cards.filter((c) => c.form === "evo");
  const heroes = cards.filter((c) => c.form === "hero"); // disjoint from evolved

  const evoSlot = evolved[0] || null;
  const heroSlot = heroes[0] || null;
  let wildSlot = null;
  if (evolved.length >= 2) wildSlot = evolved[1];     // wild used as 2nd evo
  else if (heroes.length >= 2) wildSlot = heroes[1];  // wild used as 2nd hero

  const slots = [];
  const used = new Set();
  const place = (card, kind, label) => {
    if (!card || used.has(card.id)) return;
    used.add(card.id);
    slots.push({ card, kind, label });
  };
  place(evoSlot, "evo", "Evolution");
  place(heroSlot, "hero", "Hero");
  place(wildSlot, "wild", "Wild");

  cards
    .filter((c) => !used.has(c.id))
    .sort((a, b) => a.elixir - b.elixir || a.name.localeCompare(b.name))
    .forEach((c) => slots.push({ card: c, kind: null, label: "" }));

  return slots;
}

function resetResult() {
  $("#result-actions").hidden = true;
  $("#result-note").hidden = true;
  $("#matchups-panel").hidden = true;
}

function renderDeck(deck, opts = {}) {
  $("#result-empty").hidden = true;
  const grid = $("#deck-grid");
  grid.innerHTML = "";

  for (const slot of orderDeckForDisplay(deck.cards)) {
    const card = cardEl(slot.card, {
      evolved: slot.card.form === "evo",
      hero: slot.card.form === "hero",
      champion: slot.card.is_champion,
    });
    if (opts.live) card.classList.add("just-updated"); // pulse on real mutations

    const cell = el("div", "deck-slot");
    const tag = el("div", "slot-tag" + (slot.kind ? " on slot-" + slot.kind : ""));
    tag.textContent = slot.label || "";
    cell.append(tag, card);
    grid.appendChild(cell);
  }

  const kind = deck.fitness_kind || state.run.kind;
  const stats = $("#deck-stats");
  stats.hidden = false;
  stats.innerHTML = "";
  stats.appendChild(statBox(deck.avg_elixir.toFixed(2), "avg elixir"));
  stats.appendChild(statBox(deck.num_evolutions, "evolutions"));
  stats.appendChild(statBox(deck.num_heroes, "heroes"));
  const fit = statBox(fmtFit(deck.fitness, kind), fitLabel(kind));
  fit.classList.add("fit");
  stats.appendChild(fit);
  if (!deck.valid) {
    const s = statBox("✗", "invalid");
    s.classList.add("bad");
    s.title = deck.valid_reason;
    stats.appendChild(s);
  }
}

function showResultNote(deck, stopped) {
  const note = $("#result-note");
  const kind = deck.fitness_kind || state.run.kind;
  const scorer = scorerById(deck.scorer || state.run.scorerId);
  const parts = [];
  if (stopped) {
    parts.push(`⏹ Stopped at generation ${state.run.lastGen} of ${state.run.total} — this is the best deck found so far.`);
  }
  if (kind === "winrate") {
    const n = deck.matchups ? deck.matchups.length : null;
    parts.push(
      `ⓘ ${fitLabel(kind)} is the usage-weighted expected win rate` +
      (n ? ` against ${n} meta deck${n === 1 ? "" : "s"}` : " against the meta decks") +
      `, predicted by the ${scorer ? scorer.label.toLowerCase() : "learned model"}.`
    );
  } else if (deck.fitness === 0) {
    parts.push(
      "ⓘ The heuristic scored this deck 0, so it's valid but effectively unranked. " +
      "Tune optimizer/heuristic.py (or switch to the learned model) and re-run."
    );
  }
  if (deck.matchups_error) parts.push(`⚠ Matchup breakdown failed: ${deck.matchups_error}`);
  note.textContent = parts.join(" ");
  note.hidden = parts.length === 0;
}

function statBox(val, label) {
  const box = el("div", "stat");
  const v = el("div", "stat-val"); v.textContent = val;
  const l = el("div", "stat-lbl"); l.textContent = label;
  box.appendChild(v); box.appendChild(l);
  return box;
}

function showRunError(msg) {
  $("#result-empty").hidden = false;
  $("#result-empty").textContent = msg;
  $("#deck-grid").innerHTML = "";
  $("#deck-stats").hidden = true;
  $("#deck-stats").innerHTML = "";
  resetResult();
}

function copyBestDeck() {
  if (!state.best) return;
  const lines = state.best.cards.map((c) => {
    const tag = c.is_champion ? "champion" : (c.form !== "base" ? c.form : "");
    return `${c.name}${tag ? ` (${tag})` : ""}`;
  });
  const text = lines.join(", ");
  const done = () => {
    const btn = $("#copy-deck-btn");
    btn.textContent = "✓ Copied";
    setTimeout(() => { btn.textContent = "📋 Copy card list"; }, 1400);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, () => window.prompt("Copy your deck:", text));
  } else {
    window.prompt("Copy your deck:", text);
  }
}

/* ----------------------- matchups vs the meta ------------------------ */
function pwinClass(p) {
  if (!isNum(p)) return "na";
  if (p >= 0.55) return "good";
  if (p <= 0.45) return "bad";
  return "even";
}

function renderMatchups(deck) {
  const panel = $("#matchups-panel");
  const rows = deck.matchups;
  if (!rows || !rows.length) { panel.hidden = true; return; }
  panel.hidden = false;

  const scored = rows.filter((r) => isNum(r.p_win));
  const weighted = scored.reduce((s, r) => s + r.p_win * r.share, 0);
  const favorable = scored.filter((r) => r.p_win >= 0.5).length;
  const worst = scored.reduce((m, r) => (m === null || r.p_win < m.p_win ? r : m), null);
  const bestRow = scored.reduce((m, r) => (m === null || r.p_win > m.p_win ? r : m), null);

  const sum = $("#matchups-summary");
  sum.innerHTML = "";
  sum.appendChild(statBox(pct(weighted), "weighted"));
  sum.appendChild(statBox(`${favorable}/${rows.length}`, "favorable"));
  if (bestRow) { const s = statBox(pct(bestRow.p_win), "best"); s.title = bestRow.name; sum.appendChild(s); }
  if (worst) {
    const s = statBox(pct(worst.p_win), "worst");
    s.title = worst.name;
    if (worst.p_win < 0.5) s.classList.add("bad"); // only alarming when it's an actual losing matchup
    sum.appendChild(s);
  }

  const list = $("#matchups-list");
  list.innerHTML = "";
  rows.forEach((r, i) => {
    const row = el("div", "mu-row");

    const head = el("div", "mu-head");
    const name = el("div", "mu-name");
    name.textContent = r.name;
    const share = el("span", "mu-share");
    share.textContent = `${pct(r.share, 1)} usage`;
    share.title = isNum(r.weight) ? `raw weight ${r.weight}` : "";
    name.appendChild(share);
    const vs = el("button", "ghost-btn xs");
    vs.textContent = "🆚";
    vs.title = "Open this matchup in the Matchups tab";
    vs.addEventListener("click", () => {
      loadDeckInto("a", deck);
      loadDeckInto("b", r.deck);
      showTab("matchups");
    });
    head.append(name, vs);

    const bar = el("div", "mu-bar " + pwinClass(r.p_win));
    const fill = el("div", "mu-fill");
    fill.style.width = isNum(r.p_win) ? `${Math.max(0, Math.min(100, r.p_win * 100))}%` : "0%";
    const mid = el("div", "mu-mid");
    const label = el("div", "mu-pct");
    label.textContent = pct(r.p_win);
    bar.append(fill, mid, label);

    row.append(head, miniDeck(r.deck), bar);
    if (i >= 12) row.classList.add("mu-extra");
    list.appendChild(row);
  });

  if (rows.length > 12) {
    const more = el("button", "ghost-btn sm mu-more");
    more.textContent = `Show all ${rows.length} meta decks`;
    more.addEventListener("click", () => {
      list.classList.add("expanded");
      more.remove();
    });
    list.appendChild(more);
  }
}

/* --------------------------- deck builder ---------------------------- */
function emptyDeck() { return { slots: Array(8).fill(null) }; }

function deckSize() { return (state.config && state.config.deck_size) || 8; }

// Mirror of optimizer/config.py's slots_ok(); the server re-checks anyway.
function slotsOk(nEvo, nHero) {
  const s = (state.config && state.config.slots) || { base_evolution: 1, base_champion: 1, wild: 1 };
  const extraEvo = Math.max(0, nEvo - s.base_evolution);
  const extraHero = Math.max(0, nHero - s.base_champion);
  return nEvo <= s.base_evolution + s.wild &&
    nHero <= s.base_champion + s.wild &&
    extraEvo + extraHero <= s.wild;
}

function eligibleForms(card) {
  if (card.is_champion) return ["hero"]; // champions have no base form
  const forms = ["base"];
  if (card.has_evolution) forms.push("evo");
  if (card.is_champion_hero) forms.push("hero");
  return forms;
}

function filledSlots(side) { return state.builder[side].slots.filter(Boolean); }
function formCounts(side, skipIndex = -1) {
  let nEvo = 0, nHero = 0;
  state.builder[side].slots.forEach((s, i) => {
    if (!s || i === skipIndex) return;
    if (s.form === "evo") nEvo++;
    else if (s.form === "hero") nHero++;
  });
  return { nEvo, nHero };
}

// `ok` = 8 cards, so the model can score it. `legal` additionally means the
// evo/hero forms fit the engine's slot budget; loaded ladder decks may not,
// and the model still rates them, so that's a warning rather than a block.
function builderValidity(side) {
  const slots = filledSlots(side);
  const size = deckSize();
  if (slots.length < size) return { ok: false, legal: false, partial: true, reason: `${slots.length} / ${size}` };
  const { nEvo, nHero } = formCounts(side);
  if (!slotsOk(nEvo, nHero)) {
    return { ok: true, legal: false, partial: false, reason: `${nEvo} evo + ${nHero} hero exceed the slots` };
  }
  // A loaded ladder deck may hold a form the card doesn't have (per cards.csv).
  const odd = slots.find((s) => !eligibleForms(state.byId.get(s.id)).includes(s.form));
  if (odd) {
    const card = state.byId.get(odd.id);
    const why = odd.form === "evo" ? "has no evolution" : "can't take hero form";
    return { ok: true, legal: false, partial: false, reason: `${card.name} ${why}` };
  }
  return { ok: true, legal: true, partial: false, reason: "valid" };
}

function saveBuilder() {
  lsSet(LS_BUILDER, JSON.stringify({ a: state.builder.a.slots, b: state.builder.b.slots }));
}

function restoreBuilder() {
  let saved = null;
  try { saved = JSON.parse(lsGet(LS_BUILDER) || "null"); } catch (_) { saved = null; }
  if (!saved) return;
  for (const side of ["a", "b"]) {
    const slots = Array.isArray(saved[side]) ? saved[side] : [];
    const clean = Array(deckSize()).fill(null);
    slots.slice(0, deckSize()).forEach((s, i) => {
      if (s && state.byId.has(s.id)) clean[i] = { id: s.id, form: s.form || "base" };
    });
    state.builder[side].slots = clean;
  }
}

function renderBuilder(side) {
  const grid = $("#builder-" + side);
  grid.innerHTML = "";
  const deck = state.builder[side];
  deck.slots.forEach((slot, i) => {
    if (!slot) {
      const e = el("button", "slot-empty");
      e.type = "button";
      e.innerHTML = "<span>+</span>";
      e.title = "Add a card";
      e.addEventListener("click", () => openPicker(side, i));
      grid.appendChild(e);
      return;
    }
    const card = state.byId.get(slot.id);
    const node = cardEl(card, {
      evolved: slot.form === "evo",
      hero: slot.form === "hero",
      champion: card.is_champion,
    });
    const wrap = el("div", "b-slot");
    wrap.appendChild(node);

    const tools = el("div", "b-tools");
    const forms = eligibleForms(card);
    const formBtn = el("button", "pill form-" + slot.form);
    formBtn.type = "button";
    formBtn.textContent = card.is_champion ? "👑 HERO" : slot.form.toUpperCase();
    if (!forms.includes(slot.form)) {
      formBtn.classList.add("form-invalid");
      formBtn.title = `${card.name} has no ${slot.form} form in cards.csv — click to reset`;
      formBtn.addEventListener("click", () => cycleForm(side, i));
    } else if (forms.length > 1) {
      formBtn.title = "Cycle form: " + forms.join(" → ");
      formBtn.addEventListener("click", () => cycleForm(side, i));
    } else {
      formBtn.disabled = true;
      formBtn.title = card.is_champion ? "Champions are always hero form" : "No alternate form";
    }
    const rm = el("button", "pill rm");
    rm.type = "button";
    rm.textContent = "✕";
    rm.title = "Remove";
    rm.addEventListener("click", () => { deck.slots[i] = null; builderChanged(side); });
    tools.append(formBtn, rm);
    wrap.appendChild(tools);
    grid.appendChild(wrap);
  });
  updateBuilderStatus(side);
}

function updateBuilderStatus(side) {
  const v = builderValidity(side);
  const st = $("#status-" + side);
  const filled = filledSlots(side);
  const avg = filled.length
    ? (filled.reduce((s, x) => s + state.byId.get(x.id).elixir, 0) / filled.length).toFixed(2)
    : null;
  const { nEvo, nHero } = formCounts(side);
  if (v.legal) {
    st.textContent = `✓ valid · ${avg} avg elixir · ${nEvo} evo · ${nHero} hero`;
    st.className = "builder-status ok";
  } else if (v.ok) {
    st.textContent = `⚠ ${v.reason} · ${avg} avg elixir`;
    st.title = "Not a legal ladder deck under the current slot rules; the model still scores it.";
    st.className = "builder-status warn";
  } else {
    st.textContent = `${filled.length} / ${deckSize()}` + (avg ? ` · ${avg} avg elixir` : "");
    st.className = "builder-status";
  }
}

function cycleForm(side, index) {
  const slot = state.builder[side].slots[index];
  const card = state.byId.get(slot.id);
  const forms = eligibleForms(card);
  const { nEvo, nHero } = formCounts(side, index);
  let next = forms[(forms.indexOf(slot.form) + 1) % forms.length];
  // Skip forms the slot budget can't take; "base" always fits.
  for (let tries = 0; tries < forms.length; tries++) {
    const fits = next === "base" ||
      (next === "evo" && slotsOk(nEvo + 1, nHero)) ||
      (next === "hero" && slotsOk(nEvo, nHero + 1));
    if (fits) break;
    next = forms[(forms.indexOf(next) + 1) % forms.length];
  }
  slot.form = next;
  builderChanged(side);
}

function builderChanged(side) {
  renderBuilder(side);
  saveBuilder();
  scheduleH2H();
}

function deckToSlots(deck) {
  return deck.cards.map((c) => ({ id: c.id, form: c.form || (c.is_champion ? "hero" : "base") }));
}

function loadDeckInto(side, deckPayload) {
  const slots = Array(deckSize()).fill(null);
  deckToSlots(deckPayload).slice(0, deckSize()).forEach((s, i) => {
    if (state.byId.has(s.id)) slots[i] = s;
  });
  state.builder[side].slots = slots;
  builderChanged(side);
}

function addCardToDeck(side, index, card) {
  const deck = state.builder[side];
  if (deck.slots.some((s) => s && s.id === card.id)) return false;
  if (index < 0) index = deck.slots.findIndex((s) => !s);
  if (index < 0) return false;
  deck.slots[index] = { id: card.id, form: card.is_champion ? "hero" : "base" };
  builderChanged(side);
  return true;
}

function setupBuilder() {
  document.querySelectorAll(".builder-foot button[data-act]").forEach((btn) => {
    const side = btn.dataset.side;
    btn.addEventListener("click", (e) => {
      const act = btn.dataset.act;
      if (act === "clear") {
        state.builder[side] = emptyDeck();
        builderChanged(side);
      } else if (act === "best") {
        if (state.best) loadDeckInto(side, state.best);
        else flashStatus(side, "run the optimizer first");
      } else if (act === "meta-menu") {
        e.stopPropagation();
        toggleMetaMenu(side);
      }
    });
  });
  $("#swap-btn").addEventListener("click", () => {
    const a = state.builder.a;
    state.builder.a = state.builder.b;
    state.builder.b = a;
    renderBuilder("a"); renderBuilder("b");
    saveBuilder();
    scheduleH2H();
  });
  document.addEventListener("click", () => closeMetaMenus());
  renderBuilder("a");
  renderBuilder("b");
}

function flashStatus(side, msg) {
  const st = $("#status-" + side);
  st.textContent = msg;
  st.classList.add("bad");
  setTimeout(() => updateBuilderStatus(side), 1400);
}

/* ------------------------- meta-deck dropdown ------------------------ */
function closeMetaMenus() {
  document.querySelectorAll(".dd-menu").forEach((m) => { m.hidden = true; });
}

function toggleMetaMenu(side) {
  const menu = $("#meta-menu-" + side);
  const wasOpen = !menu.hidden;
  closeMetaMenus();
  if (wasOpen) return;
  menu.innerHTML = "";
  const decks = state.meta.decks;
  if (!decks || !decks.length) {
    const p = el("div", "dd-empty");
    p.textContent = state.meta.loading ? "Loading meta decks…" : (state.meta.error || "No meta decks available.");
    menu.appendChild(p);
    if (!state.meta.loading && !state.meta.decks) ensureMeta();
  } else {
    decks.forEach((d) => {
      const item = el("button", "dd-item");
      item.type = "button";
      const n = el("span"); n.textContent = d.name;
      const s = el("span", "dd-share"); s.textContent = pct(d.share, 1);
      item.append(n, s);
      item.addEventListener("click", () => { loadDeckInto(side, d.deck); closeMetaMenus(); });
      menu.appendChild(item);
    });
  }
  menu.hidden = false;
  menu.addEventListener("click", (e) => e.stopPropagation());
}

/* --------------------------- head to head ---------------------------- */
function deckParams(slots, prefix, scorerId) {
  const p = new URLSearchParams();
  p.set(prefix, slots.map((s) => s.id).join(","));
  const evo = slots.filter((s) => s.form === "evo").map((s) => s.id);
  const hero = slots.filter((s) => s.form === "hero").map((s) => s.id);
  if (evo.length) p.set(prefix + "_evo", evo.join(","));
  if (hero.length) p.set(prefix + "_hero", hero.join(","));
  if (scorerId) p.set("scorer", scorerId);
  return p.toString();
}

function renderMatchupAvailability() {
  const note = $("#matchup-unavailable");
  const scorer = predictScorer();
  if (scorer) { note.hidden = true; return; }
  const learned = scorerById("learned");
  let msg;
  if (learned && !learned.available) {
    msg = `Head-to-head predictions need the learned matchup model, which is unavailable: ${learned.reason}`;
  } else if (learned && !learned.supports.predict) {
    msg = "The learned model module doesn't expose predict(deck_a, deck_b), so head-to-head predictions are off.";
  } else {
    msg = "No available scorer can predict head-to-head matchups.";
  }
  note.textContent = msg + " You can still build decks here.";
  note.hidden = false;
}

function scheduleH2H() {
  clearTimeout(state.h2h.timer);
  state.h2h.timer = setTimeout(computeH2H, 220);
}

function setGauge(pAB, pBA) {
  const fill = $("#vs-fill");
  const a = $("#vs-a"), b = $("#vs-b");
  const cap = $("#vs-caption"), sub = $("#vs-sub");
  if (!isNum(pAB)) {
    fill.style.width = "50%";
    $("#vs-gauge").classList.add("idle");
    a.textContent = "—"; b.textContent = "—";
    sub.textContent = "";
    return;
  }
  $("#vs-gauge").classList.remove("idle");
  fill.style.width = `${Math.max(0, Math.min(100, pAB * 100))}%`;
  a.textContent = pct(pAB);
  b.textContent = pct(1 - pAB);
  const lean = pAB > 0.5 ? "Deck A is favored" : (pAB < 0.5 ? "Deck B is favored" : "Dead even");
  cap.textContent = `${lean} — A beats B ${pct(pAB)} of the time`;
  if (isNum(pBA)) {
    const gap = Math.abs(pAB + pBA - 1);
    sub.textContent = `reverse check · B beats A ${pct(pBA)}` + (gap > 0.03 ? ` (asymmetry ${pct(gap)})` : "");
    sub.classList.toggle("warn", gap > 0.03);
  } else {
    sub.textContent = "";
  }
}

function renderH2H() {
  // Called on load / scorer change: repaint statuses and kick a compute.
  updateBuilderStatus("a");
  updateBuilderStatus("b");
  scheduleH2H();
}

async function computeH2H() {
  const seq = ++state.h2h.seq;
  const va = builderValidity("a"), vb = builderValidity("b");
  const fitA = $("#fit-a"), fitB = $("#fit-b");
  const cap = $("#vs-caption");
  fitA.textContent = ""; fitB.textContent = "";

  const scorer = predictScorer();
  const scoreOnly = !scorer && state.scorers.find((s) => s.available && s.supports.score);

  if (!(va.ok && vb.ok)) {
    setGauge(null);
    cap.textContent = va.ok || vb.ok
      ? `Finish Deck ${va.ok ? "B" : "A"} to compare`
      : "Build two valid decks to compare";
    // Still score whichever single deck is complete.
    const single = scorer || scoreOnly;
    if (single) {
      for (const [side, v] of [["a", va], ["b", vb]]) {
        if (!v.ok) continue;
        try {
          const r = await getJSON("/api/evaluate?" + deckParams(filledSlots(side), "cards", single.id));
          if (seq !== state.h2h.seq) return;
          $("#fit-" + side).textContent = `${fitLabel(r.fitness_kind)}: ${fmtFit(r.fitness, r.fitness_kind)}`;
        } catch (_) { /* leave blank */ }
      }
    }
    return;
  }

  if (!scorer) {
    setGauge(null);
    cap.textContent = "Head-to-head prediction unavailable";
    if (scoreOnly) {
      // At least show each deck's standalone score.
      for (const side of ["a", "b"]) {
        try {
          const r = await getJSON("/api/evaluate?" + deckParams(filledSlots(side), "cards", scoreOnly.id));
          if (seq !== state.h2h.seq) return;
          $("#fit-" + side).textContent = `${fitLabel(r.fitness_kind)}: ${fmtFit(r.fitness, r.fitness_kind)}`;
        } catch (_) { /* leave blank */ }
      }
    }
    return;
  }

  cap.textContent = "Predicting…";
  $("#vs-gauge").classList.add("busy");
  try {
    const q = deckParams(filledSlots("a"), "a") + "&" + deckParams(filledSlots("b"), "b", scorer.id);
    const r = await getJSON("/api/matchup?" + q);
    if (seq !== state.h2h.seq) return;
    setGauge(r.p_ab, r.p_ba);
    if (isNum(r.a.fitness)) fitA.textContent = `${fitLabel(r.kind)} vs meta: ${fmtFit(r.a.fitness, r.kind)}`;
    if (isNum(r.b.fitness)) fitB.textContent = `${fitLabel(r.kind)} vs meta: ${fmtFit(r.b.fitness, r.kind)}`;
  } catch (err) {
    if (seq !== state.h2h.seq) return;
    setGauge(null);
    cap.textContent = "⚠ " + err.message;
  } finally {
    if (seq === state.h2h.seq) $("#vs-gauge").classList.remove("busy");
  }
}

/* ----------------------------- meta list ----------------------------- */
async function ensureMeta() {
  const scorer = metaScorer();
  const empty = $("#meta-empty");
  const list = $("#meta-list");
  const count = $("#meta-count");
  if (!scorer) {
    const learned = scorerById("learned");
    state.meta.decks = null;
    state.meta.error = learned && !learned.available
      ? `No meta deck set: ${learned.reason}`
      : "No meta deck set: the scorer doesn't expose meta_decks().";
    empty.textContent = state.meta.error;
    empty.hidden = false;
    list.innerHTML = "";
    count.textContent = "";
    return;
  }
  if (state.meta.decks && state.meta.scorerId === scorer.id) return;
  if (state.meta.loading) return;

  state.meta.loading = true;
  state.meta.error = null;
  empty.textContent = "Loading meta decks…";
  empty.hidden = false;
  try {
    const r = await getJSON("/api/meta?scorer=" + encodeURIComponent(scorer.id));
    state.meta.decks = r.decks || [];
    state.meta.scorerId = scorer.id;
  } catch (err) {
    state.meta.decks = null;
    state.meta.error = "Couldn't load meta decks: " + err.message;
  } finally {
    state.meta.loading = false;
  }
  renderMetaList();
}

function renderMetaList() {
  const empty = $("#meta-empty");
  const list = $("#meta-list");
  const count = $("#meta-count");
  list.innerHTML = "";
  const decks = state.meta.decks;
  if (!decks || !decks.length) {
    empty.textContent = state.meta.error || "The meta deck set is empty.";
    empty.hidden = false;
    count.textContent = "";
    return;
  }
  empty.hidden = true;
  count.textContent = `${decks.length} deck${decks.length === 1 ? "" : "s"}`;
  decks.forEach((d, i) => {
    const row = el("div", "meta-row");
    const head = el("div", "meta-head");
    const rank = el("span", "meta-rank"); rank.textContent = `#${i + 1}`;
    const name = el("span", "meta-name"); name.textContent = d.name;
    const share = el("span", "meta-share"); share.textContent = `${pct(d.share, 1)} usage`;
    const elx = el("span", "meta-elx"); elx.textContent = `${d.deck.avg_elixir.toFixed(1)} elixir`;
    head.append(rank, name, share, elx);
    const acts = el("div", "meta-acts");
    for (const side of ["a", "b"]) {
      const b = el("button", "ghost-btn xs");
      b.type = "button";
      b.textContent = `→ ${side.toUpperCase()}`;
      b.addEventListener("click", () => {
        loadDeckInto(side, d.deck);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      acts.appendChild(b);
    }
    head.appendChild(acts);
    row.append(head, miniDeck(d.deck));
    list.appendChild(row);
  });
}

/* ------------------------------ picker ------------------------------- */
function openPicker(side, index) {
  state.picker = { side, index };
  $("#picker-title").textContent = `Add a card to Deck ${side.toUpperCase()}`;
  $("#picker-search").value = "";
  renderPicker();
  $("#picker").hidden = false;
  setTimeout(() => $("#picker-search").focus(), 30);
}

function closePicker() {
  $("#picker").hidden = true;
  state.picker = { side: null, index: -1 };
}

function renderPicker() {
  const { side, index } = state.picker;
  if (!side) return;
  const q = $("#picker-search").value.trim().toLowerCase();
  const inDeck = new Set(filledSlots(side).map((s) => s.id));
  const { nEvo, nHero } = formCounts(side, index);
  const grid = $("#picker-grid");
  grid.innerHTML = "";

  const matches = state.cards
    .filter((c) => !inDeck.has(c.id) && (!q || c.name.toLowerCase().includes(q)))
    .sort((a, b) => a.elixir - b.elixir || a.name.localeCompare(b.name));

  for (const card of matches) {
    // Champions are hero-form only, so they need a free hero/wild slot.
    const blocked = card.is_champion && !slotsOk(nEvo, nHero + 1);
    grid.appendChild(cardEl(card, {
      champion: card.is_champion,
      evoAvailable: card.has_evolution,
      heroAvailable: card.is_champion_hero,
      clickable: !blocked,
      dim: blocked,
      title: blocked ? "No hero slot left for another champion" : "",
      onClick: (c) => { addCardToDeck(side, index, c); closePicker(); },
    }));
  }
  if (!matches.length) {
    const p = el("p", "empty-state");
    p.textContent = "No cards match.";
    grid.appendChild(p);
  }
}

function setupPicker() {
  document.querySelectorAll("#picker [data-close]").forEach((n) => n.addEventListener("click", closePicker));
  $("#picker-search").addEventListener("input", renderPicker);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#picker").hidden) closePicker();
  });
}

/* ---------------------------- card pool ------------------------------ */
async function loadCards() {
  try {
    state.cards = await getJSON("/api/cards");
  } catch (_) {
    $("#pool-count").textContent = "Couldn't load the card pool.";
    return;
  }
  state.byId = new Map(state.cards.map((c) => [c.id, c]));
  populateElixirFilter();
  renderPool();
  restoreBuilder();
  renderBuilder("a");
  renderBuilder("b");
  scheduleH2H();
}

function populateElixirFilter() {
  const sel = $("#filter-elixir");
  const values = [...new Set(state.cards.map((c) => c.elixir))].sort((a, b) => a - b);
  for (const v of values) {
    const opt = el("option");
    opt.value = String(v);
    opt.textContent = `${v} elixir`;
    sel.appendChild(opt);
  }
}

function setupPool() {
  ["#search", "#filter-rarity", "#filter-type", "#filter-elixir"].forEach((sel) =>
    $(sel).addEventListener("input", renderPool)
  );
}

function renderPool() {
  const q = $("#search").value.trim().toLowerCase();
  const rarity = $("#filter-rarity").value;
  const type = $("#filter-type").value;
  const elixir = $("#filter-elixir").value;

  const matches = state.cards
    .filter((c) => {
      if (q && !c.name.toLowerCase().includes(q)) return false;
      if (rarity && c.rarity !== rarity) return false;
      if (type && c.type !== type) return false;
      if (elixir !== "" && String(c.elixir) !== elixir) return false;
      return true;
    })
    .sort(
      (a, b) =>
        a.elixir - b.elixir ||
        (RARITY_ORDER[a.rarity] ?? 9) - (RARITY_ORDER[b.rarity] ?? 9) ||
        a.name.localeCompare(b.name)
    );

  const grid = $("#pool-grid");
  grid.innerHTML = "";
  for (const card of matches) {
    grid.appendChild(
      cardEl(card, {
        champion: card.is_champion,
        evoAvailable: card.has_evolution,
        heroAvailable: card.is_champion_hero,
        clickable: true,
        onClick: openCard,
      })
    );
  }

  const n = matches.length;
  $("#pool-count").textContent = `${n} card${n === 1 ? "" : "s"}`;
}

/* ---------------------------- card modal ----------------------------- */
function openCard(card) {
  const body = $("#modal-body");
  body.innerHTML = "";

  const head = el("div", "modal-head");

  const art = el("div", "m-art art-wrap");
  const ini = el("div", "initials");
  ini.textContent = initials(card.name);
  art.appendChild(ini);
  const img = el("img", "art");
  img.alt = card.name;
  img.src = artUrl(card.name);
  img.addEventListener("error", () => { art.classList.add("no-art"); img.remove(); });
  art.appendChild(img);

  const meta = el("div");
  const h = el("h3");
  h.textContent = card.name;
  const sub = el("div", "m-sub");
  sub.textContent = `${card.rarity} · ${card.type} · ${card.elixir} elixir`;
  meta.append(h, sub);

  head.append(art, meta);
  body.appendChild(head);

  const table = el("table", "stat-table");
  const stats = card.stats || {};
  const keys = Object.keys(stats);
  if (keys.length) {
    for (const k of keys) {
      const tr = el("tr");
      const label = el("td");
      label.textContent = STAT_LABELS[k] || k.replace(/_/g, " ");
      const val = el("td");
      val.textContent = stats[k];
      tr.append(label, val);
      table.appendChild(tr);
    }
  } else {
    const tr = el("tr");
    const td = el("td", "none");
    td.colSpan = 2;
    td.textContent = "No combat stats recorded for this card.";
    tr.appendChild(td);
    table.appendChild(tr);
  }
  body.appendChild(table);

  // quick add to either builder deck
  const acts = el("div", "modal-acts");
  for (const side of ["a", "b"]) {
    const b = el("button", "ghost-btn sm");
    b.type = "button";
    const has = state.builder[side].slots.some((s) => s && s.id === card.id);
    const full = filledSlots(side).length >= deckSize();
    b.textContent = has ? `✓ In Deck ${side.toUpperCase()}` : `+ Deck ${side.toUpperCase()}`;
    b.disabled = has || full;
    if (full && !has) b.title = `Deck ${side.toUpperCase()} is full`;
    b.addEventListener("click", () => {
      if (addCardToDeck(side, -1, card)) {
        closeModal();
        showTab("matchups");
      }
    });
    acts.appendChild(b);
  }
  body.appendChild(acts);

  $("#modal").hidden = false;
}

function closeModal() {
  $("#modal").hidden = true;
}

function setupModal() {
  document.querySelectorAll("#modal [data-close]").forEach((n) =>
    n.addEventListener("click", closeModal)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#modal").hidden) closeModal();
  });
}

/* ------------------------------ init --------------------------------- */
function init() {
  setupTabs();
  setupSlider("population");
  setupSlider("generations");
  setupOptimize();
  setupPool();
  setupModal();
  setupPicker();
  setupBuilder();
  loadConfig();
  loadCards();
}

document.addEventListener("DOMContentLoaded", init);
