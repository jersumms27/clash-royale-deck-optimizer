/* =====================================================================
   Clash Royale Deck Optimizer — front-end logic
   Talks to the local server (server.py). No frameworks, no build step.
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

const state = {
  cards: [],
  config: null,
  eventSource: null,
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

/* ----------------------- shared card renderer ------------------------ */
// opts: { evolved, hero, champion, evoAvailable, heroAvailable, clickable, onClick }
function cardEl(card, opts = {}) {
  const node = el("div", "card" + (opts.clickable ? " clickable" : ""));
  node.dataset.rarity = card.rarity;

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

/* ------------------------------ tabs --------------------------------- */
function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
      tab.classList.add("is-active");
      $("#tab-" + tab.dataset.tab).classList.add("is-active");
    });
  });
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
    state.config = await (await fetch("/api/config")).json();
  } catch (_) { return; }
  const c = state.config;
  applyLimit("population", c.limits.population, c.defaults.population);
  applyLimit("generations", c.limits.generations, c.defaults.generations);
}

function applyLimit(id, [lo, hi], def) {
  const input = $("#" + id);
  input.min = lo;
  input.max = hi;
  input.value = def;
  $("#" + id + "-val").textContent = def;
  fillRange(input);
}

/* ----------------------------- optimize ------------------------------ */
function setupOptimize() {
  $("#run-btn").addEventListener("click", runOptimize);
}

function runOptimize() {
  if (state.eventSource) state.eventSource.close();

  const population = $("#population").value;
  const generations = $("#generations").value;
  const seed = $("#seed").value.trim();

  const params = new URLSearchParams({ population, generations });
  if (seed !== "") params.set("seed", seed);

  setRunning(true);
  resetProgress(Number(generations));
  resetLive(Number(generations));

  const es = new EventSource("/api/optimize?" + params.toString());
  state.eventSource = es;
  let finished = false;

  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    updateProgress(d);
    pushLive(d);
    if (d.deck) renderDeck(d.deck, { live: true });
  });

  es.addEventListener("done", (e) => {
    finished = true;
    renderDeck(JSON.parse(e.data), { live: false });
    $("#live-pill").hidden = true;
    es.close();
    setRunning(false);
  });

  es.addEventListener("failed", (e) => {
    finished = true;
    showRunError(JSON.parse(e.data).message || "The optimizer failed.");
    $("#live-pill").hidden = true;
    es.close();
    setRunning(false);
  });

  es.onerror = () => {
    if (finished) return;
    showRunError("Lost the connection to the optimizer. Is the server still running?");
    $("#live-pill").hidden = true;
    es.close();
    setRunning(false);
  };
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
  drawChart();
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
  div.appendChild(popChip(fmt(d.best_fitness), "best"));
  div.appendChild(popChip(fmt(d.avg_fitness), "avg"));
  div.appendChild(popChip(fmt(d.worst_fitness), "worst"));
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
    `best <b>${fmt(d.best_fitness)}</b> · avg ${fmt(d.avg_fitness)} · ` +
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

  const pad = { l: 48, r: 14, t: 12, b: 22 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const { best, avg } = chartState;
  const total = Math.max(chartState.total, 2);

  // y-range across both series (handles the flat-line / heuristic=0 case)
  let lo = Infinity, hi = -Infinity;
  for (const v of best) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const v of avg) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const margin = (hi - lo) * 0.08;
  lo -= margin; hi += margin;

  const X = (i) => pad.l + (total <= 1 ? 0 : (i / (total - 1)) * w);
  const Y = (v) => pad.t + h - ((v - lo) / (hi - lo)) * h;

  // gridlines + y labels
  ctx.strokeStyle = "rgba(160,190,240,0.14)";
  ctx.fillStyle = "rgba(200,215,245,0.7)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = pad.t + (h / 4) * g;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + w, yy); ctx.stroke();
    ctx.fillText((hi - ((hi - lo) / 4) * g).toFixed(2), 6, yy + 3);
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
  btn.disabled = running;
  btn.querySelector(".run-btn-label").textContent = running ? "⏳ Evolving…" : "⚡ Optimize Deck";
  if (running) $("#progress").hidden = false; // stays visible after finishing too
}

function resetProgress(total) {
  $("#progress-bar").style.width = "0%";
  $("#progress-gen").textContent = `Generation 0 / ${total}`;
  $("#progress-fitness").textContent = "best fitness —";
}

function updateProgress(d) {
  const pct = d.total ? (d.gen / d.total) * 100 : 0;
  $("#progress-bar").style.width = pct + "%";
  $("#progress-gen").textContent = `Generation ${d.gen} / ${d.total}`;
  $("#progress-fitness").textContent = "best fitness " + fmt(d.best_fitness);
}

function fmt(x) {
  return (typeof x === "number") ? x.toFixed(4) : x;
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

  const stats = $("#deck-stats");
  stats.hidden = false;
  stats.innerHTML = "";
  stats.appendChild(statBox(deck.avg_elixir.toFixed(2), "avg elixir"));
  stats.appendChild(statBox(deck.num_evolutions, "evolutions"));
  stats.appendChild(statBox(deck.num_heroes, "heroes"));
  stats.appendChild(statBox(fmt(deck.fitness), "fitness"));
  if (!deck.valid) {
    const s = statBox("✗", "invalid");
    s.classList.add("bad");
    s.title = deck.valid_reason;
    stats.appendChild(s);
  }

  // Only judge "heuristic still flat" on the final deck, not on every tick.
  if (!opts.live) $("#heuristic-note").hidden = !(deck.fitness === 0);
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
}

/* ---------------------------- card pool ------------------------------ */
async function loadCards() {
  try {
    state.cards = await (await fetch("/api/cards")).json();
  } catch (_) {
    $("#pool-count").textContent = "Couldn't load the card pool.";
    return;
  }
  populateElixirFilter();
  renderPool();
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

  const art = el("div", "m-art");
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
  loadConfig();
  loadCards();
}

document.addEventListener("DOMContentLoaded", init);
