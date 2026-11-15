// Offline functional harness: runs the dashboard's own inline script against the
// real embedded payload, using a minimal DOM shim. Catches property-name typos in
// the renderers, which is the failure mode a static syntax check cannot see.
const fs = require("fs");

const html = fs.readFileSync(process.argv[2], "utf8");
const payloadMatch = /<script id="radar-data" type="application\/json">([\s\S]*?)<\/script>/.exec(html);
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/g)];
const appScript = scripts.find((m) => !/application\/json/.test(m[1]) && m[2].length > 5000)[2];

// ── minimal DOM ───────────────────────────────────────────────────────────────
class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(c) { this.set.add(c); }
  remove(c) { this.set.delete(c); }
  contains(c) { return this.set.has(c); }
  toggle(c, force) {
    if (force === undefined) force = !this.set.has(c);
    if (force) this.set.add(c); else this.set.delete(c);
    return force;
  }
}

class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.textContentValue = "";
    this.className = "";
    this.classList = new ClassList(this);
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.value = "";
    this.title = "";
    this.offsetWidth = 100;
    this.listeners = {};
  }
  get firstChild() { return this.children[0] || null; }
  append(child) { this.children.push(child); child.parentNode = this; return child; }
  appendChild(child) { return this.append(child); }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i < 0) throw new Error("removeChild: not a child");
    this.children.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k]; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, event) { (this.listeners[type] || []).forEach((fn) => fn(event || {})); }
  prepend(child) { this.children.unshift(child); child.parentNode = this; return child; }
  scrollIntoView() { this.scrolled = true; }
  get textContent() { return this.textContentValue; }
  set textContent(v) { this.textContentValue = String(v); this.children = []; }
  // Walk every descendant, so assertions can look for rendered text.
  all() {
    const out = [];
    const walk = (n) => { n.children.forEach((c) => { out.push(c); walk(c); }); };
    walk(this);
    return out;
  }
}

const registry = new Map();
const layerButtons = [1, 2, 3, 4, 5].map(() => {
  const b = new Node("button");
  b.dataset.layer = "none";
  return b;
});
["none", "P_boundary", "W", "G", "S"].forEach((layer, i) => {
  layerButtons[i].dataset.layer = layer;
  layerButtons[i].textContentValue = ["Papers + gaps", "P_boundary", "W", "G", "S"][i];
});

const documentStub = {
  readyState: "complete",
  body: new Node("body"),
  documentElement: new Node("html"),
  createElement(tag) { return new Node(tag); },
  createTextNode(text) { const n = new Node("#text"); n.textContentValue = String(text); return n; },
  createRange() { return { selectNodeContents() {}, }; },
  getElementById(id) {
    if (!registry.has(id)) registry.set(id, new Node(id === "radar-data" ? "script" : "div"));
    return registry.get(id);
  },
  querySelectorAll(selector) {
    if (selector === "[data-layer]") return layerButtons;
    return [];
  },
  addEventListener() {},
  execCommand() { return true; },
};

const payloadNode = documentStub.getElementById("radar-data");
payloadNode.textContent = payloadMatch[1];

// Parsed once here so assertions compare the DOM against the real payload
// instead of against numbers frozen at whatever corpus run wrote this file.
const DATA = JSON.parse(payloadMatch[1]);

let plotlyCalls = [];
function resolve(target) {
  return typeof target === "string" ? documentStub.getElementById(target) : target;
}
const Plotly = {
  newPlot(element, traces, layout, options) {
    plotlyCalls.push({ kind: "newPlot", el: resolve(element), traces, layout, config: options });
    return Promise.resolve();
  },
  restyle(element, update, indices) { plotlyCalls.push({ kind: "restyle", el: resolve(element), update, indices }); return Promise.resolve(); },
  react(element, traces, layout) { plotlyCalls.push({ kind: "react", el: resolve(element), traces, layout }); return Promise.resolve(); },
  Plots: { resize() {} },
};

const manifoldNode = documentStub.getElementById("manifold");
manifoldNode.on = (type, fn) => { (manifoldNode.listeners[type] = manifoldNode.listeners[type] || []).push(fn); };
manifoldNode.parentElement = new Node("div");

const windowStub = {
  location: { hash: "" },
  history: { replaceState() {} },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
  ResizeObserver: class { observe() {} },
  getSelection: () => ({ removeAllRanges() {}, addRange() {} }),
  URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
  Blob: class { constructor(parts) { this.parts = parts; } },
  lucide: { createIcons() { windowStub.__icons = (windowStub.__icons || 0) + 1; } },
  setTimeout: (fn) => { fn(); },
};

// ── run ───────────────────────────────────────────────────────────────────────
const sandbox = {
  document: documentStub, window: windowStub, Plotly,
  navigator: windowStub.navigator, history: windowStub.history,
  location: windowStub.location, ResizeObserver: windowStub.ResizeObserver,
  setTimeout: windowStub.setTimeout, console,
  Blob: windowStub.Blob, URL: windowStub.URL,
  Node, // browsers expose this globally; the page uses `instanceof Node`
  Text: Node,
};
sandbox.globalThis = sandbox;
sandbox.self = sandbox;

const vm = require("vm");
const context = vm.createContext(sandbox);
let failure = null;
try {
  vm.runInContext(appScript, context, { filename: "dashboard-inline.js" });
} catch (error) {
  failure = error;
}

function settle() {
  return new Promise((done) => setImmediate(done));
}

function fail(message) {
  console.error("FAIL  " + message);
  process.exitCode = 1;
}
function pass(message) { console.log("ok    " + message); }

if (failure) {
  fail("script threw: " + failure.message + "\n" + String(failure.stack).split("\n").slice(0, 6).join("\n"));
} else {
  pass("inline script executed without throwing");
}

let manifoldTraces = [];

async function assertions() {
const main = plotlyCalls.find((c) => c.kind === "newPlot" && c.el === manifoldNode);
if (!main) {
  fail("Plotly.newPlot was not called for the manifold");
} else {
  const traces = main.traces;
  manifoldTraces = traces;
  const names = traces.map((t) => t.name);
  pass("manifold traces: " + traces.length + " -> " + names.filter(Boolean).join(", "));

  /* Harvested papers sit one trace per regime (the taxonomy is data-driven, so
     the names come from DATA.regimes rather than a hardcoded list); the 15
     seeding papers are a separate trace with a hollow-diamond marker so their
     positions read separately against the gaps.  Together they must account for
     every paper. */
  const regimeTraceNames = DATA.regimes.map((r) => r.label);
  const papers = traces.filter((t) => t.type === "scatter" && regimeTraceNames.includes(t.name));
  const seeds = traces.filter((t) => t.type === "scatter" && t.name === "Seeding papers");
  const totalPapers = papers.reduce((s, t) => s + t.x.length, 0) +
    seeds.reduce((s, t) => s + t.x.length, 0);
  /* The corpus size belongs to the payload, not to this file: a Zero-Seed run
     has no seeding papers and any corpus change moves the count. */
  const expectedPapers = DATA.papers.length;
  if (totalPapers !== expectedPapers)
    fail("paper traces cover " + totalPapers + " papers, expected " + expectedPapers);
  else pass(regimeTraceNames.length + " regime traces + seed trace cover all " +
    expectedPapers + " papers (" +
    papers.map((t) => t.name + ":" + t.x.length).join(" ") + " seeds:" +
    seeds.reduce((s, t) => s + t.x.length, 0) + ")");

  /* The regime legend must be generated from the payload, so it cannot drift
     from config.yaml the way hardcoded markup would. */
  const legendItems = (documentStub.getElementById("regime-legend") || { children: [] }).children;
  if (legendItems.length !== DATA.regimes.length)
    fail("regime legend has " + legendItems.length + " items, payload has " + DATA.regimes.length);
  else pass("regime legend rendered from the payload: " +
    DATA.regimes.map((r) => r.label).join(", "));

  /* Two phrases name a region but do not explain it, so the hover title carries
     the rest.  Checked only when the payload has summaries at all: a seed-first
     corpus has no discovered regions and therefore nothing to summarise. */
  const summarised = DATA.regimes.filter((r) => r.summary);
  if (summarised.length > 0) {
    const titled = legendItems.filter((item) => item.attributes.title);
    if (titled.length !== summarised.length)
      fail("regime legend has " + titled.length + " hover titles, payload has " +
        summarised.length + " summaries");
    else pass("regime legend carries a hover summary for every discovered region");
  }

  if (seeds.length > 1) fail("expected at most one seeding-paper trace, found " + seeds.length);
  else if (seeds.length === 0) {
    if (DATA.papers.some((p) => p.is_seed))
      fail("payload has seeding papers but no seed trace");
    else pass("no seeding papers in this corpus (Zero-Seed Mode), no seed trace");
  } else if (seeds[0].marker.symbol !== "diamond-open")
    fail("seeding papers use marker " + JSON.stringify(seeds[0].marker.symbol) + ", expected diamond-open");
  else pass("seeding papers drawn as hollow diamonds");
  const seededPapers = DATA.papers.filter((p) => p.is_seed);
  if (seeds.length === 0) {
    /* already reported above */
  } else if (seededPapers.length !== seeds[0].x.length)
    fail("seed trace has " + seeds[0].x.length + " points, payload has " + seededPapers.length);
  else pass("seed trace matches the payload's " + seededPapers.length + " seeding papers");

  const gaps = traces.find((t) => t.name === "Supported Knowledge Gaps");  if (!gaps) fail("no 'Supported Knowledge Gaps' trace");
  else if (gaps.x.length !== DATA.gaps.length)
    fail("gap trace has " + gaps.x.length + " points, payload has " + DATA.gaps.length);
  else pass(DATA.gaps.length + " clickable gap markers");

  const fields = traces.filter((t) => t.type === "heatmap");
  if (fields.length !== 4) fail("expected 4 heatmap layers, got " + fields.length);
  else if (!fields.every((f) => f.z.length === 21 && f.z[0].length === 21))
    fail("a heatmap is not 21x21");
  else if (fields.some((f) => f.visible !== false))
    fail("a heatmap starts visible; the first frame must be papers + gaps");
  else pass("4 heatmap layers, 21x21, all hidden on load");

  const firstField = fields[0];
  const transposed = firstField.z[17][3];
  const direct = firstField.z[3][17];
  if (Math.abs(transposed - direct) < 1e-9) fail("field is symmetric; transposition cannot be judged");
  else pass("field orientation check: z[y=+0.7][x=-0.7]=" + transposed.toFixed(4) +
            " vs z[y=-0.7][x=+0.7]=" + direct.toFixed(4));

  const top = traces.find((t) => t.name === "#1 compromise");
  const best = DATA.gaps[0];
  if (!top) fail("no #1 compromise trace");
  else if (Math.abs(top.x[0] - best.x) > 1e-9 || Math.abs(top.y[0] - best.y) > 1e-9)
    fail("#1 compromise is at (" + top.x[0] + ", " + top.y[0] + "), payload says ("
         + best.x + ", " + best.y + ")");
  else pass("#1 compromise marker at the payload's best point (" + best.x + ", " + best.y + ")");

  // Hover text must carry real content, not placeholders.
  const sample = papers[0].text[0];
  if (!/<b>/.test(sample) || sample.length < 40) fail("paper hover text looks empty");
  else pass("paper hover text populated (" + sample.length + " chars)");
  const gapHover = gaps.text[0];
  if (!/Gap #1/.test(gapHover) || !/P/.test(gapHover)) fail("gap hover text looks empty");
  else pass("gap hover text populated (" + gapHover.length + " chars)");
}

// Drawer state after boot: gap #1 rendered.
const rank = documentStub.getElementById("gap-rank").textContent;
const coords = documentStub.getElementById("gap-coords").textContent;
const promptText = documentStub.getElementById("prompt").textContent;
const stats = documentStub.getElementById("stats").all().map((n) => n.textContent).filter(Boolean);
const anchors = documentStub.getElementById("anchors").all().filter((n) => n.tagName === "A");
const options = documentStub.getElementById("gap-select").all().filter((n) => n.tagName === "OPTION");

if (rank !== "#1") fail("drawer rank is " + JSON.stringify(rank) + ", expected #1");
else pass("drawer shows gap #1");
const bestGap = DATA.gaps[0];
/* The page renders coordinates through signed(), which prefixes an explicit
   "+" for non-negative values so a pair always shows two signs. */
const signed = (value) => (Number(value) >= 0 ? "+" : "") + Number(value).toFixed(2);
const expectedCoords = "(" + signed(bestGap.x) + ", " + signed(bestGap.y) + ")";
if (coords !== expectedCoords)
  fail("drawer coords are " + JSON.stringify(coords) + ", payload says " + expectedCoords);
else pass("drawer coordinates match the #1 point (" + coords + ")");
if (!promptText.includes(bestGap.filename)) fail("prompt does not name its evidence file " + bestGap.filename);
else if (!promptText.includes("Rules (binding)")) fail("prompt is missing the template body");
else pass("prompt rendered with the evidence filename (" + promptText.length + " chars)");
if (stats.length < 12) fail("stats grid rendered only " + stats.length + " text nodes");
else pass("stats grid rendered (" + stats.length + " text nodes)");
if (anchors.length !== 5) fail("rendered " + anchors.length + " anchor links, expected 5");
else pass("5 anchor paper links rendered");

/* The regime mixture renders three Eq. 6 views per regime. A payload key change
   once left every column blank, which looks like a styling issue and is easy to
   miss, so the numbers themselves are asserted. */
const mixRows = documentStub.getElementById("mixture").all().filter((n) => n.className === "mix-row");
if (mixRows.length !== DATA.regimes.length)
  fail("mixture rendered " + mixRows.length + " rows, expected " + DATA.regimes.length);
else {
  const blank = mixRows.filter((row) => {
    const values = row.all().filter((n) => n.className === "vl" || n.className === "ad");
    return values.length !== 3 || values.some((v) => !/^\d+\.\d{3}$/.test(String(v.textContent)));
  });
  if (blank.length) fail("mixture rows have missing or non-numeric values");
  else pass("regime mixture renders 3 numeric views for all " + mixRows.length + " regimes");
}
if (options.length !== DATA.gaps.length)
  fail("gap dropdown has " + options.length + " options, payload has " + DATA.gaps.length);
else pass("gap dropdown has all " + DATA.gaps.length + " gaps");

const radarCall = plotlyCalls.find((c) => c.traces && c.traces[0] && c.traces[0].type === "scatterpolar");
if (!radarCall) fail("polar radar chart was never created");
else {
  const trace = radarCall.traces[0];
  if (trace.r.length !== 5) fail("radar trace has " + trace.r.length + " radii, expected 5 (closed)");
  else if (Math.abs(trace.r[0] - trace.r[4]) > 1e-12) fail("radar polygon is not closed");
  else pass("4-axis radar closed at " + trace.r.slice(0, 4).map((v) => v.toFixed(2)).join("/"));
}

// Interaction: layer switch, gap stepping, click handler, clipboard.
try {
  layerButtons[2].dispatch("click"); // W
  const last = plotlyCalls.filter((c) => c.kind === "restyle").pop();
  if (!last || !Array.isArray(last.update.visible)) fail("layer button did not restyle visibility");
  else {
    const vis = last.update.visible;
    const visibleFields = vis.filter((v, i) => i < 4 && v).length;
    if (vis.length !== manifoldTraces.length) {
      fail("visibility array has " + vis.length + " entries for " + manifoldTraces.length + " traces");
    } else if (visibleFields !== 1) {
      fail("layer switch left " + visibleFields + " fields visible, expected 1");
    } else if (vis.slice(4).some((v) => v !== true)) {
      fail("layer switch hid a non-field trace (markers would vanish)");
    } else {
      pass("layer toggle: 1 of 4 fields visible, all " + (vis.length - 4) + " marker traces kept");
    }
  }
} catch (error) { fail("layer toggle threw: " + error.message); }

try {
  documentStub.getElementById("next-btn").dispatch("click");
  const nextRank = documentStub.getElementById("gap-rank").textContent;
  const nextPrompt = documentStub.getElementById("prompt").textContent;
  if (nextRank === "#1") fail("next button did not advance the drawer");
  else if (nextPrompt === promptText) fail("next button did not change the prompt");
  else pass("next button advanced the drawer to " + nextRank);
  documentStub.getElementById("prev-btn").dispatch("click");
  if (documentStub.getElementById("gap-rank").textContent !== "#1") fail("prev button did not return to #1");
  else pass("prev button returned to #1");
} catch (error) { fail("gap stepping threw: " + error.message); }

try {
  const clickHandlers = manifoldNode.listeners["plotly_click"] || [];
  if (!clickHandlers.length) fail("no plotly_click handler was attached");
  else {
    clickHandlers[0]({ points: [{ data: { name: "Supported Knowledge Gaps" }, pointNumber: 7 }] });
    const rankAfter = documentStub.getElementById("gap-rank").textContent;
    const optionValue = documentStub.getElementById("gap-select").value;
    if (optionValue !== "7") fail("clicking gap point 7 left select at " + optionValue);
    else pass("clicking a gap point updates the drawer (" + rankAfter + ")");
  }
} catch (error) { fail("click handler threw: " + error.message); }

if (!windowStub.__icons) fail("lucide.createIcons was never called");
else pass("lucide icons requested " + windowStub.__icons + " times");

/* ── external idea overlay ────────────────────────────────────────────────
   The overlay is optional in the page, so it is only asserted when the payload
   carries one: the artifact is a separate file that `src.explore` may not have
   written yet, and the corpus dashboard must build without it. */
const idea = DATA.idea;
if (!idea) {
  pass("no idea overlay in this payload (corpus-only build)");
} else {
  try {
    const ideaTrace = manifoldTraces.find((t) => t.name === "Your idea");
    if (!ideaTrace) fail("no 'Your idea' trace on the manifold");
    else if (Math.abs(ideaTrace.x[0] - idea.x) > 1e-9 || Math.abs(ideaTrace.y[0] - idea.y) > 1e-9) {
      fail("idea trace at (" + ideaTrace.x[0] + ", " + ideaTrace.y[0] + ") but payload says (" + idea.x + ", " + idea.y + ")");
    } else pass("idea trace sits at the payload's point (" + idea.x + ", " + idea.y + ")");

    const ideaHalo = manifoldTraces.filter((t) => t.name === "idea halo");
    if (ideaHalo.length !== 1) fail("expected 1 idea halo trace, got " + ideaHalo.length);
    else pass("idea halo trace present");

    const ideaAnnotation = ((main.layout || {}).annotations || []).filter((a) => a.text === "your idea");
    if (ideaAnnotation.length !== 1) fail("expected 1 'your idea' annotation, got " + ideaAnnotation.length);
    else pass("'your idea' annotation drawn at (" + ideaAnnotation[0].x + ", " + ideaAnnotation[0].y + ")");

    const section = documentStub.getElementById("sec-idea");
    if (section.classList.contains("hidden")) fail("idea card is still hidden");
    else pass("idea card un-hidden");

    const title = documentStub.getElementById("idea-title").textContent;
    if (title !== idea.label) fail("idea card title is \"" + title + "\", payload says \"" + idea.label + "\"");
    else pass("idea card titled \"" + title + "\" · " + documentStub.getElementById("idea-sub").textContent);

    const dideal = documentStub.getElementById("idea-dideal").textContent;
    if (Math.abs(Number(dideal) - idea.ideal_distance) > 5e-4) {
      fail("idea card d_ideal is " + dideal + ", payload says " + idea.ideal_distance);
    } else pass("idea card d_ideal matches the artifact (" + dideal + ")");

    const rows = documentStub.getElementById("idea-objectives").children;
    if (rows.length !== idea.objectives.length) {
      fail("idea card rendered " + rows.length + " objective rows, expected " + idea.objectives.length);
    } else {
      const wrong = rows.filter((row, i) => {
        const cells = row.children;
        return cells.length !== 4 || cells[0].textContent !== idea.objectives[i].name;
      });
      if (wrong.length) fail("objective rows are missing their name/value/percentile cells");
      else pass("idea card renders all " + rows.length + " objectives with values and percentiles");
    }

    /* The idea has no gap rank. What the card must show instead: the gap nearest
       on the map, the gap nearest in objective space, the hypothetical insertion
       slot, and -- the decisive one -- whether any gap dominates it. */
    const front = idea.front || {};
    const frontText = documentStub.getElementById("idea-front").all().map((n) => n.textContent).join(" ");
    if (!front.nearest_chart || !front.nearest_objective) {
      fail("payload carries no front reference for the idea");
    } else if (!frontText.includes("gap #" + front.nearest_chart.rank)) {
      fail("idea card never names the map-nearest gap #" + front.nearest_chart.rank);
    } else if (!frontText.includes("gap #" + front.nearest_objective.rank)) {
      fail("idea card never names the objective-nearest gap #" + front.nearest_objective.rank);
    } else if (front.nearest_chart.rank === front.nearest_objective.rank) {
      fail("the two gap references collapsed to one gap; the distinction is untested");
    } else if (!/hypothetical/.test(frontText)) {
      fail("insertion slot is not labelled hypothetical");
    } else pass(
      "idea card separates map-nearest gap #" + front.nearest_chart.rank +
        " from objective-nearest #" + front.nearest_objective.rank +
        ", insertion #" + front.insertion_rank + " of " + front.insertion_of + " labelled hypothetical"
    );

    if (front.dominated_by) {
      if (!frontText.includes("dominates it")) {
        fail("dominating gap #" + front.dominated_by.rank + " is not stated as dominance");
      } else pass("idea card reports domination by gap #" + front.dominated_by.rank);
    } else if (front.on_front && !/no gap dominates/.test(frontText)) {
      fail("payload says no dominator but the card does not say so");
    } else pass("idea card reports front membership consistently with the payload");

    /* The two neighbour lists are the point of this card: ranking by chart
       distance alone once hid the paper that is rank 1 in text but rank 25 on
       the map. Both orderings must be present, and each must lead with its own
       winner. */
    const groups = documentStub.getElementById("idea-neighbours").children;
    const textTop = idea.neighbours.slice().sort((a, b) => a.cosine_rank - b.cosine_rank)[0];
    const chartTop = idea.neighbours.slice().sort((a, b) => a.chart_rank - b.chart_rank)[0];
    const groupText = (group) => group.all().map((n) => n.textContent).join(" ");
    if (groups.length !== 2) {
      fail("idea card rendered " + groups.length + " neighbour groups, expected 2 (text and map order)");
    } else {
      const heads = groups.map((group) => group.children[0].all().map((n) => n.textContent).join(" "));
      if (!/nearest in text/.test(heads[0]) || !/nearest on the map/.test(heads[1])) {
        fail("neighbour groups are headed \"" + heads.join("\" / \"") + "\"");
      } else pass("neighbour lists are separated: text order, then map order");

      if (!groupText(groups[0]).includes(textTop.title)) {
        fail("text-nearest list omits the text rank 1 paper \"" + textTop.title + "\"");
      } else if (!groupText(groups[0]).includes("text #1")) {
        fail("text-nearest list does not print the text rank");
      } else pass("text-nearest list leads with the text rank 1 paper (map rank " + (textTop.chart_rank + 1) + ")");

      if (!groupText(groups[1]).includes(chartTop.title)) {
        fail("map-nearest list omits the map rank 1 paper \"" + chartTop.title + "\"");
      } else pass("map-nearest list leads with the map rank 1 paper (text rank " + (chartTop.cosine_rank + 1) + ")");
    }

    const noteText = documentStub.getElementById("idea-note").all().map((n) => n.textContent).join(" ");
    const whiteShare = Math.round(idea.white_space_plateau_share * 100) + "%";
    if (idea.white_space_plateau_share > 0 && !noteText.includes(whiteShare)) {
      fail("idea card never prints the W plateau share (" + whiteShare + ")");
    } else pass("idea card prints the W plateau share (" + whiteShare + " of " + idea.candidates + " grid points)");

    /* The numbers are only useful with a reading attached. Every axis line must
       carry its value, its percentile and a sentence about what that means for
       this document, and the readings must not contradict the payload. */
    const readings = (idea.plain && idea.plain.readings) || [];
    const axisText = documentStub.getElementById("idea-legend-objectives").all().map((n) => n.textContent).join(" ");
    const missingAxis = readings.filter((entry) => {
      const row = (idea.objectives || []).find((o) => o.name === entry.name);
      if (!row) return true;
      return !axisText.includes(entry.name) || !axisText.includes(Number(row.value).toFixed(3));
    });
    if (!readings.length) {
      fail("idea card has no per-axis reading (plain_language.objectives missing)");
    } else if (missingAxis.length) {
      fail("axis readings omit the value for: " + missingAxis.map((e) => e.name).join(", "));
    } else if (!/For this document:/.test(axisText)) {
      fail("axis readings state the general meaning but never apply it to this document");
    } else pass("every axis line carries its value, percentile and this-document reading (" + readings.length + " axes)");

    const plain = idea.plain || {};
    const plainText = documentStub.getElementById("idea-plain").all().map((n) => n.textContent).join(" ");
    const plainParagraphs = documentStub.getElementById("idea-plain").children.length;
    if (!/In plain words/.test(plainText)) {
      fail("plain-language block has no headline");
    } else if (!plain.position || !plain.novelty || !plain.front) {
      fail("payload's plain_language is missing position, novelty or front");
    } else if (plainParagraphs !== 4) {
      fail("plain-language block rendered " + plainParagraphs + " paragraphs, expected 4 (headline + 3)");
    } else if (!plainText.includes(plain.front.slice(0, 40))) {
      fail("plain-language block does not print the payload's front sentence");
    } else pass("plain-language block renders headline + position + novelty + front in sentences");

    const novelChips = documentStub.getElementById("idea-terms").children.length;
    if (novelChips !== Math.min(10, (idea.novel_terms || []).length)) {
      fail("idea card rendered " + novelChips + " novel-term chips for " + idea.novel_terms.length + " terms");
    } else pass("idea card renders " + novelChips + " novel-term chips");

    const warnings = documentStub.getElementById("idea-warnings").children.length;
    if (warnings !== (idea.warnings || []).length) {
      fail("idea card rendered " + warnings + " warnings for " + idea.warnings.length + " in the artifact");
    } else pass("idea card renders " + warnings + " placement caveat(s)");

    const legendText = documentStub.getElementById("idea-legend").all().map((n) => n.textContent).join(" ");
    if (!/your idea/.test(legendText)) fail("map legend never names the idea marker");
    else pass("map legend explains the idea glyph");

    const handlers = manifoldNode.listeners["plotly_click"] || [];
    handlers[0]({ points: [{ data: { name: "Your idea" }, pointNumber: 0 }] });
    if (!section.scrolled) fail("clicking the idea marker did not scroll to its card");
    else pass("clicking the idea marker focuses the idea card");

    /* Focusing the idea must not leave a gap number sitting under the idea
       card: the corpus panes are hidden, and the map's selection ring and the
       diamond both follow the idea. */
    const scroll = documentStub.getElementById("drawer-scroll");
    if (!scroll.classList.contains("idea-mode")) {
      fail("focusing the idea did not switch the drawer into idea mode");
    } else {
      const restyles = plotlyCalls.filter((c) => c.kind === "restyle");
      const ring = restyles.filter((c) => c.update.x).pop();
      const ringAt = ring && Array.isArray(ring.update.x[0]) ? ring.update.x[0][0] : null;
      const diamondGrow = restyles.filter((c) => c.update["marker.size"] === 19).pop();
      if (!ring || Math.abs(ringAt - idea.x) > 1e-9) {
        fail("selection ring did not move to the idea point (restyle x = " + JSON.stringify(ring && ring.update.x) + ")");
      } else if (!diamondGrow) {
        fail("focusing the idea never enlarged the diamond marker");
      } else pass(
        "focusing the idea switches the drawer to idea mode, enlarges the diamond, and rings it at (" + ringAt + ")"
      );
    }

    const back = documentStub.getElementById("idea-back");
    if (!back) fail("idea card has no way back to the gaps");
    else {
      back.dispatch("click");
      if (scroll.classList.contains("idea-mode")) fail("back button left the drawer in idea mode");
      else pass("idea card's back button returns the drawer to the selected gap");
    }
  } catch (error) {
    fail("idea overlay assertions threw: " + error.message);
  }
}

}

settle().then(assertions).then(settle).then(() => {
  console.log(process.exitCode ? "\nRESULT: FAIL" : "\nRESULT: PASS");
});
