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

  const papers = traces.filter((t) => t.type === "scatter" && ["SysID", "Sim2Real", "RL"].includes(t.name));
  const totalPapers = papers.reduce((s, t) => s + t.x.length, 0);
  if (totalPapers !== 120) fail("regime traces cover " + totalPapers + " papers, expected 120");
  else pass("3 regime traces cover all 120 papers (" + papers.map((t) => t.name + ":" + t.x.length).join(" "));

  const gaps = traces.find((t) => t.name === "Supported Knowledge Gaps");
  if (!gaps) fail("no 'Supported Knowledge Gaps' trace");
  else if (gaps.x.length !== 33) fail("gap trace has " + gaps.x.length + " points, expected 33");
  else pass("33 clickable gap markers");

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
  if (!top) fail("no #1 compromise trace");
  else if (Math.abs(top.x[0] + 0.1) > 1e-9 || Math.abs(top.y[0] + 0.5) > 1e-9)
    fail("#1 compromise is at (" + top.x[0] + ", " + top.y[0] + "), expected (-0.1, -0.5)");
  else pass("#1 compromise marker at (-0.10, -0.50)");

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
if (coords !== "(-0.10, -0.50)") fail("drawer coords are " + JSON.stringify(coords));
else pass("drawer coordinates match the #1 point");
if (!promptText.includes("evidence_point_-0.10_-0.50.json")) fail("prompt does not name the evidence file");
else if (!promptText.includes("Rules (binding)")) fail("prompt is missing the template body");
else pass("prompt rendered with the evidence filename (" + promptText.length + " chars)");
if (stats.length < 12) fail("stats grid rendered only " + stats.length + " text nodes");
else pass("stats grid rendered (" + stats.length + " text nodes)");
if (anchors.length !== 5) fail("rendered " + anchors.length + " anchor links, expected 5");
else pass("5 anchor paper links rendered");
if (options.length !== 33) fail("gap dropdown has " + options.length + " options, expected 33");
else pass("gap dropdown has all 33 gaps");

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

}

settle().then(assertions).then(settle).then(() => {
  console.log(process.exitCode ? "\nRESULT: FAIL" : "\nRESULT: PASS");
});
