"use strict";
const $ = (id) => document.getElementById(id);
const state = { graph: null, tab: "detail", selected: null, compare: null, ai: [] };

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
const short = (id) => (id || "").slice(0, 12);
const announce = (msg) => { $("live").textContent = msg; };

async function api(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

function kv(rows) {
  return h("table", {}, rows.map(([k, v]) => h("tr", {}, h("th", { scope: "row" }, k), h("td", { class: "mono" }, v))));
}

function nodeById(id) {
  const g = state.graph;
  const c = g.commits.find((n) => n.id === id);
  if (c) return { type: "commit", data: c };
  const e = g.experiments.find((n) => n.id === id);
  return e ? { type: "experiment", data: e } : null;
}

function layout(graph) {
  const byGen = new Map();
  graph.commits.forEach((c) => { (byGen.get(c.generation) || byGen.set(c.generation, []).get(c.generation)).push(c); });
  const lane = new Map();
  const pos = new Map();
  let maxLane = 0;
  [...byGen.keys()].sort((a, b) => a - b).forEach((gen) => {
    const used = new Set();
    byGen.get(gen).sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id)).forEach((c) => {
      let l = c.parents.length ? lane.get(c.parents[0]) : undefined;
      if (l === undefined || used.has(l)) { l = 0; while (used.has(l)) l++; }
      used.add(l); lane.set(c.id, l); maxLane = Math.max(maxLane, l);
      pos.set(c.id, { x: 60 + l * 150, y: 50 + (gen - 1) * 90 });
    });
  });
  const perCommit = new Map();
  graph.experiments.forEach((e) => {
    const k = perCommit.get(e.commit) || 0;
    perCommit.set(e.commit, k + 1);
    const base = pos.get(e.commit) || { x: 0, y: 0 };
    pos.set(e.id, { x: 60 + (maxLane + 1) * 150 + k * 70, y: base.y });
  });
  return pos;
}

function visible(n, type) {
  const q = $("q").value.trim().toLowerCase();
  const branch = $("branch").value;
  if (type === "experiment") {
    if (!$("showExp").checked) return false;
    if ($("status").value && n.status !== $("status").value) return false;
    return !q || [n.id, n.status, n.code_revision || ""].some((s) => s.toLowerCase().includes(q));
  }
  if (branch && !n.branches.includes(branch)) return false;
  return !q || [n.id, n.message, ...n.branches, ...n.tags].some((s) => s.toLowerCase().includes(q));
}

function drawGraph() {
  const graph = state.graph;
  const svg = d3.select("#graph");
  svg.selectAll("*").remove();
  const stateBox = $("graph-state");
  stateBox.hidden = graph.commits.length > 0;
  if (!graph.commits.length) { stateBox.textContent = "No versions yet. Run datavault add and datavault commit."; return; }
  const pos = layout(graph);
  const root = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (ev) => root.attr("transform", ev.transform)));
  const ids = new Set([...graph.commits.map((c) => c.id), ...graph.experiments.map((e) => e.id)]);
  root.selectAll("path.edge").data(graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target))).join("path")
    .attr("class", (e) => "edge " + e.kind)
    .attr("d", (e) => {
      const a = pos.get(e.source), b = pos.get(e.target);
      const my = (a.y + b.y) / 2;
      return e.kind === "parent" ? `M${a.x},${a.y} C${a.x},${my} ${b.x},${my} ${b.x},${b.y}` : `M${a.x},${a.y} L${b.x},${b.y}`;
    });
  const items = [
    ...graph.commits.map((d) => ({ type: "commit", d })),
    ...graph.experiments.map((d) => ({ type: "experiment", d })),
  ];
  const g = root.selectAll("g.node").data(items).join("g")
    .attr("class", "node").attr("tabindex", 0).attr("role", "button")
    .attr("transform", (n) => `translate(${pos.get(n.d.id).x},${pos.get(n.d.id).y})`)
    .attr("aria-label", (n) => n.type === "commit"
      ? `Version ${short(n.d.id)}: ${n.d.message}` : `Experiment ${short(n.d.id)}, ${n.d.status}${n.d.accuracy != null ? ", accuracy " + n.d.accuracy : ""}`)
    .on("click", (ev, n) => choose(n.d.id, ev.shiftKey))
    .on("keydown", (ev, n) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); choose(n.d.id, ev.shiftKey); }
      if (ev.key === "c") { state.compare = n.d.id; refresh(); }
    });
  g.filter((n) => n.type === "commit").append("circle").attr("r", 13)
    .attr("fill", (n) => (n.d.id === state.graph.head.commit ? "var(--accent)" : "var(--bg3)"))
    .attr("stroke", (n) => (n.d.quality && n.d.quality.errors ? "var(--red)" : "var(--border)")).attr("stroke-width", 2);
  g.filter((n) => n.type === "experiment").append("rect").attr("x", -10).attr("y", -10).attr("width", 20).attr("height", 20)
    .attr("transform", "rotate(45)")
    .attr("fill", (n) => (n.d.status === "failed" ? "var(--red)" : "var(--green)")).attr("stroke", "var(--border)").attr("stroke-width", 2);
  g.append("text").attr("class", "lbl").attr("y", 28).attr("text-anchor", "middle").text((n) => short(n.d.id).slice(0, 7));
  g.filter((n) => n.type === "commit").append("text").attr("class", "sub").attr("y", 40).attr("text-anchor", "middle")
    .text((n) => [...n.d.branches, ...n.d.tags.map((t) => "#" + t)].join(" ").slice(0, 22));
  g.filter((n) => n.type === "experiment").append("text").attr("class", "sub").attr("y", 40).attr("text-anchor", "middle")
    .text((n) => (n.d.accuracy != null ? "acc " + n.d.accuracy : "failed"));
  g.each(function (n) {
    const ok = visible(n.d, n.type);
    const el = d3.select(this);
    el.classed("dim", !ok).classed("sel", n.d.id === state.selected).classed("cmp", n.d.id === state.compare);
  });
}

function choose(id, asCompare) {
  if (asCompare) state.compare = id; else state.selected = id;
  if (asCompare) state.tab = "compare";
  refresh();
  announce(`${asCompare ? "Compare target" : "Selected"} ${short(id)}`);
}

function evidenceLinks(item, evidence) {
  return item.evidence.map((id) => {
    const fact = evidence.find((e) => e.id === id);
    return h("button", { class: "ev", type: "button", title: fact ? fact.fact : "unknown evidence", onclick: (ev) => {
      const note = ev.target.parentElement.querySelector(".fact") || h("div", { class: "fact muted mono" });
      note.textContent = `${id}: ${fact ? fact.fact : "unknown"}`;
      ev.target.parentElement.append(note);
    } }, id + " ");
  });
}

function renderAnalysis(record) {
  const out = [h("p", { class: "muted" }, `${record.task} · ${record.usage.provider}/${record.usage.model} · ${record.created_at}`),
    h("p", { class: "warn" }, record.note)];
  for (const [name, items] of Object.entries(record.analysis.sections)) {
    out.push(h("h2", {}, name.replaceAll("_", " ")));
    out.push(h("ul", {}, items.map((i) => h("li", {},
      i.status !== "note" && h("span", { class: "badge " + (i.status === "grounded" ? "ok" : i.status === "hypothesis" ? "hyp" : "bad") }, i.status),
      i.text, " ", evidenceLinks(i, record.evidence), i.problems.map((p) => h("div", { class: "bad" }, "! " + p))))));
  }
  return out;
}

async function renderDetail(p) {
  const sel = state.selected && nodeById(state.selected);
  if (!sel) return p.append(h("div", { class: "state" }, "Select a version or experiment."));
  if (sel.type === "commit") {
    const c = await api("/api/commit/" + sel.data.id);
    const exps = state.graph.experiments.filter((e) => e.commit === c.id);
    p.append(h("h2", {}, "Version"), kv([["id", c.id], ["content id", c.content_id], ["message", c.message], ["author", c.author],
      ["created", c.created_at], ["branches", sel.data.branches.join(", ") || "-"], ["tags", sel.data.tags.join(", ") || "-"],
      ["parents", c.parents.map(short).join(", ") || "root"]]));
    p.append(h("h2", {}, "Files"), h("table", {}, Object.entries(c.files).map(([f, e]) =>
      h("tr", {}, h("td", { class: "mono" }, f), h("td", { class: "mono" }, short(e.sha256)), h("td", {}, e.size + " B")))));
    if (c.preprocessing) p.append(h("h2", {}, "Preprocessing"), h("div", { class: "mono" },
      Object.entries(c.preprocessing).map(([k, v]) => h("span", { class: "badge " + (v ? "ok" : "muted") }, k))));
    if (c.quality_report) p.append(h("h2", {}, "Data quality"), h("div", {}, `${c.quality_report.errors} errors, ${c.quality_report.warnings} warnings`),
      h("ul", {}, c.quality_report.issues.map((i) => h("li", { class: i.severity === "error" ? "bad" : "warn" }, `${i.path}: ${i.message}`))));
    p.append(h("h2", {}, "Experiments on this version"), exps.length
      ? h("ul", {}, exps.map((e) => h("li", {}, h("button", { class: "ev", type: "button", onclick: () => choose(e.id) }, short(e.id)),
        ` ${e.status} ${e.accuracy != null ? "acc " + e.accuracy : ""}`))) : h("div", { class: "muted" }, "none"));
    p.append(h("button", { type: "button", onclick: () => { state.compare = c.id; state.tab = "compare"; refresh(); } }, "Compare with this version"));
    return;
  }
  const e = await api("/api/experiment/" + sel.data.id);
  p.append(h("h2", {}, "Experiment"), kv([["id", e.id], ["status", e.status], ["created", e.created_at], ["config hash", short(e.config_hash)],
    ["parent", e.parent_experiment ? short(e.parent_experiment) : "-"]]));
  p.append(h("h2", {}, "Dataset"), h("button", { class: "ev", type: "button", onclick: () => choose(e.dataset.commit) }, short(e.dataset.commit)),
    h("div", { class: "muted mono" }, "content " + short(e.dataset.content_id)));
  p.append(h("h2", {}, "Configuration"), kv(Object.entries(e.config).map(([k, v]) => [k, JSON.stringify(v)])));
  p.append(h("h2", {}, "Environment"), kv(Object.entries(e.environment).map(([k, v]) => [k, v == null ? "-" : String(v)])));
  if (e.error) p.append(h("h2", {}, "Error"), h("div", { class: "bad" }, e.error));
  if (e.metrics) {
    const t = e.metrics.test;
    p.append(h("h2", {}, "Test metrics"), kv([["accuracy", t.accuracy], ["macro F1", t.macro_f1], ["test rows", t.n], ["train accuracy", e.metrics.train.accuracy]]));
    const classes = Object.keys(t.confusion_matrix);
    p.append(h("h2", {}, "Confusion matrix (rows = true)"), h("table", {}, h("tr", {}, h("th", {}, ""), classes.map((c) => h("th", {}, c))),
      classes.map((r) => h("tr", {}, h("th", { scope: "row" }, r), classes.map((c) => h("td", {}, t.confusion_matrix[r][c]))))));
  }
  p.append(h("h2", {}, "Logs"), h("pre", { class: "mono" }, e.logs.join("\n")));
}

async function renderCompare(p) {
  const a = state.selected, b = state.compare;
  if (!a || !b) return p.append(h("div", { class: "state" }, "Select one node, then Shift+click (or press c) on another of the same kind."));
  const na = nodeById(a), nb = nodeById(b);
  if (!na || !nb || na.type !== nb.type) return p.append(h("div", { class: "state" }, "Compare two versions or two experiments."));
  if (na.type === "commit") {
    const d = await api(`/api/diff?a=${a}&b=${b}`);
    p.append(h("h2", {}, `${short(a)} → ${short(b)}`), h("div", {}, d.identical_content ? "Identical content" : "Content differs"));
    const f = d.files;
    p.append(h("ul", {}, [...f.added.map((x) => h("li", { class: "ok" }, "+ " + x.path)), ...f.removed.map((x) => h("li", { class: "bad" }, "- " + x.path)),
      ...f.modified.map((x) => h("li", {}, `~ ${x.path} ${x.size_a} → ${x.size_b} B`))]));
    for (const [k, v] of Object.entries(d.preprocessing)) p.append(h("div", {}, `preprocessing ${k}: ${v.a} → ${v.b}`));
    for (const [path, ch] of Object.entries(d.stats)) {
      p.append(h("h2", {}, path), h("pre", { class: "mono" }, JSON.stringify(ch, null, 2)));
    }
    return;
  }
  const c = await api(`/api/compare?a=${a}&b=${b}`);
  p.append(h("h2", {}, `${short(a)} → ${short(b)}`), h("div", {}, c.same_dataset_content ? "Same dataset content" : "Different dataset content"));
  p.append(h("h2", {}, "Settings changed"), Object.keys(c.config_changes).length
    ? kv(Object.entries(c.config_changes).map(([k, v]) => [k, `${JSON.stringify(v.a)} → ${JSON.stringify(v.b)}`])) : h("div", { class: "muted" }, "none"));
  p.append(h("h2", {}, "Metrics"), kv(Object.entries(c.metrics).map(([k, v]) => [k, `${v.a} → ${v.b}${v.delta != null ? " (" + (v.delta > 0 ? "+" : "") + v.delta + ")" : ""}`])));
  c.notes.forEach((n) => p.append(h("div", { class: "warn" }, n)));
}

function renderMetrics(p) {
  const exps = state.graph.experiments.filter((e) => e.accuracy != null);
  if (!exps.length) return p.append(h("div", { class: "state" }, "No completed experiments."));
  const W = 420, H = 220, m = { l: 36, r: 10, t: 10, b: 28 };
  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("role", "img")
    .attr("aria-label", "Accuracy and macro F1 per experiment");
  const x = d3.scalePoint().domain(exps.map((e) => e.id)).range([m.l, W - m.r]).padding(0.5);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).attr("color", "var(--muted)");
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat((id) => short(id).slice(0, 4))).attr("color", "var(--muted)");
  [["accuracy", "var(--accent)"], ["macro_f1", "var(--green)"]].forEach(([key, color]) => {
    svg.append("path").datum(exps).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2)
      .attr("d", d3.line().x((e) => x(e.id)).y((e) => y(e[key])));
    svg.selectAll(null).data(exps).join("circle").attr("cx", (e) => x(e.id)).attr("cy", (e) => y(e[key])).attr("r", 4).attr("fill", color);
  });
  p.append(h("h2", {}, "Test metrics by experiment (oldest to newest)"), svg.node(),
    h("div", { class: "muted" }, "blue: accuracy, green: macro F1"),
    h("table", {}, h("tr", {}, ["experiment", "dataset", "accuracy", "macro F1"].map((t) => h("th", {}, t))),
      exps.map((e) => h("tr", {}, h("td", { class: "mono" }, short(e.id)), h("td", { class: "mono" }, short(e.commit)), h("td", {}, e.accuracy), h("td", {}, e.macro_f1)))));
}

async function renderAI(p) {
  const list = await api("/api/ai");
  if (!list.length) return p.append(h("div", { class: "state" }, "No saved AI analyses. Run datavault ai dataset A B, then reload."));
  const sel = state.selected;
  const relevant = list.filter((a) => !sel || Object.values(a.inputs).some((v) => v === sel));
  const shown = relevant.length ? relevant : list;
  if (sel && !relevant.length) p.append(h("div", { class: "muted" }, "None for the selection; showing all."));
  const holder = h("div", {});
  p.append(h("h2", {}, "Saved analyses"), h("ul", {}, shown.map((a) => h("li", {},
    h("button", { class: "ev", type: "button", onclick: async () => {
      holder.replaceChildren(...renderAnalysis(await api("/api/ai/" + a.file)));
    } }, `${a.task} ${a.created_at}`), ` · ${a.stats.grounded} grounded, ${a.stats.hypothesis} hyp, ${a.stats.unsupported} unsupported`))), holder);
}

function renderList(p) {
  const g = state.graph;
  p.append(h("h2", {}, "Versions"), h("ul", {}, g.commits.filter((c) => visible(c, "commit")).map((c) =>
    h("li", {}, h("button", { class: "ev", type: "button", onclick: () => choose(c.id) }, short(c.id)), ` ${c.message}`))));
  p.append(h("h2", {}, "Experiments"), h("ul", {}, g.experiments.filter((e) => visible(e, "experiment")).map((e) =>
    h("li", {}, h("button", { class: "ev", type: "button", onclick: () => choose(e.id) }, short(e.id)), ` ${e.status} ${e.accuracy ?? ""}`))));
}

async function renderPanel() {
  const p = $("panel");
  p.replaceChildren();
  try {
    if (state.tab === "detail") await renderDetail(p);
    else if (state.tab === "compare") await renderCompare(p);
    else if (state.tab === "metrics") renderMetrics(p);
    else if (state.tab === "ai") await renderAI(p);
    else renderList(p);
  } catch (err) {
    p.replaceChildren(h("div", { class: "state bad", role: "alert" }, "Could not load: " + err.message));
  }
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === state.tab)));
}

function refresh() { drawGraph(); renderPanel(); }

async function load() {
  $("panel").replaceChildren(h("div", { class: "state" }, "Loading…"));
  try {
    state.graph = await api("/api/graph");
  } catch (err) {
    $("panel").replaceChildren(h("div", { class: "state bad", role: "alert" }, "Could not load repository: " + err.message));
    return;
  }
  const head = state.graph.head;
  $("head").textContent = head.branch ? `${head.branch} @ ${short(head.commit)}` : `detached @ ${short(head.commit)}`;
  const sel = $("branch");
  [...new Set(state.graph.commits.flatMap((c) => c.branches))].sort().forEach((b) => sel.append(h("option", { value: b }, b)));
  if (head.commit) state.selected = head.commit;
  refresh();
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => { state.tab = b.dataset.tab; renderPanel(); }));
["q", "branch", "showExp", "status"].forEach((id) => $(id).addEventListener("input", drawGraph));
$("verify").addEventListener("click", async () => {
  try {
    const r = await api("/api/verify");
    announce(r.ok ? "Integrity check passed" : `Integrity check found ${r.findings.length} problems`);
    const p = $("panel");
    state.tab = "detail"; await renderPanel();
    p.prepend(h("div", { class: r.ok ? "ok" : "bad", role: "status" },
      `Structural check: ${r.commits_checked} commits, ${r.findings.length} findings`), h("ul", {}, r.findings.map((f) => h("li", { class: "bad" }, `${f.code}: ${f.subject} ${f.message}`))));
  } catch (err) { announce("Verification failed: " + err.message); }
});
load();
