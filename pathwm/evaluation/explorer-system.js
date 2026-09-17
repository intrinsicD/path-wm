// Non-neural system navigation. Embedded in the standalone HTML by write_explorer.
function moduleLabel(path) {
  if (!path)
    return context === "world"
      ? "World State neural model"
      : "Main training model";
  const label = D.metadata.roles?.[path]?.label;
  return label ? `${M[path].name} · ${label}` : M[path].name;
}
function systemGo(name, selection = null) {
  go({
    context: "main",
    scope: "",
    systemPage: name,
    systemSelection: selection,
  });
}
function systemNavigation(root) {
  if (!bundle.system) return;
  const nav = document.createElement("div");
  nav.className = "system-nav";
  const entries = [
    ["System overview", () => systemGo("overview")],
    ["Knowledge graph", () => systemGo("store")],
    ["WorldSession · runtime calls", () => systemGo("world")],
    ["World State · neural layers", () => go({ context: "world", scope: "" })],
    ["Record schemas & interfaces", () => systemGo("interfaces")],
    [
      "Main learner · agent & teacher",
      () => go({ context: "main", scope: "" }),
    ],
  ];
  for (const [label, action] of entries) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = action;
    nav.append(b);
  }
  root.append(nav);
}
function systemCrumbs(label) {
  const c = $("crumbs");
  c.innerHTML = "";
  const b = document.createElement("button");
  b.textContent = "←";
  b.className = "back";
  b.setAttribute("aria-label", "Go back");
  b.onclick = back;
  b.disabled = !history.length && systemPage === "overview";
  c.append(b);
  const home = document.createElement("button");
  home.textContent = "System";
  home.onclick = () => systemGo("overview");
  c.append(home);
  if (systemPage !== "overview") {
    const world = document.createElement("button");
    world.textContent = "World State";
    world.onclick = () => systemGo("world");
    c.append(world);
  }
  const current = document.createElement("span");
  current.textContent = " / " + label;
  current.className = "fine";
  c.append(current);
}
function sourceButton(parent, source, title = "Source") {
  if (!source) return;
  const b = document.createElement("button");
  b.textContent = "Read captured " + title;
  b.onclick = () => showSource({ type: title, source });
  parent.append(b);
}
function systemCard(parent, { title, eyebrow, text, action, style = "" }) {
  const b = document.createElement("button");
  b.className = "system-card " + style;
  b.innerHTML = `<span class="eyebrow">${esc(eyebrow)}</span><b>${esc(title)}</b><p>${esc(text)}</p><span class="fine">Click to explore →</span>`;
  b.onclick = action;
  parent.append(b);
}
function worldComponent(key) {
  if (key === "store") {
    systemGo("store");
    return;
  }
  const component = bundle.system.components[key];
  if (component.neural_path) {
    go({ context: "world", scope: component.neural_path });
    return;
  }
  systemGo("component", key);
}
function describeCall(edge) {
  $("layout").classList.add("show-inspector");
  const s = bundle.system;
  $("inspector").innerHTML =
    `<div class="eyebrow">Observed runtime call</div><h2>${esc(s.components[edge.source].name)} → ${esc(s.components[edge.target].name)}</h2><p>${fmt(edge.count)} calls during the synthetic session.</p><pre>${esc(JSON.stringify(edge.methods, null, 2))}</pre><p>This arrow shows which component called another. It is not an autograd edge or a claim that every tensor flows in this direction.</p>`;
  sourceButton(
    $("inspector"),
    s.components[edge.target].source,
    s.components[edge.target].type,
  );
}
function worldCallGraph() {
  const s = bundle.system;
  const nodes = Object.entries(s.components).map(([id, c]) => ({
    id,
    label: id === "store" ? "Knowledge graph" : c.name,
    type: c.type,
    detail: fmt(c.calls) + " recorded calls",
    hint: c.neural_path
      ? "click for layers & weights"
      : id === "store"
        ? "entities · relations · evidence"
        : "click for interface & source",
    role: id === "store" ? "memory" : role(id, c.type),
    active: c.calls > 0,
    onClick: () => worldComponent(id),
  }));
  const edges = s.edges.map((e) => ({
    ...e,
    samples: Object.entries(e.methods).map(
      ([name, count]) => `${name} × ${count}`,
    ),
    inspect: () => describeCall(e),
  }));
  // Position the current components without inventing call edges. Spread the
  // session's fan-out horizontally; reserve space for newly added components.
  const slots = {
    caller: [0, 1],
    encoder: [1, 0],
    session: [1, 1],
    retrieval: [2, 0],
    binding: [2, 1],
    context: [2, 2],
    updater: [3, 0],
    scorer: [3, 1],
    agent: [3, 2],
    store: [4, 1],
    predictor: [0, 2],
    state_head: [1, 2],
    readout: [4, 2],
  };
  let extra = 0;
  for (const n of nodes) {
    const [column, row] = slots[n.id] ?? [
      extra % 5,
      3 + Math.floor(extra++ / 5),
    ];
    Object.assign(n, {
      column,
      row,
      x: 24 + column * 260,
      y: 68 + row * 156,
      w: 180,
      h: 94,
    });
  }
  const rows = 1 + Math.max(...nodes.map((n) => n.row));
  const layout = {
    width: Math.max(...nodes.map((n) => n.x + n.w + 48)),
    height: 68 + (rows - 1) * 156 + 162,
    top: 68,
    stepY: 156,
    rows,
    headers: [],
  };
  graph({
    nodes,
    edges,
    layout,
    label: "Observed WorldSession component calls",
  });
}
function recordId(record, index) {
  return record.id ?? record.event?.id ?? String(index);
}
function storeRecord(category, id) {
  systemGo("record", { category, id });
}
function graphRelations() {
  const s = bundle.system.store,
    components = new Map(s.components.map((c) => [c.id, c]));
  const owner = (endpoint) =>
    endpoint.kind === "entity"
      ? endpoint.ref
      : components.get(endpoint.ref)?.entity_id;
  const nodes = s.entities.map((e) => ({
    id: e.id,
    label: e.label || e.id,
    type: e.kind || "Entity",
    detail: `${s.components.filter((c) => c.entity_id === e.id).length} components`,
    hint: "click for records & evidence",
    role: "memory",
    active: true,
    onClick: () => storeRecord("entities", e.id),
  }));
  const known = new Set(nodes.map((n) => n.id)),
    edges = [];
  for (const r of s.relations) {
    const source = owner(r.source),
      target = owner(r.target);
    if (known.has(source) && known.has(target))
      edges.push({
        source,
        target,
        count: 1,
        samples: [`${r.type} · ${r.id}`],
        inspect: () => storeRecord("relations", r.id),
      });
  }
  return { nodes, edges, label: "WorldStore entities and stored relations" };
}
function recordRows(parent, category, records) {
  const filter = document.createElement("input");
  filter.className = "search";
  filter.placeholder = `Find ${category} by label, ID or content…`;
  filter.setAttribute("aria-label", `Find ${category}`);
  parent.append(filter);
  const table = document.createElement("table");
  table.className = "system-table";
  table.innerHTML = "<thead><tr><th>Record</th><th>Contents</th></tr></thead>";
  const body = document.createElement("tbody");
  table.append(body);
  parent.append(table);
  const populate = () => {
    body.innerHTML = "";
    const query = filter.value.toLowerCase();
    records.forEach((r, i) => {
      if (query && !JSON.stringify(r).toLowerCase().includes(query)) return;
      const id = recordId(r, i),
        row = document.createElement("tr"),
        name = document.createElement("td"),
        detail = document.createElement("td"),
        b = document.createElement("button");
      b.textContent = r.label || r.name || r.type || id;
      b.onclick = () => storeRecord(category, id);
      name.append(b);
      detail.textContent =
        category === "components"
          ? `${id} · owner ${r.entity_id} · [${r.shape.join(" × ")}] · ${r.values.length} exact values`
          : category === "relations"
            ? `${id} · ${r.source.ref} → ${r.target.ref}${r.active ? "" : " · inactive"}`
            : category === "evidence"
              ? `${id} · ${r.source} · ${r.modality}`
              : id;
      row.append(name, detail);
      body.append(row);
    });
    if (!body.childElementCount) {
      const row = document.createElement("tr");
      row.innerHTML =
        '<td colspan="2">No matching records in this diagnostic snapshot.</td>';
      body.append(row);
    }
  };
  filter.oninput = populate;
  populate();
}
function renderStore(parent) {
  const s = bundle.system.store;
  parent.innerHTML = `<h2>Knowledge graph · WorldStore</h2><p><b>Fresh synthetic diagnostic session</b> · revision ${s.revision}. These are actual records produced by the existing foundation exercise, not loaded personal or trained knowledge. Components retain their exact values and evidence references.</p>`;
  const diagram = document.createElement("div");
  diagram.className = "canvas store-diagram";
  parent.append(diagram);
  const data = graphRelations();
  if (data.nodes.length) graph(data, diagram);
  else
    diagram.innerHTML =
      "<p>No entities were created in this execution. Record schemas remain inspectable.</p>";
  if (!data.edges.length) {
    const note = document.createElement("p");
    note.textContent =
      "No relations were created in this execution. The Relations tab and current Relation schema remain available; no relationships are invented for the picture.";
    parent.append(note);
  }
  const tabs = document.createElement("div");
  tabs.className = "system-tabs";
  parent.append(tabs);
  const records = document.createElement("div");
  parent.append(records);
  const show = (category) => {
    records.innerHTML = "";
    recordRows(records, category, s[category]);
  };
  for (const category of [
    "entities",
    "components",
    "relations",
    "evidence",
    "events",
  ]) {
    const b = document.createElement("button");
    b.textContent = `${category[0].toUpperCase() + category.slice(1)} (${s[category].length})`;
    b.onclick = () => show(category);
    tabs.append(b);
  }
  const save = document.createElement("button");
  save.textContent = "Export exact store snapshot";
  save.onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([JSON.stringify(s.snapshot, null, 2)], {
        type: "application/json",
      }),
    );
    a.download = "world-snapshot.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  tabs.append(save);
  show("entities");
}
function renderRecord(parent) {
  const { category, id } = systemSelection,
    store = bundle.system.store;
  const record = store[category]?.find((r, i) => recordId(r, i) === id);
  if (!record) {
    parent.textContent = "This record is absent from this snapshot.";
    return;
  }
  parent.innerHTML = `<div class="eyebrow">${esc(category)}</div><h2>${esc(record.label || record.name || record.type || id)}</h2><p>${esc(id)} · stored data, not trainable parameters</p><pre>${esc(JSON.stringify(record, null, 2))}</pre>`;
  const links = document.createElement("div");
  links.className = "system-tabs";
  parent.append(links);
  const link = (category, id, text) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.onclick = () => storeRecord(category, id);
    links.append(b);
  };
  if (category === "entities") {
    const related = store.components.filter((c) => c.entity_id === id);
    for (const c of related)
      link(
        "components",
        c.id,
        `${c.name} · ${c.id}${c.active ? "" : " · inactive"}`,
      );
    const componentIds = new Set(related.map((c) => c.id));
    for (const r of store.relations)
      if (
        [r.source.ref, r.target.ref].some(
          (ref) => ref === id || componentIds.has(ref),
        )
      )
        link("relations", r.id, `${r.type} · ${r.id}`);
    const canonical = store.canonical_ids[id];
    if (canonical !== id)
      link("entities", canonical, "Canonical entity " + canonical);
  }
  if (record.entity_id) link("entities", record.entity_id, "Owning entity");
  if (record.event_id)
    link("events", record.event_id, "Event " + record.event_id);
  for (const proof of record.evidence || [])
    link("evidence", proof, "Evidence " + proof);
  for (const parentId of record.parents || [])
    link("components", parentId, "Parent component " + parentId);
  if (category === "relations")
    for (const endpoint of [record.source, record.target])
      link(
        endpoint.kind === "entity" ? "entities" : "components",
        endpoint.ref,
        endpoint.kind + " " + endpoint.ref,
      );
  const schema = {
    entities: "Entity",
    components: "Component",
    relations: "Relation",
    evidence: "Evidence",
    events: "Event",
  }[category];
  const b = document.createElement("button");
  b.textContent = "Inspect " + schema + " schema";
  b.onclick = () => systemGo("schema", schema);
  links.append(b);
}
function renderSystem() {
  const s = bundle.system;
  const titles = {
    overview: "System overview",
    world: "WorldSession · observed runtime calls",
    store: "Knowledge graph",
    component: "Runtime component",
    record: "Graph record",
    schema: "Record schema",
    interfaces: "Schemas & optional interfaces",
  };
  const label = titles[systemPage] || systemPage;
  systemCrumbs(label);
  $("title").textContent = label;
  $("subtitle").textContent =
    "Main categorical model and optional World State foundation are separate existing recipe configurations.";
  $("stats").innerHTML =
    `<div><b>${s.store.entities.length}</b><span>GRAPH ENTITIES</span></div><div><b>${s.store.relations.length}</b><span>RELATIONS</span></div>`;
  $("pager").hidden = true;
  $("pager").innerHTML = "";
  $("inspector").innerHTML =
    '<div class="eyebrow">System scope</div><p>Inspect the main neural learner, the optional World State assembly, or its stored records. Neural data/gradient edges and runtime call edges are labelled separately.</p>';
  $("caption").textContent =
    "World State records come from a fresh synthetic exercise. Persistent storage has no autograd graph; functional neural modules have their own captured weight gradients.";
  if (systemPage === "world") {
    worldCallGraph();
    $("caption").textContent =
      "Arrows show observed Python component calls in the diagnostic WorldSession execution. Click a neural component for exact layers, tensor flow and gradients; click Knowledge graph for stored records.";
    return;
  }
  const page = document.createElement("div");
  page.className = "system-page";
  $("canvas").replaceChildren(page);
  if (systemPage === "overview") {
    page.innerHTML =
      '<h2>Neural model, training teacher, and persistent knowledge</h2><p>The two assemblies below are constructed by their own current recipes. The main learner and the optional World State foundation have different configurations; their weights and runtime evidence remain distinct.</p><div class="eyebrow">Main categorical recipe · experiments/multimodal.py</div>';
    const cards = document.createElement("div");
    cards.className = "system-cards";
    page.append(cards);
    for (const path of ["agent", "target"]) {
      const m = bundle.modules[path],
        r = bundle.metadata.roles[path];
      systemCard(cards, {
        title: r.label,
        eyebrow: `${path} · ${fmt(m.parameters)} parameters`,
        text: r.description,
        style: path === "target" ? "teacher-card" : "",
        action: () => go({ context: "main", scope: path }),
      });
    }
    const rule = document.createElement("p");
    rule.textContent =
      bundle.metadata.roles.target.formula +
      ". The capture runs no optimizer or EMA update.";
    page.append(rule);
    sourceButton(
      page,
      bundle.metadata.roles.target.source,
      "EMA update source",
    );
    const h = document.createElement("h2");
    h.textContent = "World State foundation · optional integration";
    h.style.marginTop = "30px";
    page.append(h);
    const detail = document.createElement("p");
    detail.textContent =
      "experiments/world_state.py constructs its own agent, learned binding/update/context components, and WorldSession. Its diagnostic runtime execution supplies the graph below.";
    page.append(detail);
    const world = document.createElement("div");
    world.className = "system-cards";
    page.append(world);
    systemCard(world, {
      title: "Knowledge graph",
      eyebrow: "WorldStore · persistent records",
      text: `${s.store.entities.length} entities · ${s.store.components.length} components · ${s.store.relations.length} relations · ${s.store.evidence.length} evidence records. Fresh synthetic diagnostic state.`,
      style: "graph-card",
      action: () => systemGo("store"),
    });
    systemCard(world, {
      title: "WorldSession",
      eyebrow: "Update, retrieval and agent interface",
      text: "Inspect observed calls between storage, retrieval, binding, state update, context encoding and the actual BeliefAgent.",
      action: () => systemGo("world"),
    });
    systemCard(world, {
      title: "World State neural modules",
      eyebrow: "Separate foundation configuration",
      text: "Candidate encoder, association scorer, recurrent updater, context projections, predictor, readout and agent. Every neural layer and weight is inspectable.",
      action: () => go({ context: "world", scope: "" }),
    });
    const interfaces = document.createElement("button");
    interfaces.textContent = "Explore record schemas and optional interfaces";
    interfaces.onclick = () => systemGo("interfaces");
    page.append(interfaces);
  } else if (systemPage === "store") renderStore(page);
  else if (systemPage === "record") renderRecord(page);
  else if (systemPage === "component") {
    const c = s.components[systemSelection];
    page.innerHTML = `<h2>${esc(c.name)}</h2><p>${esc(c.type)} · ${fmt(c.calls)} recorded calls</p><pre>${esc(JSON.stringify(c.methods, null, 2))}</pre><p>This is a runtime interface. Its data records and any registered neural modules have separate inspection views.</p>`;
    sourceButton(page, c.source, c.type);
  } else if (systemPage === "schema") {
    const name = systemSelection,
      schema = s.schemas[name];
    page.innerHTML = `<h2>${esc(name)}</h2><p>Fields read from the current Python dataclass. This is a record schema, not an invented graph instance.</p><table class="system-table"><tr><th>Field</th><th>Type</th></tr>${schema.fields.map((f) => `<tr><td>${esc(f.name)}</td><td>${esc(f.type)}</td></tr>`).join("")}</table>`;
    sourceButton(page, schema.source, name);
  } else if (systemPage === "interfaces") {
    page.innerHTML =
      "<h2>Current record schemas</h2><p>Classes are discovered from current source. Optional interfaces listed below need an explicit consumer; presence in this inventory does not mean they executed.</p>";
    const schemas = document.createElement("div");
    schemas.className = "system-tabs";
    page.append(schemas);
    for (const name of Object.keys(s.schemas)) {
      const b = document.createElement("button");
      b.textContent = name;
      b.onclick = () => systemGo("schema", name);
      schemas.append(b);
    }
    const heading = document.createElement("h2");
    heading.textContent = "Neural, binding and extension interfaces";
    heading.style.marginTop = "24px";
    page.append(heading);
    for (const [name, item] of Object.entries(s.interfaces)) {
      const section = document.createElement("details");
      section.innerHTML = `<summary>${esc(name)}</summary><p>${esc(item.description)}</p>`;
      sourceButton(section, item.source, name);
      page.append(section);
    }
  }
}
