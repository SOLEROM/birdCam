/* BEV Web Sim dashboard: parameter panel + live readouts. */
"use strict";

const $ = (sel) => document.querySelector(sel);

let camerasCfg = null;
let bevCfg = null;
let sceneCfg = null;

const debounce = (fn, ms) => {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
};

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.text();
    console.error(`POST ${url} failed`, detail);
    throw new Error(detail);
  }
  return r.json();
}

/* ---------------- camera sliders ---------------- */

const CAM_SLIDERS = [
  { key: "z", label: "height m", min: 0.1, max: 3.0, step: 0.05 },
  { key: "yaw_deg", label: "yaw °", min: -180, max: 180, step: 1 },
  { key: "pitch_deg", label: "pitch °", min: -10, max: 89, step: 1 },
  { key: "roll_deg", label: "roll °", min: -45, max: 45, step: 1 },
  { key: "fov_deg", label: "fov °", min: 40, max: 160, step: 1 },
  { key: "x", label: "x m", min: -2, max: 2, step: 0.05 },
  { key: "y", label: "y m", min: -2, max: 2, step: 0.05 },
];

const pushCameras = debounce(async () => {
  camerasCfg = await postJSON("/config/cameras", camerasCfg);
}, 250);

function renderCamSliders() {
  const name = $("#cam-select").value;
  const cam = camerasCfg.cameras[name];
  const host = $("#cam-sliders");
  host.innerHTML = "";
  for (const s of CAM_SLIDERS) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="name">${s.label}</span>
      <input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${cam[s.key]}">
      <span class="val">${cam[s.key]}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      row.querySelector(".val").textContent = input.value;
      camerasCfg.cameras[name][s.key] = parseFloat(input.value);
      pushCameras();
    });
    host.appendChild(row);
  }
}

/* ---------------- BEV controls ---------------- */

const BEV_FIELDS = [
  { key: "x_min", min: -30, max: 0, step: 0.5 },
  { key: "x_max", min: 0.5, max: 30, step: 0.5 },
  { key: "y_min", min: -30, max: -0.5, step: 0.5 },
  { key: "y_max", min: 0.5, max: 30, step: 0.5 },
  { key: "resolution", min: 0.01, max: 0.1, step: 0.005 },
  { key: "max_range", min: 2, max: 50, step: 1 },
];

const pushBev = debounce(async () => {
  bevCfg = await postJSON("/config/bev", bevCfg);
}, 350);

function renderBevControls() {
  const host = $("#bev-controls");
  host.innerHTML = "";
  for (const f of BEV_FIELDS) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="name">${f.key}</span>
      <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${bevCfg[f.key]}">
      <span class="val">${bevCfg[f.key]}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      row.querySelector(".val").textContent = input.value;
      bevCfg[f.key] = parseFloat(input.value);
      pushBev();
    });
    host.appendChild(row);
  }
  $("#blend-select").value = bevCfg.blend;
}

/* ---------------- overlays ---------------- */

async function renderOverlays() {
  const flags = await getJSON("/debug/overlays");
  const host = $("#overlay-checks");
  host.innerHTML = "";
  for (const [name, on] of Object.entries(flags)) {
    const row = document.createElement("label");
    row.className = "row";
    row.innerHTML = `<input type="checkbox" ${on ? "checked" : ""}> ${name}`;
    row.querySelector("input").addEventListener("change", async () => {
      const current = {};
      host.querySelectorAll("label").forEach((l) => {
        current[l.textContent.trim()] = l.querySelector("input").checked;
      });
      await postJSON("/debug/overlays", current);
    });
    host.appendChild(row);
  }
}

/* ---------------- obstacles ---------------- */

const pushScene = debounce(async () => {
  sceneCfg = await postJSON("/config/scene", sceneCfg);
}, 350);

function renderObstacles() {
  const host = $("#obstacle-list");
  host.innerHTML = "";
  sceneCfg.obstacles.forEach((obs, i) => {
    const div = document.createElement("div");
    div.className = "obstacle";
    div.innerHTML = `<a class="remove" href="#">remove</a><b>${obs.type}</b>`;
    for (const key of ["x", "y", "size_x", "size_y", "size_z"]) {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span class="name">${key}</span>
        <input type="number" step="0.1" value="${obs[key]}">`;
      row.querySelector("input").addEventListener("change", (e) => {
        sceneCfg.obstacles[i][key] = parseFloat(e.target.value);
        pushScene();
      });
      div.appendChild(row);
    }
    div.querySelector(".remove").addEventListener("click", (e) => {
      e.preventDefault();
      sceneCfg.obstacles.splice(i, 1);
      renderObstacles();
      pushScene();
    });
    host.appendChild(div);
  });
}

/* ---------------- BEV hover readout ---------------- */

function setupBevReadout() {
  const img = $("#img-bev");
  img.addEventListener("mousemove", (e) => {
    if (!bevCfg) return;
    const rect = img.getBoundingClientRect();
    const scaleX = bevCfg.width_px / rect.width;
    const scaleY = bevCfg.height_px / rect.height;
    const col = (e.clientX - rect.left) * scaleX;
    const row = (e.clientY - rect.top) * scaleY;
    const x = bevCfg.x_max - (row + 0.5) * bevCfg.resolution;
    const y = bevCfg.y_max - (col + 0.5) * bevCfg.resolution;
    $("#bev-readout").textContent = `X=${x.toFixed(2)} m  Y=${y.toFixed(2)} m`;
  });
  img.addEventListener("mouseleave", () => { $("#bev-readout").textContent = ""; });
}

/* ---------------- status websocket ---------------- */

function setupStatus() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
  ws.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    $("#status").textContent =
      `${s.fps} fps · render ${s.render_ms} ms · frame #${s.seq}`;
  };
  ws.onclose = () => setTimeout(setupStatus, 2000);
}

/* ---------------- init ---------------- */

async function init() {
  [camerasCfg, bevCfg, sceneCfg] = await Promise.all([
    getJSON("/config/cameras"), getJSON("/config/bev"), getJSON("/config/scene"),
  ]);

  const sel = $("#cam-select");
  for (const name of Object.keys(camerasCfg.cameras)) {
    const opt = document.createElement("option");
    opt.textContent = name;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", renderCamSliders);
  renderCamSliders();
  renderBevControls();
  renderOverlays();
  renderObstacles();
  setupBevReadout();
  setupStatus();

  $("#blend-select").addEventListener("change", (e) => {
    bevCfg.blend = e.target.value;
    pushBev();
  });
  $("#add-obstacle").addEventListener("click", () => {
    sceneCfg.obstacles.push(
      { type: "box", x: 3.0, y: 0.0, size_x: 0.5, size_y: 0.5, size_z: 0.5,
        color: [40, 90, 200] });
    renderObstacles();
    pushScene();
  });
  $("#save-btn").addEventListener("click", async () => {
    await postJSON("/config/save", {});
    $("#save-msg").textContent = "saved ✓";
    setTimeout(() => { $("#save-msg").textContent = ""; }, 2000);
  });
}

init().catch((err) => { $("#status").textContent = `init failed: ${err}`; });
