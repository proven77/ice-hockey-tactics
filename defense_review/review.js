const byId = id => document.getElementById(id);
const board = byId("board");
const defender = byId("defender");
const shot = byId("shot");
const markers = byId("markers");

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error((await response.json()).error);
  return response;
}

async function loadImage(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("無法讀取原始影像");
  return createImageBitmap(await response.blob());
}

function status(message, error = false) {
  byId("status").textContent = message;
  byId("status").dataset.error = error;
}

function drawLane(canvas, defense, project, coverage, scale = 1) {
  if (defense?.status !== "ready") return;
  const ctx = canvas.getContext("2d");
  const puck = project(defense.puck_m);
  const posts = defense.goal_posts_m.map(project);
  if (!puck || posts.some(point => !point)) return;

  function wedge([a, b], fill, outline) {
    ctx.beginPath();
    ctx.moveTo(...puck);
    ctx.lineTo(...a);
    ctx.lineTo(...b);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = outline;
    ctx.lineWidth = 2 * scale;
    ctx.stroke();
  }

  wedge(posts, "rgba(67, 184, 202, .16)", "#3daaba");
  for (const endpoints of coverage?.blocked_goal_spans_m ?? []) {
    const span = endpoints.map(project);
    if (span.every(Boolean)) wedge(span, "rgba(240, 161, 47, .42)", "#ce8624");
  }
  for (const [i, point] of [puck, ...posts].entries()) {
    ctx.beginPath();
    ctx.arc(...point, 4 * scale, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#172025" : "#5de0ea";
    ctx.fill();
    ctx.lineWidth = scale;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
}

function draw(canvas, bitmap, foot, defense, project, coverage) {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  drawLane(canvas, defense, project, coverage, 2);
  if (!foot) return;

  const [x, y] = foot;
  ctx.beginPath();
  ctx.ellipse(x, y + 5, 30, 13, 0, 0, Math.PI * 2);
  ctx.lineWidth = 9;
  ctx.strokeStyle = "#162328";
  ctx.stroke();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#ffcf75";
  ctx.stroke();

  const tagX = x + 115 < canvas.width ? x + 39 : x - 114;
  const tagY = Math.max(0, Math.min(canvas.height - 48, y - 20));
  ctx.fillStyle = "#162328";
  ctx.fillRect(tagX, tagY, 75, 48);
  ctx.fillStyle = "#ffcf75";
  ctx.fillRect(tagX, tagY, 5, 48);
  ctx.font = "bold 36px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("D1", tagX + 40, tagY + 25);
}

function download(name, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

async function run() {
  const [data, original, reference] = await Promise.all([
    fetch("/case").then(response => response.json()),
    loadImage("/original.png"),
    loadImage("/reference.png"),
  ]);
  [shot.width, shot.height] = data.image_size;
  byId("reference").textContent = `第 ${data.reference_frame} 幀`;
  byId("keyframe").textContent = `第 ${data.frame} 幀`;

  const state = {
    reference: false, selected: false, ready: false, preparing: null,
    revision: 0, target: data.source_m, drag: null, result: null, showResult: false,
    measurement: null, measurementError: "",
  };
  const lane = byId("shot-lane");
  const suggest = byId("suggest");
  const { defense } = data;
  let measuring = false;
  let queuedMeasurement = null;

  function locate(node, target) {
    node.style.left = `${100 * target[0] / data.rink_size[0]}%`;
    node.style.top = `${100 * target[1] / data.rink_size[1]}%`;
  }

  for (const track of data.tracks) {
    const dot = document.createElement("span");
    dot.className = `track team-${track.team}`;
    locate(dot, track.position);
    byId("tracks").append(dot);
  }

  function iceToPixel([x, y]) {
    const [u, v, w] = data.ice_to_image.map(row => row[0] * x + row[1] * y + row[2]);
    const point = [u / w, v / w];
    return point.every((coordinate, index) => coordinate >= 0 && coordinate < data.image_size[index])
      ? point : null;
  }

  function updateDefense(measurement) {
    lane.getContext("2d").clearRect(0, 0, lane.width, lane.height);
    const summary = byId("coverage");
    const detail = byId("coverage-detail");
    suggest.hidden = true;
    if (state.reference || defense.status !== "ready") {
      summary.textContent = state.reference ? "前一幀僅供畫面對照" : defense.reason;
      detail.textContent = "";
      return;
    }
    drawLane(lane, defense,
      ([x, y]) => [x * lane.width / data.rink_size[0], y * lane.height / data.rink_size[1]],
      measurement?.after);
    const before = defense.before.blocked_pct.toFixed(1);
    const after = measurement?.after;
    if (after) {
      const change = measurement.delta_percentage_points;
      const signedChange = (change > 0 ? "+" : "") + change.toFixed(1);
      const central = after.central_blocked ? "已封住" : "仍開放";
      summary.textContent = `原位 ${before}% → 落點 ${after.blocked_pct.toFixed(1)}%（${signedChange} 個百分點）`;
      detail.textContent = `遮擋 ${after.blocked_angle_deg.toFixed(1)}° / ${defense.opening_angle_deg.toFixed(1)}° · 中央射門線${central} · D1 半徑 ${defense.blocking_radius_m.toFixed(2)} m`;
    } else {
      summary.textContent = state.measurementError || `原位 ${before}% · 量測落點中`;
      detail.textContent = "";
    }
    const suggestion = defense.suggestion;
    if (suggestion.status === "available") {
      suggest.hidden = false;
      suggest.textContent = `套用規則站位（${suggestion.distance_m.toFixed(2)} m）`;
    } else {
      const note = suggestion.status === "keep" ? "原位已封住中央方向，建議保留。" : suggestion.reason;
      detail.textContent = [detail.textContent, note].filter(Boolean).join(" · ");
    }
  }

  async function measurePreview() {
    if (measuring || !queuedMeasurement) return;
    measuring = true;
    const { target, revision } = queuedMeasurement;
    queuedMeasurement = null;
    try {
      const response = await post("/measure", { target });
      const measurement = await response.json();
      if (revision === state.revision && target === state.target && !state.result) {
        state.measurement = measurement;
        update();
      }
    } catch (error) {
      if (revision === state.revision && target === state.target && !state.result) {
        state.measurementError = error.message;
        update();
      }
    } finally {
      measuring = false;
      measurePreview();
    }
  }

  function previewTarget() {
    state.measurement = null;
    state.measurementError = "";
    if (defense.status === "ready" && state.target !== data.source_m) {
      queuedMeasurement = { target: state.target, revision: state.revision };
      measurePreview();
    }
    update();
  }

  function update() {
    const { result, target, selected, reference: preview } = state;
    const show = !preview && state.showResult && !!result;
    let foot = null;
    if (markers.checked && !preview && selected) {
      foot = show ? result.report.applied_foot_px : result ? data.source_px : iceToPixel(target);
    }
    const measurement = result?.report.defense ?? state.measurement ?? (target === data.source_m ? defense : null);
    const overlay = markers.checked && !preview ? defense : null;
    draw(shot, preview ? reference : show ? result.bitmap : original, foot, overlay, iceToPixel,
      show || !result ? measurement?.after : defense.before);
    updateDefense(measurement);
    byId("shot-title").textContent = show ? "What-If 影像" : "原鏡頭";
    byId("original").setAttribute("aria-pressed", !show);
    byId("result").setAttribute("aria-pressed", show);
    byId("result").disabled = preview || !result;
    for (const id of ["save-clean", "save-annotated", "save-report"]) byId(id).disabled = preview || !result;
    byId("reference").setAttribute("aria-pressed", preview);
    byId("keyframe").setAttribute("aria-pressed", !preview);
    defender.hidden = preview;
    byId("tracks").hidden = preview;
    defender.disabled = preview;
    defender.setAttribute("aria-pressed", selected);
    byId("reset").disabled = preview || !selected;
    locate(defender, target);
    const distance = Math.hypot(target[0] - data.source_m[0], target[1] - data.source_m[1]);
    byId("position").textContent = preview ? "前一幀 · 影片參考"
      : `D1　(${target[0].toFixed(2)}, ${target[1].toFixed(2)}) m　·　位移 ${distance.toFixed(2)} m`;
  }

  function invalidate() {
    state.revision += 1;
    state.result?.bitmap.close();
    state.result = null;
    state.showResult = false;
    state.measurement = null;
    state.measurementError = "";
    queuedMeasurement = null;
    return state.revision;
  }

  function prepare(retry = false) {
    if (state.ready) return Promise.resolve();
    if (state.preparing) return state.preparing;
    state.preparing = (async () => {
      let response = await post("/prepare", { retry });
      while (response.status === 202) {
        await new Promise(resolve => setTimeout(resolve, 500));
        response = await post("/prepare", {});
      }
      state.ready = true;
    })().finally(() => { state.preparing = null; });
    return state.preparing;
  }

  function showError(error) {
    status(error.message, true);
    byId("retry").hidden = state.ready;
  }

  async function select(retry = false) {
    const wasSelected = state.selected;
    state.selected = true;
    byId("retry").hidden = true;
    update();
    if (state.ready) {
      if (!wasSelected) status("已選取 D1，可以拖曳球員");
      return;
    }
    const revision = state.revision;
    status("修補中，可以先拖曳 D1");
    try {
      await prepare(retry);
      if (revision === state.revision) status("修補背景已就緒");
    } catch (error) {
      if (revision === state.revision) showError(error);
    }
  }

  async function apply(endedAt, retry = false) {
    const measurement = state.measurement;
    const revision = invalidate();
    const target = state.target;
    const warm = state.ready;
    if (measurement?.target_m?.every((value, i) => value === target[i])) {
      state.measurement = measurement;
      update();
    } else {
      previewTarget();
    }
    byId("retry").hidden = true;
    status(warm ? "合成中" : "修補中，已保留目前落點");
    try {
      await prepare(retry);
      if (revision !== state.revision) return;
      const response = await post("/render", { target });
      const report = JSON.parse(response.headers.get("X-Placement"));
      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob);
      if (revision !== state.revision) {
        bitmap.close();
        return;
      }
      state.result = { report, blob, bitmap };
      state.showResult = true;
      update();
      status("What-If 已就緒");
      if (warm) {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (revision === state.revision) performance.measure("what-if-preview", { start: endedAt, end: performance.now() });
        }));
      }
    } catch (error) {
      if (revision === state.revision) showError(error);
    }
  }

  function switchFrame(preview) {
    invalidate();
    state.reference = preview;
    state.selected = false;
    state.target = data.source_m;
    state.drag = null;
    byId("retry").hidden = true;
    status(preview ? "前一幀供對照。回到關鍵幀可編輯。" : "選取 D1 開始復盤");
    update();
  }

  function pointerPosition(event, rect, offset) {
    const position = [event.clientX - rect.left, event.clientY - rect.top];
    return position.map((value, i) => (value - offset[i]) * data.rink_size[i] / [rect.width, rect.height][i]);
  }

  defender.addEventListener("pointerdown", event => {
    if (event.button !== 0 || state.drag) return;
    event.preventDefault();
    defender.focus();
    const rect = board.getBoundingClientRect();
    const offset = [event.clientX - rect.left - state.target[0] / data.rink_size[0] * rect.width,
                    event.clientY - rect.top - state.target[1] / data.rink_size[1] * rect.height];
    state.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, rect, offset, moved: false, original: state.target };
    defender.setPointerCapture(event.pointerId);
    select();
  });

  defender.addEventListener("pointermove", event => {
    const drag = state.drag;
    if (!drag || event.pointerId !== drag.id) return;
    if (!drag.moved) {
      if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 3) return;
      drag.moved = true;
      invalidate();
    }
    state.target = pointerPosition(event, drag.rect, drag.offset);
    previewTarget();
  });

  defender.addEventListener("pointerup", event => {
    const drag = state.drag;
    if (!drag || event.pointerId !== drag.id) return;
    state.drag = null;
    if (drag.moved) {
      state.target = pointerPosition(event, drag.rect, drag.offset);
      apply(event.timeStamp);
    }
  });

  function cancelDrag() {
    if (!state.drag) return;
    state.target = state.drag.original;
    state.drag = null;
    status("已取消拖曳");
    previewTarget();
  }

  defender.addEventListener("pointercancel", cancelDrag);
  defender.addEventListener("lostpointercapture", cancelDrag);
  defender.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (state.target === data.source_m) select();
      else apply(event.timeStamp);
      return;
    }
    const delta = { ArrowLeft: [-.2, 0], ArrowRight: [.2, 0], ArrowUp: [0, -.2], ArrowDown: [0, .2] }[event.key];
    if (!delta) return;
    event.preventDefault();
    invalidate();
    state.target = state.target.map((value, index) => value + delta[index]);
    select();
    previewTarget();
  });

  byId("reset").onclick = () => switchFrame(false);
  byId("reference").onclick = () => switchFrame(true);
  byId("keyframe").onclick = () => switchFrame(false);
  suggest.onclick = event => {
    state.selected = true;
    state.target = defense.suggestion.target_m;
    apply(event.timeStamp);
  };
  byId("original").onclick = () => { state.showResult = false; update(); };
  byId("result").onclick = () => { state.showResult = true; update(); };
  byId("retry").onclick = () => {
    if (state.target === data.source_m) select(true);
    else apply(performance.now(), true);
  };
  markers.onchange = update;
  byId("save-clean").onclick = () => download("what-if.png", state.result.blob);
  byId("save-report").onclick = () => download("report.json",
    new Blob([JSON.stringify(state.result.report, null, 2) + "\n"], { type: "application/json" }));
  byId("save-annotated").onclick = () => {
    const canvas = document.createElement("canvas");
    [canvas.width, canvas.height] = data.image_size;
    const result = state.result;
    const measurement = result.report.defense;
    draw(canvas, result.bitmap, result.report.applied_foot_px, measurement, iceToPixel, measurement.after);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#162328";
    const caption = measurement.status === "ready"
      ? `What-If   ·   ${measurement.before.blocked_pct.toFixed(1)}% → ${measurement.after.blocked_pct.toFixed(1)}%`
      : "What-If";
    ctx.font = "bold 28px system-ui";
    ctx.fillRect(20, 20, ctx.measureText(caption).width + 30, 48);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#ffcf75";
    ctx.fillText(caption, 34, 53);
    canvas.toBlob(blob => {
      if (blob) download("what-if-annotated.png", blob);
      else status("無法匯出分析圖", true);
    }, "image/png");
  };
  switchFrame(false);
}

run().catch(error => status(error.message, true));
