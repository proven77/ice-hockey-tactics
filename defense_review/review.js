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

function draw(canvas, bitmap, foot) {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
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
  };

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

  function update() {
    const { result, target, selected, reference: preview } = state;
    const show = !preview && state.showResult && !!result;
    let foot = null;
    if (markers.checked && !preview && selected) {
      foot = show ? result.report.applied_foot_px : result ? data.source_px : iceToPixel(target);
    }
    draw(shot, preview ? reference : show ? result.bitmap : original, foot);
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
    const revision = invalidate();
    const target = state.target;
    const warm = state.ready;
    update();
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
    update();
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
    update();
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
  });

  byId("reset").onclick = () => switchFrame(false);
  byId("reference").onclick = () => switchFrame(true);
  byId("keyframe").onclick = () => switchFrame(false);
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
    draw(canvas, state.result.bitmap, state.result.report.applied_foot_px);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#162328";
    ctx.fillRect(20, 20, 153, 48);
    ctx.font = "bold 28px system-ui";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#ffcf75";
    ctx.fillText("What-If", 34, 53);
    canvas.toBlob(blob => {
      if (blob) download("what-if-annotated.png", blob);
      else status("無法匯出分析圖", true);
    }, "image/png");
  };
  switchFrame(false);
}

run().catch(error => status(error.message, true));
