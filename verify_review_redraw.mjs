// Local Chrome CDP acceptance: node verify_review_redraw.mjs offline|live <150 port> <592 port> <output>
import assert from "node:assert/strict";
import {mkdir, readFile, readdir, writeFile} from "node:fs/promises";
import {resolve, join} from "node:path";
const mode = process.argv[2] || "offline";
const ports = [Number(process.argv[3] || 8781), Number(process.argv[4] || 8782)];
const out = resolve(process.argv[5] || "output-videos/whatif-defense-step5-4/browser-" + mode);
await mkdir(out, {recursive: true});
const target = await (await fetch("http://127.0.0.1:9234/json/new?about:blank", {method: "PUT"})).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((yes, no) => { ws.onopen = yes; ws.onerror = no; });
let id = 0;
const pending = new Map();
ws.onmessage = ({data}) => {
  const m = JSON.parse(data);
  if (!m.id) return;
  const p = pending.get(m.id); pending.delete(m.id);
  m.error ? p.no(Error(JSON.stringify(m.error))) : p.yes(m.result);
};
const call = (method, params = {}) => new Promise((yes, no) => {pending.set(++id, {yes, no}); ws.send(JSON.stringify({id, method, params}));});
const pause = ms => new Promise(yes => setTimeout(yes, ms));
const js = async expression => {
  const r = await call("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
  if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
const wait = async (predicate, timeout = 12000) => {
  const until = Date.now() + timeout;
  while (Date.now() < until) { const r = await predicate(); if (r) return r; await pause(80); }
  throw Error("Timed out: " + await js("document.querySelector('#redraw-status')?.textContent"));
};
const click = selector => js(`document.querySelector(${JSON.stringify(selector)}).click()`);
const ready = selector => js(`document.querySelector(${JSON.stringify(selector)}).getAttribute('aria-pressed') === 'true'`);
const waitPose = () => wait(async () => {
  if (await ready("#pose-result")) return true;
  const message = await js("document.querySelector('#redraw-status').textContent");
  if (message.includes("重繪失敗")) throw Error(message);
  return false;
}, 1200000);
async function drag([x, y]) {
  const [sx, sy, dx, dy] = await js(`(()=>{const p=document.querySelector('#defender').getBoundingClientRect();const b=document.querySelector('#board').getBoundingClientRect();return[p.x+p.width/2,p.y+p.height/2,b.x+b.width*${x}/60.96,b.y+b.height*${y}/25.908]})()`);
  await call("Input.dispatchMouseEvent", {type:"mousePressed", x:sx,y:sy,button:"left",clickCount:1});
  await call("Input.dispatchMouseEvent", {type:"mouseMoved", x:dx,y:dy,button:"left",buttons:1});
  await call("Input.dispatchMouseEvent", {type:"mouseReleased", x:dx,y:dy,button:"left",clickCount:1});
  await wait(() => ready("#result"));
}
async function exports(directory, animation = false) {
  await mkdir(directory, {recursive: true});
  await call("Page.setDownloadBehavior", {behavior: "allow", downloadPath: directory});
  for (const name of ["clean", "annotated", "report", ...(animation ? ["animation"] : [])]) await click("#save-" + name);
  const files = ["what-if.png", "what-if-annotated.png", "report.json", ...(animation ? ["tactical-path.mp4"] : [])];
  await wait(async () => {const names = await readdir(directory); return files.every(name => names.includes(name));}, 20000);
  return JSON.parse(await readFile(join(directory, "report.json"), "utf8"));
}
const records = [];
try {
  await call("Emulation.setDeviceMetricsOverride", {width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  for (let i = 0; i < ports.length; i++) {
    const frame = i ? 592 : 150;
    await call("Page.navigate", {url:"http://127.0.0.1:" + ports[i]});
    await call("Page.bringToFront");
    await wait(() => js("!!document.querySelector('#defender') && !document.querySelector('#defender').disabled"));
    await click("#reviewed-target");
    await wait(() => ready("#result"));
    await click("#redraw");
    await waitPose();
    const directory = join(out, "frame" + frame);
    const cached = await exports(join(directory, "cached-pose"), true);
    assert.equal(cached.result_kind, "pose_redraw");
    assert.equal(cached.source_frame, frame);
    assert.equal(cached.seed, i ? 104 : 105);
    assert.equal(cached.provenance, "reviewed_5.3_cache");
    const expected = resolve("output-videos/whatif-defense-step5-3", i ? "frame592-goal-clear" : "frame150-open-ice", "seed-" + cached.seed, "redrawn.png");
    assert.deepEqual(await readFile(join(directory,"cached-pose/what-if.png")), await readFile(expected));
    await click("#original");
    const original = await exports(join(directory,"original"));
    assert.equal(original.result_kind,"original");
    assert.deepEqual(original.target_m, original.source_m);
    await click("#result");
    assert.equal((await exports(join(directory,"preview"))).result_kind, "opencv_preview");
    if (mode === "offline") {
      // Deliver a cached response late, after switching away from this editable frame.
      await js(`window.baseFetch=fetch; window.fetch=async(url,opts)=>{const r=await baseFetch(url,opts);if(String(url).startsWith('/redraw/status') || String(url).startsWith('/redraw/image')) await new Promise(y=>setTimeout(y,800));return r;}; true`);
      await click("#redraw"); await pause(100);
      await drag(i ? [53.4,12.5] : [45.2,15.7]); await pause(1000);
      assert.equal(await js("document.querySelector('#pose-result').disabled"),true);
      const latest = await exports(join(directory,"late-target"));
      assert.equal(latest.result_kind,"opencv_preview");
      assert.ok(Math.abs(latest.target_m[0] - (i ? 53.4 : 45.2)) < .01);
      await click("#reviewed-target"); await wait(() => ready("#result"));
      await click("#redraw"); await pause(150); await click("#reference"); await pause(1300);
      assert.equal(await js("document.querySelector('#pose-result').disabled"),true);
      assert.equal(await js("document.querySelector('#save-report').disabled"),true);
      await click("#keyframe"); await click("#reviewed-target"); await wait(() => ready("#result"));
      await click("#redraw"); await pause(100); await click("#reset"); await pause(1000);
      assert.equal(await js("document.querySelector('#pose-result').disabled"),true);
      assert.equal(await js("document.querySelector('#save-report').disabled"),true);
      await click("#reviewed-target"); await wait(() => ready("#result"));
      await drag(i ? [52.9,12.5] : [45.9,15.7]);
      await click("#redraw");
      await wait(() => js("document.querySelector('#redraw-status').textContent.includes('重繪失敗')"));
      assert.equal(await js("document.querySelector('#redraw').disabled"),false);
      assert.equal(await ready("#result"),true);
      const message = await js("document.querySelector('#redraw-status').textContent");
      await click("#redraw");
      await wait(() => js("!document.querySelector('#redraw').disabled"));
      records.push({frame,cache_display_seconds:cached.ui_display_seconds,offline_cache:true,stale_frame_blocked:true,
        stale_target_blocked:true,reset_blocked:true,retry:true,message});
    } else {
      const targets = i ? [[53.4,12.5],[54.0,13.1]] : [[45.2,15.7],[44.1,16.3]];
      for (let j = 0; j < targets.length; j++) {
        await drag(targets[j]);
        await click("#redraw");
        if (j === 0) {
          await wait(() => js("/重繪中|快取已載入|重繪完成/.test(document.querySelector('#redraw-status').textContent)"));
          // Abandon a running request; its image may be cached, but cannot appear on the reference frame.
          await click("#reference"); await pause(250);
          assert.equal(await js("document.querySelector('#pose-result').disabled"),true);
          await click("#keyframe"); await click("#reviewed-target"); await wait(() => ready("#result"));
          await drag(targets[j]); await click("#redraw");
        }
        await waitPose();
        await wait(() => js("!document.querySelector('#redraw').disabled && document.querySelector('#redraw-status').textContent.includes('顯圖')"));
        assert.match(await js("document.querySelector('#contact-status').textContent"),/位置待確認/);
        const location = join(directory,"new-"+j);
        const report = await exports(location, true);
        assert.equal(report.result_kind,"pose_redraw");
        assert.equal(report.position_review.status,"needs_review");
        for (const k of ["outside_target_changed_pixels","outside_all_edits_changed_pixels","protected_changed_pixels"]) assert.equal(report.verification[k],0);
        report.target_m.forEach((v,k)=>assert.ok(Math.abs(v-targets[j][k])<.01));
        // UI mechanics only: synthetic click is not human or ground-truth validation.
        await click("#contact-mode");
        await js(`(()=>{const c=document.querySelector('#contact-zoom');const r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('click',{clientX:r.x+r.width/2,clientY:r.y+r.height/2,bubbles:true}));return true})()`);
        await wait(() => js("document.querySelector('#contact-status').textContent.includes('±2 px')"));
        assert.match(await js("document.querySelector('#contact-status').textContent"),/位置待確認/);
        await click("#contact-clear");
        await wait(() => js("!document.querySelector('#contact-status').textContent.includes('±2 px')"));
        const capture = await call("Page.captureScreenshot", {format:"png",captureBeyondViewport:true});
        await writeFile(join(location,"interface.png"),Buffer.from(capture.data,"base64"));
        records.push({frame,target:report.target_m,job:report.job_id,provenance:report.provenance,delivery:report.delivery,ui_seconds:report.ui_display_seconds,
          queue_seconds:report.queue_seconds,generation:report.generation,verification:report.verification,
          manual_review_ui:true,synthetic_click_not_quality_evidence:true});
      }
    }
  }
  await writeFile(join(out,"acceptance.json"),JSON.stringify({mode,records},null,2)+"\n");
  console.log(JSON.stringify({mode,records}));
} finally {ws.close();}
