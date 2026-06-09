"""LiveSD - real-time webcam transformation.

Two front-ends, same engines:
  * Gradio UI at  /        — full controls, side-by-side feeds (great on desktop)
  * Custom page  /live     — FaceTime-style immersive view (full-screen result +
                             picture-in-picture source), ideal on a phone

Modes: Lite (SD-Turbo img2img), Wardrobe (SD1.5 LCM-inpaint of clothing from a
prompt), Try-On (garment image -> IP-Adapter live + CatVTON HQ capture).

Run:  .venv\\Scripts\\python.exe app.py
"""
import base64
import io
import json
import threading

import numpy as np
import gradio as gr
from PIL import Image

from engine import Engine
from engine_wardrobe import WardrobeEngine
from engine_tryon import TryOnEngine
from engine_catvton import CatVTONEngine
from captioner import captioner

engine = Engine()
engine.load()

# Heavy engines (inpaint / VTON) -> lazy load on first use.
wardrobe_engine = WardrobeEngine()
tryon_engine = TryOnEngine()
catvton_engine = CatVTONEngine()
_wardrobe_lock = threading.Lock()
_tryon_lock = threading.Lock()
_catvton_lock = threading.Lock()


def _ensure(eng, lock):
    if not eng.loaded:
        with lock:
            if not eng.loaded:
                eng.load()


def ensure_wardrobe():
    _ensure(wardrobe_engine, _wardrobe_lock)


def ensure_tryon():
    _ensure(tryon_engine, _tryon_lock)


def ensure_catvton():
    _ensure(catvton_engine, _catvton_lock)


DEFAULT_NEG = "blurry, distorted, deformed, ugly, low quality, extra limbs"

# Each style preset: (prompt, negative, recommended_strength).
PRESETS = {
    "Girl": ("a beautiful young woman, feminine face, long flowing hair, soft delicate features, natural makeup, portrait, detailed, photorealistic",
             "man, male, beard, mustache, stubble, masculine, " + DEFAULT_NEG, 0.7),
    "Disney / Pixar": ("a disney pixar 3d animated character, big expressive eyes, vibrant colors, cinematic lighting",
                       DEFAULT_NEG, 0.6),
    "Anime": ("anime style portrait, cel shaded, studio ghibli, clean lineart, vibrant",
              DEFAULT_NEG, 0.6),
    "Wearing a suit": ("a person wearing an elegant business suit, professional portrait, sharp, detailed",
                       DEFAULT_NEG, 0.55),
    "Oil painting": ("a classical oil painting portrait, baroque, rich brush strokes, dramatic lighting",
                     DEFAULT_NEG, 0.6),
    "Cyberpunk": ("cyberpunk character, neon lights, futuristic, blade runner aesthetic, detailed",
                  DEFAULT_NEG, 0.6),
    "Superhero": ("a marvel superhero in costume, comic book style, dynamic, heroic, detailed",
                  DEFAULT_NEG, 0.6),
}

DEFAULT_PROMPT = PRESETS["Disney / Pixar"][0]

# Wardrobe mode: garment-only prompts (the clothing region is repainted, you stay you).
WARDROBE_NEG = "blurry, distorted, deformed, extra limbs, low quality, naked, nsfw"
# Specific, directive prompts -> consistent garment identity (color/fabric/cut/lighting)
WARDROBE_PRESETS = {
    "Suit": "wearing a charcoal-grey single-breasted wool business suit, crisp white dress shirt, navy silk tie, slim fit, sharp notch lapels, studio lighting, detailed fabric",
    "Tuxedo": "wearing a classic black tuxedo with satin peak lapels, white pleated dress shirt, black bow tie, formal eveningwear, slim fit, studio lighting, detailed",
    "Kurta": "wearing a white cotton kurta with subtle gold thread embroidery on the collar, traditional indian menswear, straight cut, soft natural lighting, detailed fabric",
    "Hoodie": "wearing a heather-grey pullover hoodie, soft cotton fleece, relaxed fit, ribbed cuffs, drawstring hood, casual streetwear, detailed fabric",
    "Leather jacket": "wearing a black leather biker jacket, asymmetric silver zip, fitted, matte finish, quilted shoulders, edgy streetwear, detailed",
    "T-shirt": "wearing a plain white crew-neck cotton t-shirt, regular fit, soft combed cotton, clean seams, detailed fabric",
}


# ---------------------------------------------------------------------------
# Shared inference helper (used by both the Gradio UI and the /live page)
# ---------------------------------------------------------------------------
LIVE_SHARPEN = 0.7  # unsharp amount to counter TAESD softness on the live feed


def run_transform(frame, mode, prompt, negative, strength, steps, stabilize,
                  smoothing, keep_bg, garment, ip_scale):
    if mode == "Wardrobe":
        ensure_wardrobe()
        out, fps = wardrobe_engine.transform(frame, prompt, steps, stabilize,
                                             smoothing, negative=negative)
    elif mode == "Try-On":
        ensure_tryon()
        out, fps = tryon_engine.transform(frame, garment, prompt, steps, stabilize,
                                          smoothing, ip_scale, negative=negative)
    else:
        out, fps = engine.transform(frame, prompt, strength, steps, stabilize,
                                    negative=negative, keep_bg=keep_bg)
    if out is not None:
        from perf import sharpen
        out = sharpen(out, LIVE_SHARPEN)
    return out, fps


def run_capture(frame, mode, prompt, negative, garment):
    if mode == "Try-On":
        ensure_catvton()
        return catvton_engine.capture(frame, garment)
    ensure_wardrobe()
    return wardrobe_engine.capture(frame, prompt, negative or WARDROBE_NEG)


# ===========================================================================
# Gradio UI (desktop-friendly full controls)
# ===========================================================================
def stream_fn(frame, mode, prompt, negative, strength, steps, stabilize,
              smoothing, keep_bg, garment, ip_scale):
    return run_transform(frame, mode, prompt, negative, strength, steps, stabilize,
                         smoothing, keep_bg, garment, ip_scale)


def capture_fn(frame, mode, prompt, negative, garment):
    return run_capture(frame, mode, prompt, negative, garment)


def caption_fn(garment):
    """Auto-describe an uploaded garment so the prompt agrees with the image."""
    if garment is None:
        return gr.update()
    cap = captioner.caption_garment(garment)
    return cap or gr.update()


NOTES = {
    "Lite": "**Lite**: fast plain img2img. Keeps your original (beard, features) — best for style filters.",
    "Wardrobe": "**Wardrobe**: changes ONLY your clothes (inpaint) — face/pose/background untouched. Pick a garment, then **📸 Capture (HQ)**. First switch loads inpaint models (~10s).",
    "Try-On": "**Try-On**: attach a **garment image** → live IP-Adapter preview, then **📸 Capture (HQ)** for a faithful CatVTON try-on. First use loads models (~30s).",
}


def on_mode_change(mode):
    if mode == "Wardrobe":
        wardrobe_engine.reset_temporal()
    elif mode == "Try-On":
        tryon_engine.reset_temporal()
    is_lite = mode == "Lite"
    is_wardrobe = mode == "Wardrobe"
    is_tryon = mode == "Try-On"
    return (
        gr.update(visible=is_lite),                  # strength
        gr.update(visible=is_wardrobe or is_tryon),  # smoothing
        gr.update(visible=is_lite),                  # keep_bg
        gr.update(visible=is_lite),                  # style presets
        gr.update(visible=is_wardrobe),              # garment presets
        gr.update(visible=is_wardrobe or is_tryon),  # capture row
        gr.update(visible=is_tryon),                 # garment upload
        gr.update(visible=is_tryon),                 # ip_scale
        NOTES[mode],
    )


with gr.Blocks(title="LiveSD", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 LiveSD — Real-time Webcam Transformation")
    gr.Markdown("Click **⏺ Record** to stream.")
    open_live = gr.Button("🖥️ Open full-screen immersive view", variant="primary")
    open_live.click(None, js="() => { window.location.href = '/live'; }")

    mode = gr.Radio(["Lite", "Wardrobe", "Try-On"], value="Lite", label="Mode")
    mode_note = gr.Markdown(NOTES["Lite"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Source (webcam)")
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                           label="", height=512, show_label=False)
        with gr.Column():
            gr.Markdown("### Transformed")
            out = gr.Image(type="numpy", label="", height=512, show_label=False, interactive=False)
            fps = gr.Textbox(label="Performance", interactive=False, value="warming up...")

    garment = gr.Image(label="👕 Garment image — click to upload, or drag & drop an outfit photo",
                       type="numpy", sources=["upload", "clipboard"], height=256, visible=False)

    with gr.Accordion("⚙️ Prompt & Settings", open=False):
        prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2,
                            placeholder="Describe what you want to become...")
        negative = gr.Textbox(label="Negative prompt (what to avoid)", value=DEFAULT_NEG, lines=1)
        with gr.Row() as style_row:
            preset_buttons = [(name, gr.Button(name, size="sm")) for name in PRESETS]
        with gr.Row(visible=False) as garment_row:
            garment_buttons = [(name, gr.Button(name, size="sm")) for name in WARDROBE_PRESETS]
        with gr.Row():
            strength = gr.Slider(0.2, 0.95, value=0.6, step=0.05, label="Strength (likeness ↔ transformation)")
            steps = gr.Slider(1, 4, value=2, step=1, label="Steps (speed ↔ quality)")
            stabilize = gr.Checkbox(value=True, label="Stabilize (fixed seed)")
            keep_bg = gr.Checkbox(value=False, label="Keep background (person only — Lite)")
        with gr.Row():
            smoothing = gr.Slider(0.0, 0.8, value=0.4, step=0.05, visible=False,
                                  label="Temporal smoothing (anti-flicker ↔ motion lag)")
            ip_scale = gr.Slider(0.3, 1.2, value=0.8, step=0.05, visible=False,
                                 label="Garment influence (IP-Adapter scale — Try-On live)")

    with gr.Accordion("📸 Capture (HQ)", open=False, visible=False) as capture_row:
        with gr.Row():
            capture_btn = gr.Button("Capture current frame", variant="primary", scale=1)
            capture_out = gr.Image(type="numpy", label="Capture result (HQ)", height=512,
                                   interactive=False, scale=2)

    for name, btn in preset_buttons:
        btn.click(lambda n=name: (PRESETS[n][0], PRESETS[n][1], PRESETS[n][2]),
                  outputs=[prompt, negative, strength])
    for name, btn in garment_buttons:
        btn.click(lambda n=name: (WARDROBE_PRESETS[n], WARDROBE_NEG), outputs=[prompt, negative])

    capture_btn.click(capture_fn, inputs=[cam, mode, prompt, negative, garment], outputs=[capture_out])
    # Auto-caption the uploaded garment -> fill the prompt (so text agrees with the image)
    garment.change(caption_fn, inputs=[garment], outputs=[prompt])
    mode.change(on_mode_change, inputs=[mode],
                outputs=[strength, smoothing, keep_bg,
                         style_row, garment_row, capture_row, garment, ip_scale, mode_note])
    cam.stream(
        stream_fn,
        inputs=[cam, mode, prompt, negative, strength, steps, stabilize,
                smoothing, keep_bg, garment, ip_scale],
        outputs=[out, fps],
        stream_every=0.1, concurrency_limit=1, show_progress="hidden",
    )


# ===========================================================================
# /live  — custom FaceTime-style immersive page (full DOM control)
# ===========================================================================
def _dataurl_to_np(durl):
    if not durl:
        return None
    b64 = durl.split(",", 1)[1] if "," in durl else durl
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)


def _np_to_jpeg(arr, quality=85):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_STYLE_PRESETS_JS = json.dumps({n: {"prompt": p, "negative": neg, "strength": s}
                                for n, (p, neg, s) in PRESETS.items()})
_GARMENT_PRESETS_JS = json.dumps(WARDROBE_PRESETS)

LIVE_HTML = r"""<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<title>LiveSD</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html,body { margin:0; height:100%; background:#000; font-family:-apple-system,system-ui,sans-serif; color:#fff; overflow:hidden; }
  #result { position:fixed; inset:0; width:100vw; height:100dvh; object-fit:cover; background:#000; z-index:1; }
  #camWrap { position:fixed; right:10px; bottom:10px; width:30vw; max-width:160px; aspect-ratio:3/4;
             border-radius:14px; overflow:hidden; z-index:30; box-shadow:0 4px 18px rgba(0,0,0,.6); border:2px solid rgba(255,255,255,.25); }
  #cam { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }
  #topbar { position:fixed; top:0; left:0; right:0; z-index:40; display:flex; gap:8px; align-items:center;
            padding:10px; background:linear-gradient(rgba(0,0,0,.55),rgba(0,0,0,0)); }
  select,button,input { font-size:15px; }
  select { background:rgba(20,20,20,.85); color:#fff; border:none; border-radius:10px; padding:8px 10px; }
  .pill { background:rgba(20,20,20,.85); color:#fff; border:none; border-radius:10px; padding:8px 12px; }
  #fps { margin-left:auto; font-size:12px; opacity:.8; }
  #panel { position:fixed; top:58px; left:8px; right:8px; z-index:39; background:rgba(18,18,18,.92);
           border-radius:14px; padding:10px; display:none; max-height:74dvh; overflow:auto; }
  #panel.open { display:block; }
  #panel input[type=text]{ width:100%; padding:10px; border-radius:10px; border:1px solid #444; background:#111; color:#fff; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
  .chip { background:#2b2b2b; color:#fff; border:1px solid #444; border-radius:999px; padding:8px 12px; }
  .chip:active { background:#444; }
  .row { display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap:wrap; }
  label.s { font-size:13px; opacity:.85; }
  #cap { background:#5b6cff; border:none; color:#fff; border-radius:10px; padding:10px 14px; font-weight:600; }
  #capOverlay { position:fixed; inset:0; z-index:80; background:rgba(0,0,0,.92); display:none; flex-direction:column; align-items:center; justify-content:center; gap:12px; padding:14px; }
  #capOverlay.open { display:flex; }
  #capImg { max-width:100%; max-height:80vh; border-radius:12px; }
  .hint { position:fixed; bottom:18px; left:0; right:0; text-align:center; z-index:20; opacity:.85; font-size:14px; pointer-events:none; }
</style></head><body>
<img id="result" alt="">
<div id="camWrap"><video id="cam" autoplay playsinline muted></video></div>

<div id="topbar">
  <button class="pill" id="gear">⚙︎</button>
  <select id="mode">
    <option>Lite</option><option>Wardrobe</option><option>Try-On</option>
  </select>
  <button class="pill" id="flip">📷 Front</button>
  <a class="pill" href="/" style="text-decoration:none">UI</a>
  <span id="fps">starting…</span>
</div>

<div id="panel">
  <input id="prompt" type="text" placeholder="Describe what you want…">
  <div class="chips" id="chips"></div>
  <div class="row" id="garmentRow" style="display:none">
    <label class="s">Garment image:</label>
    <input id="garmentFile" type="file" accept="image/*">
  </div>
  <div class="row">
    <label class="s" id="intLbl">Intensity</label>
    <input id="intensity" type="range" min="0.2" max="1.2" step="0.05" value="0.6" style="flex:1">
  </div>
  <div class="row">
    <label class="s">Smoothness</label>
    <input id="smooth" type="range" min="0" max="0.8" step="0.05" value="0.3" style="flex:1">
    <label class="s"><input type="checkbox" id="keepbg"> Keep background</label>
  </div>
  <div class="row" id="capRow" style="display:none">
    <button id="cap">📸 Capture (HQ)</button>
    <span class="s" id="capStatus"></span>
  </div>
</div>

<div class="hint" id="hint">Tap the camera tile, allow access, then it starts.</div>
<div id="capOverlay"><img id="capImg" alt=""><div class="row"><a id="capDl" class="pill" download="livesd.jpg">Download</a><button class="pill" id="capClose">Close</button></div></div>

<script>
const STYLE_PRESETS = __STYLE_PRESETS__;
const GARMENT_PRESETS = __GARMENT_PRESETS__;
const DEFAULT_NEG = "blurry, distorted, deformed, ugly, low quality, extra limbs";
const WARDROBE_NEG = "blurry, distorted, deformed, extra limbs, low quality, naked, nsfw";

const cam = document.getElementById('cam'), result = document.getElementById('result');
const modeSel = document.getElementById('mode'), promptEl = document.getElementById('prompt');
const chips = document.getElementById('chips'), fpsEl = document.getElementById('fps');
const intensity = document.getElementById('intensity'), smooth = document.getElementById('smooth');
const keepbg = document.getElementById('keepbg'), hint = document.getElementById('hint');
let negative = DEFAULT_NEG, garmentData = null, facing = 'user', stream = null;

document.getElementById('gear').onclick = () => document.getElementById('panel').classList.toggle('open');

function buildChips(){
  const mode = modeSel.value;
  const isWardrobe = mode === 'Wardrobe', isTryOn = mode === 'Try-On';
  document.getElementById('garmentRow').style.display = isTryOn ? 'flex' : 'none';
  document.getElementById('capRow').style.display = (isWardrobe || isTryOn) ? 'flex' : 'none';
  document.getElementById('intLbl').textContent = isTryOn ? 'Garment influence' : 'Intensity';
  chips.innerHTML = '';
  const set = isWardrobe ? GARMENT_PRESETS : (isTryOn ? {} : STYLE_PRESETS);
  for (const name in set){
    const b = document.createElement('button'); b.className='chip'; b.textContent=name;
    b.onclick = () => {
      if (isWardrobe){ promptEl.value = GARMENT_PRESETS[name]; negative = WARDROBE_NEG; }
      else { const p = STYLE_PRESETS[name]; promptEl.value = p.prompt; negative = p.negative; intensity.value = p.strength; }
    };
    chips.appendChild(b);
  }
}
modeSel.onchange = buildChips;
buildChips();
promptEl.value = STYLE_PRESETS['Disney / Pixar'].prompt;

document.getElementById('garmentFile').onchange = (e) => {
  const f = e.target.files[0]; if(!f) return;
  const r = new FileReader();
  r.onload = async () => {
    garmentData = r.result;
    fpsEl.textContent = 'describing garment…';
    try {
      const resp = await fetch('/caption', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({image: garmentData})});
      if (resp.ok) { const j = await resp.json(); if (j.caption) promptEl.value = j.caption; }
    } catch(err) {}
    fpsEl.textContent = '…';
  };
  r.readAsDataURL(f);
};

async function startCam(){
  try{
    if (stream) stream.getTracks().forEach(t=>t.stop());
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing}, width:{ideal:640}, height:{ideal:480}}, audio:false});
    cam.srcObject = stream; hint.style.display='none';
    cam.style.transform = facing==='user' ? 'scaleX(-1)' : 'none';   // mirror only the selfie cam
    document.getElementById('flip').textContent = facing==='user' ? '📷 Front' : '📷 Back';
  }catch(err){ hint.textContent = 'Camera error: ' + err.message; }
}
document.getElementById('flip').onclick = () => { facing = (facing==='user'?'environment':'user'); startCam(); };
document.getElementById('camWrap').onclick = startCam;
startCam();

const canvas = document.createElement('canvas');
function grabFrame(){
  const w = cam.videoWidth, h = cam.videoHeight; if(!w||!h) return null;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d'); ctx.drawImage(cam,0,0,w,h);
  return canvas.toDataURL('image/jpeg', 0.9);
}
function payload(){
  const mode = modeSel.value;
  return { image: grabFrame(), mode, prompt: promptEl.value, negative,
    strength: parseFloat(intensity.value), steps: 2, stabilize: true,
    smoothing: parseFloat(smooth.value), keep_bg: keepbg.checked,
    ip_scale: parseFloat(intensity.value), garment: (mode==='Try-On'? garmentData : null) };
}
let running = true, last = performance.now(), ema = 0;
async function loop(){
  while(running){
    const p = payload();
    if(!p.image){ await new Promise(r=>setTimeout(r,150)); continue; }
    try{
      const r = await fetch('/infer', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p)});
      if(r.ok){
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const old = result.src; result.src = url; if(old && old.startsWith('blob:')) URL.revokeObjectURL(old);
        const now = performance.now(); const inst = 1000/(now-last); last = now; ema = ema? ema*0.8+inst*0.2 : inst;
        fpsEl.textContent = ema.toFixed(1)+' fps';
      } else if(r.status===409){ fpsEl.textContent = 'attach a garment'; await new Promise(r=>setTimeout(r,400)); }
    }catch(e){ fpsEl.textContent='…'; }
    await new Promise(r=>setTimeout(r,20));
  }
}
loop();

// HQ capture
const capOverlay = document.getElementById('capOverlay');
document.getElementById('cap').onclick = async () => {
  const s = document.getElementById('capStatus'); s.textContent = 'rendering…';
  const p = payload();
  try{
    const r = await fetch('/capture', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p)});
    if(r.ok){ const blob = await r.blob(); const url = URL.createObjectURL(blob);
      document.getElementById('capImg').src = url; document.getElementById('capDl').href = url; capOverlay.classList.add('open'); s.textContent=''; }
    else s.textContent = 'capture failed';
  }catch(e){ s.textContent = 'error'; }
};
document.getElementById('capClose').onclick = () => capOverlay.classList.remove('open');
</script></body></html>"""
LIVE_HTML = (LIVE_HTML.replace("__STYLE_PRESETS__", _STYLE_PRESETS_JS)
                      .replace("__GARMENT_PRESETS__", _GARMENT_PRESETS_JS))


# Capability probe (open on the Quest to see what the browser allows)
PROBE_HTML = r"""<!doctype html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiveSD capability probe</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;padding:16px;line-height:1.5}
button{font-size:18px;padding:12px 18px;margin:8px 0;border-radius:10px;border:none;background:#5b6cff;color:#fff}
pre{background:#000;padding:12px;border-radius:10px;white-space:pre-wrap;word-break:break-word}
.ok{color:#5fdd7a}.no{color:#ff7a7a}</style></head><body>
<h2>LiveSD capability probe</h2>
<p>Tap each test. Screenshot the results.</p>
<button id="b1">1. Test getUserMedia (camera)</button>
<button id="b2">2. Test WebXR support</button>
<button id="b3">3. Test WebXR camera-access (enters AR briefly)</button>
<pre id="out">results will appear here…</pre>
<script>
const out = document.getElementById('out');
function log(s){ out.textContent += "\n" + s; }
document.getElementById('b1').onclick = async () => {
  log("\n--- getUserMedia ---");
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const vids = devs.filter(d=>d.kind==='videoinput');
    log("video input devices: " + vids.length);
    const s = await navigator.mediaDevices.getUserMedia({video:true});
    log("getUserMedia: OK (" + s.getVideoTracks().map(t=>t.label).join(", ") + ")");
    s.getTracks().forEach(t=>t.stop());
  } catch(e){ log("getUserMedia FAILED: " + e.name + " - " + e.message); }
};
document.getElementById('b2').onclick = async () => {
  log("\n--- WebXR support ---");
  if(!navigator.xr){ log("navigator.xr: NOT present"); return; }
  for(const m of ['immersive-vr','immersive-ar']){
    try { log(m + ": " + (await navigator.xr.isSessionSupported(m))); }
    catch(e){ log(m + ": error " + e.message); }
  }
};
document.getElementById('b3').onclick = async () => {
  log("\n--- WebXR camera-access ---");
  if(!navigator.xr){ log("navigator.xr: NOT present"); return; }
  try {
    const s = await navigator.xr.requestSession('immersive-ar', {optionalFeatures:['camera-access']});
    const feats = s.enabledFeatures || [];
    log("session started. enabledFeatures: " + JSON.stringify(feats));
    log("camera-access granted: " + (feats.includes('camera-access') ? "YES ✅" : "NO ❌"));
    await s.end();
  } catch(e){ log("requestSession FAILED: " + e.name + " - " + e.message); }
};
</script></body></html>"""


if __name__ == "__main__":
    import os
    import uvicorn
    from fastapi import FastAPI, Body
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI()

    @app.get("/live")
    def live():
        return HTMLResponse(LIVE_HTML)

    @app.get("/probe")
    def probe():
        return HTMLResponse(PROBE_HTML)

    @app.post("/infer")
    def infer(p: dict = Body(...)):
        try:
            frame = _dataurl_to_np(p.get("image"))
            garment = _dataurl_to_np(p.get("garment"))
            if p.get("mode") == "Try-On" and garment is None:
                return Response(status_code=409)  # need a garment image
            out = run_transform(
                frame, p.get("mode", "Lite"), p.get("prompt", ""), p.get("negative") or None,
                float(p.get("strength", 0.6)), int(p.get("steps", 2)), bool(p.get("stabilize", True)),
                float(p.get("smoothing", 0.3)),
                bool(p.get("keep_bg", False)), garment, float(p.get("ip_scale", 0.8)),
            )
            out = out[0] if isinstance(out, tuple) else out
            if out is None:
                return Response(status_code=204)
            return Response(content=_np_to_jpeg(out, quality=92), media_type="image/jpeg")
        except Exception as e:
            print(f"[/infer] error: {type(e).__name__}: {e}", flush=True)
            return Response(status_code=500)

    @app.post("/caption")
    def caption(p: dict = Body(...)):
        try:
            return {"caption": captioner.caption_garment(_dataurl_to_np(p.get("image")))}
        except Exception as e:
            print(f"[/caption] error: {type(e).__name__}: {e}", flush=True)
            return {"caption": ""}

    @app.post("/capture")
    def capture(p: dict = Body(...)):
        try:
            frame = _dataurl_to_np(p.get("image"))
            garment = _dataurl_to_np(p.get("garment"))
            out = run_capture(frame, p.get("mode", "Wardrobe"), p.get("prompt", ""),
                              p.get("negative") or None, garment)
            if out is None:
                return Response(status_code=204)
            return Response(content=_np_to_jpeg(out, quality=92), media_type="image/jpeg")
        except Exception as e:
            print(f"[/capture] error: {type(e).__name__}: {e}", flush=True)
            return Response(status_code=500)

    # Mount the Gradio UI at "/" (custom routes above take precedence)
    app = gr.mount_gradio_app(app, demo, path="/")

    ssl = os.path.exists("cert.pem") and os.path.exists("key.pem")
    host = "0.0.0.0" if ssl else "127.0.0.1"
    if ssl:
        print("Serving on the LAN:  https://10.0.0.153:7860/        (Gradio UI)")
        print("Immersive (phone):   https://10.0.0.153:7860/live    (FaceTime-style)")
        uvicorn.run(app, host=host, port=7860, ssl_certfile="cert.pem", ssl_keyfile="key.pem")
    else:
        print("UI: http://127.0.0.1:7860/   |   Immersive: http://127.0.0.1:7860/live")
        uvicorn.run(app, host=host, port=7860)
