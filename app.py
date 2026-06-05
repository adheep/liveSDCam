"""LiveSD - real-time webcam img2img.
Left: raw webcam.  Right: live SD-Turbo transformation.  Type a prompt below.

Three modes:
  Lite     - plain SD-Turbo img2img (fast, default; keeps original incl. beard)
  Advanced - + Canny ControlNet structure-lock + temporal smoothing
             (keeps you recognizable, more consistent; lazy-loaded)
  Wardrobe - SD1.5 LCM-inpaint of the clothing region only (change the outfit,
             keep face/pose/background); live preview + HQ Capture button.

Run:  .venv\\Scripts\\python.exe app.py
"""
import threading

import gradio as gr

from engine import Engine
from engine_advanced import AdvancedEngine
from engine_wardrobe import WardrobeEngine

engine = Engine()
engine.load()

# Heavy engines (ControlNet / inpaint) -> lazy load on first switch.
adv_engine = AdvancedEngine()
wardrobe_engine = WardrobeEngine()
_adv_lock = threading.Lock()
_wardrobe_lock = threading.Lock()


def ensure_advanced():
    if not adv_engine.loaded:
        with _adv_lock:
            if not adv_engine.loaded:
                adv_engine.load()


def ensure_wardrobe():
    if not wardrobe_engine.loaded:
        with _wardrobe_lock:
            if not wardrobe_engine.loaded:
                wardrobe_engine.load()


DEFAULT_NEG = "blurry, distorted, deformed, ugly, low quality, extra limbs"

# Each preset: (prompt, negative, recommended_strength).
# A strong, targeted negative is what makes big identity changes (like Girl) actually take.
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
WARDROBE_PRESETS = {
    "Suit": "wearing an elegant tailored business suit and tie, formal, detailed fabric",
    "Tuxedo": "wearing a black tuxedo with a bow tie, formal eveningwear, detailed",
    "Kurta": "wearing a traditional white kurta, indian ethnic wear, detailed embroidery",
    "Hoodie": "wearing a casual cozy hoodie, streetwear, detailed fabric",
    "Leather jacket": "wearing a black leather biker jacket, edgy, detailed",
    "T-shirt": "wearing a plain casual cotton t-shirt, detailed fabric",
}


def stream_fn(frame, mode, prompt, negative, strength, steps, stabilize, ctrl_scale, smoothing, keep_bg):
    if mode == "Advanced":
        ensure_advanced()
        return adv_engine.transform(frame, prompt, strength, steps, stabilize,
                                    ctrl_scale, smoothing, negative=negative, keep_bg=keep_bg)
    if mode == "Wardrobe":
        ensure_wardrobe()
        # inpaint: only the garment region changes; you stay you.
        return wardrobe_engine.transform(frame, prompt, steps, stabilize,
                                         smoothing, negative=negative)
    return engine.transform(frame, prompt, strength, steps, stabilize,
                            negative=negative, keep_bg=keep_bg)


def capture_fn(frame, mode, prompt, negative):
    """High-quality one-shot render of the current frame (Wardrobe HQ inpaint)."""
    ensure_wardrobe()
    return wardrobe_engine.capture(frame, prompt, negative or WARDROBE_NEG)


NOTES = {
    "Lite": "**Lite**: fast plain img2img. Keeps your original (beard, features) — best for style filters.",
    "Advanced": "**Advanced**: structure-locked to your webcam edges + temporal smoothing. Keeps you recognizable. First switch loads ControlNet (~8s).",
    "Wardrobe": "**Wardrobe**: changes ONLY your clothes (inpaint) — face, pose, background untouched. Pick a garment, watch the live preview (~3 fps), then hit **📸 Capture (HQ)** for a clean result. First switch loads inpaint models (~10s).",
}


def on_mode_change(mode):
    """Toggle mode-specific controls and reset temporal history on switch."""
    if mode == "Advanced":
        adv_engine.reset_temporal()
    elif mode == "Wardrobe":
        wardrobe_engine.reset_temporal()

    is_lite = mode == "Lite"
    is_adv = mode == "Advanced"
    is_wardrobe = mode == "Wardrobe"
    return (
        gr.update(visible=is_lite or is_adv),    # strength
        gr.update(visible=is_adv),               # ctrl_scale (Advanced only)
        gr.update(visible=is_adv or is_wardrobe),# smoothing
        gr.update(visible=is_lite or is_adv),    # keep_bg
        gr.update(visible=not is_wardrobe),      # style preset row
        gr.update(visible=is_wardrobe),          # garment preset row
        gr.update(visible=is_wardrobe),          # capture row
        NOTES[mode],                             # mode note
    )


with gr.Blocks(title="LiveSD") as demo:
    gr.Markdown("# 🎥 LiveSD — Real-time Webcam Transformation")
    gr.Markdown("Click the **⏺ Record** button on the webcam to start streaming.")

    mode = gr.Radio(["Lite", "Advanced", "Wardrobe"], value="Lite", label="Mode")
    mode_note = gr.Markdown(
        "**Lite**: fast plain img2img. Keeps your original (beard, features) — best for style filters.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Source (webcam)")
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                           label="", height=512, show_label=False)
        with gr.Column():
            gr.Markdown("### Transformed")
            out = gr.Image(type="numpy", label="", height=512,
                           show_label=False, interactive=False)
            fps = gr.Textbox(label="Performance", interactive=False, value="warming up...")

    with gr.Accordion("⚙️ Prompt & Settings", open=False):
        prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=2,
                            placeholder="Describe what you want to become...")
        negative = gr.Textbox(label="Negative prompt (what to avoid)", value=DEFAULT_NEG, lines=1)

        # Style presets (Lite/Advanced)
        with gr.Row() as style_row:
            preset_buttons = [(name, gr.Button(name, size="sm")) for name in PRESETS]

        # Garment presets (Wardrobe only)
        with gr.Row(visible=False) as garment_row:
            garment_buttons = [(name, gr.Button(name, size="sm")) for name in WARDROBE_PRESETS]

        with gr.Row():
            strength = gr.Slider(0.2, 0.95, value=0.6, step=0.05, label="Strength (likeness ↔ transformation)")
            steps = gr.Slider(1, 4, value=2, step=1, label="Steps (speed ↔ quality)")
            stabilize = gr.Checkbox(value=True, label="Stabilize (fixed seed)")
            keep_bg = gr.Checkbox(value=False, label="Keep background (person only — Lite/Advanced)")

        with gr.Row():
            ctrl_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, visible=False,
                                   label="Structure lock (ControlNet scale)")
            smoothing = gr.Slider(0.0, 0.8, value=0.4, step=0.05, visible=False,
                                  label="Temporal smoothing (anti-flicker ↔ motion lag)")

    # Capture (Wardrobe only): high-quality one-shot of the current frame
    with gr.Accordion("📸 Capture (HQ)", open=False, visible=False) as capture_row:
        with gr.Row():
            capture_btn = gr.Button("Capture current frame", variant="primary", scale=1)
            capture_out = gr.Image(type="numpy", label="Capture result (HQ)", height=512,
                                   interactive=False, scale=2)

    # Style preset click -> prompt + targeted negative + recommended strength
    for name, btn in preset_buttons:
        btn.click(lambda n=name: (PRESETS[n][0], PRESETS[n][1], PRESETS[n][2]),
                  outputs=[prompt, negative, strength])
    # Garment preset click -> prompt + clothing-safe negative
    for name, btn in garment_buttons:
        btn.click(lambda n=name: (WARDROBE_PRESETS[n], WARDROBE_NEG),
                  outputs=[prompt, negative])

    capture_btn.click(capture_fn, inputs=[cam, mode, prompt, negative], outputs=[capture_out])

    mode.change(on_mode_change, inputs=[mode],
                outputs=[strength, ctrl_scale, smoothing, keep_bg,
                         style_row, garment_row, capture_row, mode_note])

    cam.stream(
        stream_fn,
        inputs=[cam, mode, prompt, negative, strength, steps, stabilize, ctrl_scale, smoothing, keep_bg],
        outputs=[out, fps],
        stream_every=0.05,        # offer frames quickly; engine drops extras when busy
        concurrency_limit=1,      # one inference at a time
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=gr.themes.Soft())
