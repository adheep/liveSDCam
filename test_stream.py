"""Minimal Gradio webcam streaming test - pure passthrough, no diffusion.
If [PASSTHROUGH] lines print when you open the page, streaming wiring works."""
import gradio as gr

n = {"i": 0}


def passthrough(frame):
    n["i"] += 1
    if n["i"] <= 30 or n["i"] % 20 == 0:
        shape = None if frame is None else getattr(frame, "shape", type(frame).__name__)
        print(f"[PASSTHROUGH] frame #{n['i']} shape={shape}", flush=True)
    return frame


with gr.Blocks() as demo:
    gr.Markdown("Minimal webcam streaming test")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="src")
        out = gr.Image(type="numpy", label="mirror", interactive=False)
    cam.stream(passthrough, inputs=[cam], outputs=[out],
               stream_every=0.1, concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
