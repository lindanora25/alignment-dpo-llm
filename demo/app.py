
import json
from pathlib import Path
import gradio as gr

DATA_PATH = Path("artifacts/example_generations.json")
data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
items = data["items"]
id_to_item = {it["id"]: it for it in items}
ids = list(id_to_item.keys())

def show(prompt_id: str, variant: str):
    it = id_to_item[prompt_id]
    prompt = it["prompt"]
    outs = it["outputs"]
    base = outs["base"][variant]
    sft  = outs["sft"][variant]
    dpo  = outs["dpo"][variant]

    auto = it.get("auto_eval", {})
    def fmt_eval(m):
        e = auto.get(m, {}).get(variant)
        if not e: return "N/A"
        return f"pass_all={e.get('pass_all')} | checks={e.get('checks')}"
    eval_text = (
        f"**Base:** {fmt_eval('base')}\n\n"
        f"**SFT:** {fmt_eval('sft')}\n\n"
        f"**DPO:** {fmt_eval('dpo')}\n"
    )
    return prompt, base, sft, dpo, eval_text

with gr.Blocks() as demo:
    gr.Markdown("# Base vs SFT vs DPO — Replay Demo (from artifacts)")
    gr.Markdown("This demo replays saved generations from `artifacts/example_generations.json` (no GPU required).")

    with gr.Row():
        pid = gr.Dropdown(ids, value=ids[0], label="Prompt ID")
        variant = gr.Radio(["det", "samp"], value="det", label="Decoding variant")

    prompt = gr.Textbox(label="Prompt", lines=6)
    with gr.Row():
        base = gr.Textbox(label="Base", lines=14)
        sft = gr.Textbox(label="SFT", lines=14)
        dpo = gr.Textbox(label="DPO", lines=14)
    eval_box = gr.Markdown()

    btn = gr.Button("Show")
    btn.click(show, inputs=[pid, variant], outputs=[prompt, base, sft, dpo, eval_box])

demo.launch(share=True)

