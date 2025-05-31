#!/usr/bin/env python3

import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import gradio as gr
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='Enable CUDA if available (fallback to CPU if unavailable)')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_orig_torch_load = torch.load
def _load_device(f, *args, **kwargs):
    kwargs.pop('map_location', None)
    return _orig_torch_load(f, *args, map_location=device, **kwargs)
torch.load = _load_device

print(f"Initializing TTS model on {device} (this may take a moment)...")
with torch.device(device):
    model = ChatterboxTTS.from_pretrained(device=device)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    return model

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = ChatterboxTTS.from_pretrained(device=device)
    if seed_num != 0:
        set_seed(int(seed_num))
    wav = model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=exaggeration, temperature=temperature, cfg_weight=cfgw)
    return (model.sr, wav.squeeze(0).numpy())

with gr.Blocks() as demo:
    model_state = gr.State(None)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.", label="Text to synthesize (max chars 300)", max_lines=5)
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)
            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
    demo.load(fn=load_model, inputs=[], outputs=model_state)
    run_btn.click(fn=generate, inputs=[model_state, text, ref_wav, exaggeration, temp, seed_num, cfg_weight], outputs=audio_output)

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch(share=False)