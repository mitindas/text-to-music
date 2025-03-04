import numpy as np
import torch
import random
from queue import Queue
from threading import Thread

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, set_seed

import gradio as gr
import spaces

# Initialize the model and processor
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", attn_implementation="eager")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

title = "Text to Music!"
sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate


class MusicgenStreamer:
    def __init__(self, model, play_steps=10, device=None):
        self.model = model
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device else model.device
        self.play_steps = play_steps
        self.stride = np.prod(self.audio_encoder.config.upsampling_ratios) * (
                    self.play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None
        self.total_steps = 0

    def apply_delay_pattern_mask(self, input_ids):
        _, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1]
        )
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)
        input_ids = input_ids[input_ids != self.generation_config.pad_token_id].reshape(
            1, self.decoder.num_codebooks, -1)
        input_ids = input_ids[None, ...].to(self.audio_encoder.device)
        output_values = self.audio_encoder.decode(input_ids, audio_scales=[None])
        return output_values.audio_values[0, 0].cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("MusicgenStreamer only supports batch size 1")
        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)
        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            stream_end = False
            self.on_finalized_audio(audio_values[self.to_yield:-self.stride], stream_end)
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)
        stream_end = True
        self.on_finalized_audio(audio_values[self.to_yield:], stream_end)

    def on_finalized_audio(self, audio, stream_end):
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value


@spaces.GPU()
def generate_audio(text_prompt, audio_length_in_s=10.0):
    max_new_tokens = int(frame_rate * audio_length_in_s)
    play_steps = int(frame_rate * 1.5)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device != model.device:
        model.to(device)
        if device == "cuda:0":
            model.half()

    inputs = processor(text=text_prompt, padding=True, return_tensors="pt")
    streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)
    streamer.total_steps = max_new_tokens // play_steps

    # Use random seed instead of fixed seed
    random_seed = random.randint(1, 1000)
    set_seed(random_seed)

    generation_kwargs = dict(**inputs.to(device), streamer=streamer, max_new_tokens=max_new_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    all_audio = []
    for new_audio in streamer:
        all_audio.append(new_audio)

    generated_audio = np.concatenate(all_audio)
    return sampling_rate, generated_audio

examples = [
    ["An 80s driving pop song with heavy drums and synth pads in the background", 30, 1.5, 5],
    ["A cheerful country song with acoustic guitars", 30, 1.5, 5],
    ["90s rock song with electric guitar and heavy drums", 30, 1.5, 5],
    ["a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130", 30, 1.5, 5],
    ["lofi slow bpm electro chill with organic samples", 30, 1.5, 5],
]

css = """

.gradio-container h1 {
    font-size: 3em !important;
    color: white !important;
}
.gradio-container {
    background-color: #ec5c0c !important;
}
.primary-button {
    color: #274c77 !important;
    border-color: #274c77 !important;
}
.primary-button:hover {
    background-color: #1f3d5f !important;
    border-color: #1f3d5f !important;
}
.border {
    border-color: #274c77 !important;
}
"""

# Use Blocks for vertical layout
with gr.Blocks(css=css, title=title) as demo:
    gr.Markdown("# Text to Music!")
    with gr.Column():
        text_prompt = gr.Text(label="Text Prompt", value="80s pop track with synth and instrumentals")
        audio_length = gr.Slider(10, 30, value=15, step=5, label="Audio Length (seconds)")
        generate_btn = gr.Button("Generate Music")
        output_audio = gr.Audio(label="Generated Music", interactive=False)

    # Add examples in a box
    with gr.Group():
        gr.Markdown("\n\n## &nbsp; Prompt Examples \n\n")
        example_prompts = gr.Markdown("""
         &nbsp;&nbsp;**80s Pop**: An 80s driving pop song with heavy drums and synth pads in the background \n
         &nbsp;&nbsp;**Country**: A cheerful country song with acoustic guitars\n
         &nbsp;&nbsp;**90s Rock**: 90s rock song with electric guitar and heavy drums\n
         &nbsp;&nbsp;**EDM**: A light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130\n
         &nbsp;&nbsp;**Lo-Fi**: Lofi slow bpm electro chill with organic samples\n\n\n
         """)


    generate_btn.click(
        fn=generate_audio,
        inputs=[text_prompt, audio_length],
        outputs=output_audio
    )


demo.launch()