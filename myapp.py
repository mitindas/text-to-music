import streamlit as st
import numpy as np
from queue import Queue
from threading import Thread
import torch
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, set_seed

# Initialize the model and processor
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", attn_implementation="eager")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

title = "Text to Music!"
st.title(title)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

# Initial setup for progress bar in session state
if 'progress_bar' not in st.session_state:
    st.session_state['progress_bar'] = None

class MusicgenStreamer:
    def __init__(self, model, play_steps=10, device=None):
        self.model = model
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device else model.device
        self.play_steps = play_steps
        self.stride = np.prod(self.audio_encoder.config.upsampling_ratios) * (self.play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None

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

def generate_audio(text_prompt, audio_length_in_s=10.0, play_steps_in_s=2.0):
    max_new_tokens = int(frame_rate * audio_length_in_s)
    play_steps = int(frame_rate * play_steps_in_s)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        if device == "cuda:0":
            model.half()
    inputs = processor(text=text_prompt, padding=True, return_tensors="pt")
    streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)
    streamer.total_steps = max_new_tokens // play_steps
    set_seed(5)  # Fixing seed at 5
    generation_kwargs = dict(**inputs.to(device), streamer=streamer, max_new_tokens=max_new_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    all_audio = []
    for idx, new_audio in enumerate(streamer):
        all_audio.append(new_audio)
        # Update progress bar after each audio clip is concatenated
        if st.session_state.progress_bar:
            progress_value = min((idx + 1) / streamer.total_steps, 1.0)
            st.session_state.progress_bar.progress(progress_value)
    generated_audio = np.concatenate(all_audio)
    return generated_audio

prompt = st.text_input("Enter your text prompt", "80s pop track with synth and instrumentals")
audio_length = st.slider("Audio length in seconds", min_value=10, max_value=30, value=15, step=5)
streaming_interval = st.slider(
    "Streaming interval in seconds", min_value=0.5, max_value=2.5, value=1.5, step=0.5,
    help="Lower = shorter chunks, lower latency, more codec steps"
)

if st.button("Generate"):
    st.session_state.progress_bar = st.progress(0)
    generated_audio = generate_audio(prompt, audio_length, streaming_interval)
    st.audio(generated_audio, format="audio/wav", sample_rate=sampling_rate)
