from __future__ import annotations
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from ollama import Client
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .nodes import _filter_enabled_options, _audio_to_wav_bytes


class OllamaAudioTranscribe:
    """Speech-to-text transcription using an Ollama STT model (e.g. whisper)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to transcribe. Connect from Load Audio or any AUDIO output.",
                }),
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "URL of the Ollama server.",
                }),
                "model": ((), {
                    "tooltip": "Speech-to-text model (e.g. whisper, whisper-large-v3-turbo).",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Optional hint to guide transcription (vocabulary hints, speaker style, etc.).",
                }),
                "language": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BCP-47 language code (e.g. 'en', 'fr', 'ja'). Leave blank for auto-detect.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. 0 = deterministic.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "Ollama/Audio"
    DESCRIPTION = "Transcribe audio to text using an Ollama speech-to-text model such as whisper."

    def transcribe(self, audio, url, model, prompt="", language="", temperature=0.0):
        # Ollama has no native audio API yet. WAV data passed via images[] works because
        # Ollama detects RIFF/WAVE magic bytes and routes it as audio to the model.
        client = Client(host=url)
        audio_b64 = base64.b64encode(_audio_to_wav_bytes(audio)).decode("utf-8")

        transcribe_prompt = prompt if prompt else "Transcribe the audio exactly as spoken."
        if language:
            transcribe_prompt += f" The spoken language is {language}."

        options = {"temperature": temperature} if temperature > 0.0 else None

        response = client.generate(
            model=model,
            prompt=transcribe_prompt,
            images=[audio_b64],
            options=options,
        )
        text = response["response"] if isinstance(response, dict) else response.response
        return (text,)


class OllamaAudioChat:
    """Send audio + text to an audio-capable Ollama multimodal model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe what you hear in this audio.",
                    "tooltip": "Text prompt sent alongside the audio.",
                }),
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "System prompt.",
                }),
                "think": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable chain-of-thought reasoning before answering (model must support this).",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to send alongside the text prompt. If omitted, acts as a text-only chat.",
                }),
                "connectivity": ("OLLAMA_CONNECTIVITY", {
                    "forceInput": False,
                    "tooltip": "Ollama Connectivity node — provides URL, model, and keep_alive settings.",
                }),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False}),
                "meta": ("OLLAMA_META", {
                    "forceInput": False,
                    "tooltip": "Chain from another Ollama node to inherit connectivity and options.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "meta",)
    FUNCTION = "audio_chat"
    CATEGORY = "Ollama/Audio"
    DESCRIPTION = "Send audio and text to an audio-capable Ollama model (e.g. gemma3n, qwen2-audio). Connect an Ollama Connectivity node for server/model selection."

    def audio_chat(self, prompt, system, think, audio=None, connectivity=None, options=None, meta=None):
        if connectivity is None and meta is None:
            raise ValueError("Either 'connectivity' or 'meta' must be provided.")

        if meta is not None:
            if connectivity is not None:
                meta["connectivity"] = connectivity
            if options is not None:
                meta["options"] = options
        else:
            meta = {"connectivity": connectivity, "options": options}

        conn = meta["connectivity"]
        url = conn["url"]
        model = conn["model"]
        keep_alive_unit = "m" if conn.get("keep_alive_unit", "minutes") == "minutes" else "h"
        request_keep_alive = str(conn.get("keep_alive", 5)) + keep_alive_unit

        client = Client(host=url)
        request_options = _filter_enabled_options(options)

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        # Build images list: audio (as WAV via RIFF magic bytes) goes first, before any images
        chat_images: list[str] = []
        if audio is not None:
            chat_images.append(base64.b64encode(_audio_to_wav_bytes(audio)).decode("utf-8"))
            # Cap num_ctx at 8192 — audio embeddings cause memory overflow above this
            if request_options is None:
                request_options = {}
            request_options["num_ctx"] = min(request_options.get("num_ctx", 8192), 8192)

        user_msg: dict[str, Any] = {"role": "user", "content": prompt}
        if chat_images:
            user_msg["images"] = chat_images
        messages.append(user_msg)

        response = client.chat(
            model=model,
            messages=messages,
            options=request_options,
            keep_alive=request_keep_alive,
            think=think,
        )

        result_text = response.message.content if hasattr(response, "message") else ""
        thinking_text = response.message.thinking if (think and hasattr(response, "message")) else None

        if conn.get("unload_after", False):
            from .nodes import _unload_model
            _unload_model(client, model)

        return (result_text, thinking_text, meta,)


class OllamaVideoAnalyze:
    """Sample frames from a video IMAGE batch and analyze with an Ollama vision model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames as an IMAGE batch (e.g. from VHS Load Video or any frame sequence node).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe what happens in this video clip.",
                    "tooltip": "Prompt for the vision model.",
                }),
                "max_frames": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Maximum frames to sample and send. More = richer context but slower and higher memory.",
                }),
                "sample_mode": (["uniform", "first", "last"], {
                    "tooltip": "Sampling strategy: uniform=evenly spaced across entire clip, first=opening frames, last=closing frames.",
                }),
            },
            "optional": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are analyzing frames sampled from a video clip. Describe the action, subjects, and content.",
                    "tooltip": "System prompt for the model.",
                }),
                "think": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable chain-of-thought reasoning (model must support this).",
                }),
                "format": (["text", "json"], {
                    "tooltip": "Output format.",
                }),
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False}),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False}),
                "meta": ("OLLAMA_META", {"forceInput": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "meta",)
    FUNCTION = "analyze_video"
    CATEGORY = "Ollama/Video"
    DESCRIPTION = "Sample frames from a video and analyze them with an Ollama vision model. Works with any IMAGE batch representing a frame sequence."

    def analyze_video(self, frames, prompt, max_frames, sample_mode, system="", think=False, format="text", connectivity=None, options=None, meta=None):
        if connectivity is None and meta is None:
            raise ValueError("Either 'connectivity' or 'meta' must be provided.")

        if meta is not None:
            if connectivity is not None:
                meta["connectivity"] = connectivity
            if options is not None:
                meta["options"] = options
        else:
            meta = {"connectivity": connectivity, "options": options}

        conn = meta["connectivity"]
        url = conn["url"]
        model = conn["model"]
        keep_alive_unit = "m" if conn.get("keep_alive_unit", "minutes") == "minutes" else "h"
        request_keep_alive = str(conn.get("keep_alive", 5)) + keep_alive_unit

        client = Client(host=url)
        request_options = _filter_enabled_options(options)
        debug_print = options is not None and options.get("debug", False)

        # Sample frame indices
        total = frames.shape[0]
        if sample_mode == "uniform":
            indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        elif sample_mode == "first":
            indices = np.arange(min(max_frames, total), dtype=int)
        else:  # last
            indices = np.arange(max(0, total - min(max_frames, total)), total, dtype=int)

        images_b64: list[str] = []
        for idx in indices:
            frame = frames[int(idx)]
            i = 255.0 * frame.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            images_b64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        if debug_print:
            print(f"--- OllamaVideoAnalyze: sending {len(images_b64)}/{total} frames to {model} @ {url}")

        ollama_format = "json" if format == "json" else ""

        response = client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images_b64,
            think=think,
            options=request_options,
            keep_alive=request_keep_alive,
            format=ollama_format,
        )

        result_text = response["response"]
        thinking_text = response.get("thinking") if think else None

        if conn.get("unload_after", False):
            from .nodes import _unload_model
            _unload_model(client, model, debug=debug_print)

        return (result_text, thinking_text, meta,)


NODE_CLASS_MAPPINGS = {
    "OllamaAudioTranscribe": OllamaAudioTranscribe,
    "OllamaAudioChat": OllamaAudioChat,
    "OllamaVideoAnalyze": OllamaVideoAnalyze,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaAudioTranscribe": "Ollama Audio Transcribe",
    "OllamaAudioChat": "Ollama Audio Chat",
    "OllamaVideoAnalyze": "Ollama Video Analyze",
}