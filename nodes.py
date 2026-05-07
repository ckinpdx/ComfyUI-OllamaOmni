from __future__ import annotations
import copy
import io
import random
import re
import wave

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""


# Dictionary global per session_id
CHAT_SESSIONS: dict[str, ChatSession] = {}

# Function to filter enabled options
def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the ollama options whose 'enable_*' flag is True."""
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None


def _unload_model(client: Client, model: str, debug: bool = False) -> bool:
    """
    Force unload a model from Ollama's VRAM by sending a generate request 
    with keep_alive=0. This is the only way to actually free VRAM.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Send a minimal request with keep_alive=0 to force unload
        # An empty prompt with keep_alive=0 triggers immediate unload
        client.generate(
            model=model,
            prompt="",  # Empty prompt
            keep_alive=0,  # Immediate unload
        )
        if debug:
            print(f"Successfully unloaded model '{model}' from VRAM")
        return True
    except Exception as e:
        if debug:
            print(f"Failed to unload model '{model}': {e}")
        return False


def _audio_to_wav_bytes(audio_dict: dict, target_sr: int = 16000) -> bytes:
    """Convert a ComfyUI AUDIO dict to mono 16kHz 16-bit WAV bytes.

    Ollama detects audio by RIFF/WAVE magic bytes in the images[] field.
    Requirements: WAV with full RIFF header, 16kHz mono (raw PCM fails silently).
    """
    waveform = audio_dict["waveform"]   # [batch, channels, samples]
    sample_rate = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform[0]

    audio_np = waveform.cpu().numpy()

    if audio_np.ndim == 2:             # mix to mono
        audio_np = audio_np.mean(axis=0)

    # Resample to target_sr (Ollama requires 16kHz)
    if sample_rate != target_sr:
        num_out = int(round(len(audio_np) * target_sr / sample_rate))
        audio_np = np.interp(
            np.linspace(0, len(audio_np) - 1, num_out),
            np.arange(len(audio_np)),
            audio_np,
        )

    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")
    client = Client(host=url)

    models = client.list().get('models', [])

    try:
        models = [model['model'] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model['name'] for model in models]
        return web.json_response(models)


@PromptServer.instance.routes.post("/ollama/unload_model")
async def unload_model_endpoint(request):
    """API endpoint to unload a model from VRAM"""
    data = await request.json()
    url = data.get("url", "http://127.0.0.1:11434")
    model = data.get("model")
    
    if not model:
        return web.json_response({"success": False, "error": "No model specified"})
    
    client = Client(host=url)
    success = _unload_model(client, model, debug=True)
    return web.json_response({"success": success, "model": model})


class OllamaUnloadModel:
    """
    Force unload an Ollama model from VRAM (passthrough node).
    
    Connect meta from Ollama Generate/Chat (for connection info) and 
    pass your result through to the next node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "Connect result text here - passes through unchanged"}),
                "meta": ("OLLAMA_META", {"tooltip": "Meta from Ollama Generate/Chat - used to get model/url info"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "unload_model"
    CATEGORY = "Ollama"
    DESCRIPTION = "Force unload an Ollama model from VRAM. Connect meta for connection info, pass your result through to continue the workflow."

    def unload_model(self, text, meta):
        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        debug = True if meta.get("options") is not None and meta["options"]["debug"] else False
        
        client = Client(host=url)
        _unload_model(client, model, debug=debug)
        
        return (text,)


class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path + ".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max":2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),

                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),

                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),

                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the size of the context window used to generate the next token."}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = num_ctx)"}),

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),

                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),

                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return."}),

                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),

                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),

                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out."}),

                "debug": ("BOOLEAN", {"default": False, "tooltip": "For debugging purposes of the custom nodes, no effect on ollama api."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"
    DESCRIPTION = "Various settings for advanced configuration of Ollama inference. See Ollama documentation for more details."

    def ollama_options(self, **kargs):

        if kargs['debug']:
            print("--- ollama options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs,)


class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
                "unload_after": ("BOOLEAN", {"default": False, "tooltip": "If enabled, force unload the model from VRAM after generation completes. This ensures VRAM is freed even if keep_alive > 0."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit, unload_after=False):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
            "unload_after": unload_after,
        }

        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'"
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation. This is useful for multi-turn conversations or tasks that require context."}),
                "format": (["text", "json", "json_schema"], {"tooltip": "Output format. 'text' = plain text, 'json' = encourages JSON output, 'json_schema' = constrained decoding against a schema (paste a JSON Schema into the json_schema input â€” guarantees output matches the schema)."}),

            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False, "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set."},),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False, "tooltip": "Connect an Ollama Options node for advanced inference configuration."},),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or batch of images/video frames for vision tasks. For video, connect a frame batch and set max_frames to limit how many are sent."},),
                "audio": ("AUDIO", {"forceInput": False, "tooltip": "Provide audio for omni-modal models (e.g. Nemotron Omni). Converted to mono WAV before sending."},),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "forceInput": True, "tooltip": "When images is a video frame batch, limit how many frames are sampled and sent (uniform sampling). Leave unconnected for no limit (sends all frames)."},),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."},),
                "meta": ("OLLAMA_META", {"forceInput": False, "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along."},),
                "json_schema": ("STRING", {"forceInput": False, "multiline": True, "default": "", "tooltip": "JSON Schema string for constrained output. Only used when format is set to 'json_schema'."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_generate_v2(self, system, prompt, think, keep_context, format, context=None, options=None, connectivity=None, images=None, audio=None, max_frames=0, meta=None, json_schema=""):

        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None: # bypass the current meta connectivity
                meta["connectivity"] = connectivity
            if options is not None: # bypass the current meta options
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        url = meta['connectivity']['url']
        model = meta['connectivity']['model']
        client = Client(host=url)

        debug_print = True if meta['options'] is not None and meta['options']['debug'] else False
        
        # Check if we should unload after generation
        unload_after = meta['connectivity'].get('unload_after', False)

        if format == "json_schema":
            import json as _json
            format = _json.loads(json_schema)
        elif format == "text":
            format = ''

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit =  'm' if meta['connectivity']['keep_alive_unit'] == "minutes" else 'h'
        request_keep_alive = str(meta['connectivity']['keep_alive']) + keep_alive_unit

        request_options = _filter_enabled_options(options)

        images_b64 = None
        if images is not None:
            total_frames = images.shape[0]
            if max_frames > 0 and total_frames > max_frames:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
                frame_batch = images[indices]
            else:
                frame_batch = images
            images_b64 = []
            for (batch_number, image) in enumerate(frame_batch):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                images_b64.append(str(img_bytes, 'utf-8'))

        # Audio passed via images[] — Ollama identifies WAV by RIFF magic bytes
        audio_b64 = None
        if audio is not None:
            audio_b64 = base64.b64encode(_audio_to_wav_bytes(audio)).decode('utf-8')
            if images_b64 is None:
                images_b64 = []
            images_b64.insert(0, audio_b64)  # audio must come before image frames
            # Cap num_ctx at 8192 — audio embeddings cause memory overflow above this
            if request_options is None:
                request_options = {}
            request_options["num_ctx"] = min(request_options.get("num_ctx", 8192), 8192)

        if debug_print:
            print(f"""
--- ollama generate v2 request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
audio: {audio_b64 is not None}
context: {context}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
unload_after: {unload_after}
---------------------------------------------------------
""")

        response = client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images_b64,
            context=context,
            think=think,
            options=request_options,
            keep_alive= request_keep_alive,
            format=format,
        )

        if debug_print:
            print("\n--- ollama generate v2 response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response['response']
        ollama_response_thinking = response['thinking'] if think else None

        if keep_context:
            self.saved_context = response["context"]
            if debug_print:
                print("saving context to node memory.")

        # Force unload if requested
        if unload_after:
            _unload_model(client, model, debug=debug_print)

        return ollama_response_text, ollama_response_thinking, response['context'], meta,


class OllamaChat:
    """
    Text generation with Ollama Chat.
    Returns: (result: str, thinking: str|None, meta: dict, history: str)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json", "json_schema"],
                    {
                        "tooltip": "Output format. 'text' = plain text, 'json' = encourages JSON output, 'json_schema' = constrained decoding against a schema (paste a JSON Schema into the json_schema input â€” guarantees output matches the schema)."
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response.",
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Optionally set an existing model history, useful for multi-turn conversations, follow-up questions.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID.",
                    },
                ),
                "json_schema": (
                    "STRING",
                    {
                        "forceInput": False,
                        "multiline": True,
                        "default": "",
                        "tooltip": "JSON Schema string for constrained output. Only used when format is set to 'json_schema'.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
        json_schema: str = "",
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        # Update with provided values (override)
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
        else:
            meta["options"] = None

        # Final validation
        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        client = Client(host=url)

        debug_print = (
            True if meta["options"] is not None and meta["options"]["debug"] else False
        )
        
        # Check if we should unload after generation
        unload_after = meta["connectivity"].get("unload_after", False)

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None

        if format == "json_schema":
            import json as _json
            ollama_format = _json.loads(json_schema)
        elif format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        # 4. use the shared helper instead of self.get_request_options
        request_options = _filter_enabled_options(options)

        images_b64: list[str] | None = None
        if images is not None:
            images_b64 = []
            for batch_number, image in enumerate(images):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_bytes)

        if debug_print:
            print(
                f"""
--- ollama chat request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
audio: {audio_b64 is not None}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
unload_after: {unload_after}
---------------------------------------------------------
"""
            )

        # Determinate which session to use
        session_key = history if history is not None else unique_id

        # If reset_session is True, reset the session
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")

        # If the session doesn't exist, create it
        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()

        session = CHAT_SESSIONS[session_key]

        # Update history for return
        history = session_key

        # If there is a system prompt, replace it or add it to the beginning
        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        # Construct the user message for history
        user_message_for_history: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }

        # Add the user message to the history (without images)
        session.messages.append(user_message_for_history)

        if debug_print:
            print("\n--- ollama chat session:")
            for message in session.messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
                if "images" in message:
                    for image in message["images"]:
                        pprint(f"Image: {image[:50]}...")
            print("---------------------------------------------------------")

        # Construct the messages for the API call (with images)
        messages_for_api = copy.deepcopy(session.messages)

        # If there are images, modify the last user message for the API call
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        response = client.chat(
            model=model,
            messages=messages_for_api,
            options=request_options,
            keep_alive=request_keep_alive,
            format=ollama_format,
            think=think,
        )

        if debug_print:
            print("\n--- ollama chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response.message.content
        ollama_response_thinking = response.message.thinking if think else None

        # Add the assistant message to the history
        session.messages.append(
            {
                "role": "assistant",
                "content": ollama_response_text,
            }
        )

        # Force unload if requested
        if unload_after:
            _unload_model(client, model, debug=debug_print)

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            history,
        )


class OllamaRunningModels:
    """
    List models currently loaded in Ollama's VRAM (ollama ps).
    Returns a formatted string summary and a raw list of model names.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "URL of the Ollama server.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("summary", "model_names",)
    FUNCTION = "get_running_models"
    CATEGORY = "Ollama"
    DESCRIPTION = "List models currently loaded in Ollama's VRAM. Returns a human-readable summary and a comma-separated list of model names."

    def get_running_models(self, url):
        client = Client(host=url)
        try:
            result = client.ps()
            models = result.models if hasattr(result, 'models') else result.get('models', [])
        except Exception as e:
            return (f"Error: {e}", "",)

        if not models:
            return ("No models currently loaded in VRAM.", "",)

        lines = []
        names = []
        for m in models:
            name = m.model if hasattr(m, 'model') else m.get('model', '?')
            size_vram = m.size_vram if hasattr(m, 'size_vram') else m.get('size_vram', 0)
            expires_at = m.expires_at if hasattr(m, 'expires_at') else m.get('expires_at', '')
            size_gb = f"{size_vram / 1e9:.2f} GB" if size_vram else "?"
            lines.append(f"{name}  |  VRAM: {size_gb}  |  expires: {expires_at}")
            names.append(name)

        return ("\n".join(lines), ", ".join(names),)


class OllamaEmbed:
    """
    Generate embeddings for one or more texts using an Ollama embedding model.
    Returns embeddings as a JSON string (list of vectors).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "URL of the Ollama server.",
                }),
                "model": ((), {"tooltip": "Embedding model to use (e.g. nomic-embed-text, mxbai-embed-large)."}),
                "input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to embed. For multiple inputs, separate with a newline and enable split_lines.",
                }),
                "split_lines": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, splits input on newlines and embeds each line separately (batch embedding).",
                }),
                "truncate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate input to the model's context length if it exceeds it.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("embeddings_json",)
    FUNCTION = "embed"
    CATEGORY = "Ollama"
    DESCRIPTION = "Generate text embeddings using an Ollama embedding model. Returns a JSON array of embedding vectors."

    def embed(self, url, model, input, split_lines, truncate):
        import json as _json
        client = Client(host=url)
        texts = [line for line in input.splitlines() if line.strip()] if split_lines else input
        response = client.embed(model=model, input=texts, truncate=truncate)
        embeddings = response.embeddings if hasattr(response, 'embeddings') else response.get('embeddings', [])
        return (_json.dumps(embeddings),)


NODE_CLASS_MAPPINGS = {
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
    "OllamaChat": OllamaChat,
    "OllamaUnloadModel": OllamaUnloadModel,
    "OllamaRunningModels": OllamaRunningModels,
    "OllamaEmbed": OllamaEmbed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaOptionsV2": "Ollama Options",
    "OllamaConnectivityV2": "Ollama Connectivity",
    "OllamaGenerateV2": "Ollama Generate",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
    "OllamaChat": "Ollama Chat",
    "OllamaUnloadModel": "Ollama Unload Model",
    "OllamaRunningModels": "Ollama Running Models",
    "OllamaEmbed": "Ollama Embed",
}
