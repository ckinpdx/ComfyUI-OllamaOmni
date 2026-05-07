# ComfyUI-OllamaOmni

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for interacting with [Ollama](https://ollama.com/) — including text, vision, audio, and video support.

Forked from [comfyui-ollama](https://github.com/stavsap/comfyui-ollama) by stavsap and extended with omni-modal capabilities.

---

## Requirements

A running Ollama server reachable from the host running ComfyUI. Install from [ollama.com](https://ollama.com/).

---

## Installation

```shell
cd ComfyUI/custom_nodes
git clone https://github.com/ckinpdx/ComfyUI-OllamaOmni
pip install -r ComfyUI-OllamaOmni/requirements.txt
```

Restart ComfyUI.

---

## Nodes

### Ollama Connectivity
Configures the connection to an Ollama server. Provides URL, model selection, keep-alive duration, and optional auto-unload after generation. Connect to any Generate or Chat node.

### Ollama Options
Full control over Ollama inference parameters (temperature, top_k, top_p, seed, num_ctx, etc.). Each parameter has an enable/disable toggle — only enabled options are sent to the API. Includes a `debug` flag for CLI output.

### Ollama Generate
Text generation with support for vision, audio, and video frame inputs.

| Input | Type | Notes |
|---|---|---|
| system | STRING | System prompt |
| prompt | STRING | User prompt |
| images | IMAGE | Images or video frame batch |
| audio | AUDIO | Audio input — see Audio section below |
| max_frames | INT | Socket input. Limits frames sampled from a video batch (0 = all) |
| connectivity | OLLAMA_CONNECTIVITY | Server + model config |
| options | OLLAMA_OPTIONS | Inference parameters |
| context | OLLAMA_CONTEXT | Previous context for multi-turn use |
| meta | OLLAMA_META | Chain from another Generate node |
| format | text / json / json_schema | Output format |
| think | BOOLEAN | Enable chain-of-thought (model must support it) |

Outputs: `result`, `thinking`, `context`, `meta`

### Ollama Chat
Multi-turn conversational node using `ollama.chat()`. Maintains full conversation history per session. Supports vision and audio inputs, chaining via `history` output, and session reset.

### Ollama Audio Transcribe
Transcribe audio to text using any Ollama-compatible speech model. Converts audio to 16kHz mono WAV and passes it via the `images[]` field (see Audio section).

### Ollama Audio Chat
Send audio alongside a text prompt to an audio-capable multimodal model. Uses Ollama Connectivity for server/model selection.

### Ollama Video Analyze
Sample frames from a video IMAGE batch and send them to a vision model. Configurable frame count and sampling strategy (uniform / first / last).

### Ollama Embed
Generate text embeddings using an Ollama embedding model (e.g. `nomic-embed-text`). Returns embeddings as a JSON array.

### Ollama Running Models
Lists models currently loaded in Ollama's VRAM (`ollama ps`). Returns a summary string and comma-separated model names.

### Ollama Unload Model
Force-unload a model from VRAM immediately. Passthrough node — connect `meta` from a Generate/Chat node and pipe result text through it.

### Ollama Save / Load Context
Save and load model context to/from PNG files (context is embedded as PNG metadata). Useful for persisting multi-turn conversation state across workflows.

---

## Audio

Ollama does not yet have a native audio API. Audio works via a workaround: Ollama detects RIFF/WAVE magic bytes in the `images[]` field and routes the data to the model's audio encoder.

**Requirements for audio to work:**
- WAV format with a full RIFF header (raw PCM fails silently)
- 16kHz mono (the nodes handle resampling automatically)
- `num_ctx` capped at 8192 (the nodes enforce this automatically when audio is connected)
- Audio must be placed before any image frames in the images list (handled automatically)

Connect any ComfyUI `AUDIO` output to the `audio` input on Ollama Generate, Ollama Audio Transcribe, or Ollama Audio Chat.

---

## Cloud / Authenticated Models

For Ollama cloud models requiring authentication:

```shell
ollama signin
```

Or add your public key at [ollama.com/settings/keys](https://ollama.com/settings/keys).

| OS | Key path |
|---|---|
| Windows | `C:\Users\<username>\.ollama\id_ed25519.pub` |
| macOS | `~/.ollama/id_ed25519.pub` |
| Linux | `/usr/share/ollama/.ollama/id_ed25519.pub` |

---

## Attribution

Original project: [comfyui-ollama](https://github.com/stavsap/comfyui-ollama) by Stav Sapir, licensed under Apache 2.0.