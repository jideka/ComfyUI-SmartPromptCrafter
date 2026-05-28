# ComfyUI-SmartPromptCrafter ✨

**Automatically generates optimized positive & negative prompts tailored to your loaded checkpoint — powered by Groq (free API).**

No more guessing whether to use `score_9` tags for Pony, natural language for SDXL, or comma tokens for SD 1.5. Just describe your rough idea, connect your model, and get production-ready prompts instantly.

![Node Preview](assets/node_preview.png)

---

## Features

- 🔍 **Auto-detects model architecture** — reads directly from the ComfyUI `MODEL` object (SDXL, SD 1.5, Flux, Pony, Illustrious, SD3, and more)
- ✍️ **Rewrites your rough idea** into a fully optimized positive prompt matched to the model's token style
- 🚫 **Generates model-specific negatives** — knows which negative tags actually help each architecture
- ➕ **Extra negative input** — append your own permanent negative keywords on top
- 🔌 **Zero dependencies** — pure Python standard library (`urllib`, `json`, `re`), works with any ComfyUI install
- 💸 **Free** — uses Groq's free tier with `llama-3.3-70b-versatile`

---

## Workflow

```
[Load Checkpoint]
  └── MODEL ──► [Smart Prompt Crafter ✨] ──► MODEL ──► [KSampler]
                        │
            positive ───┼──► [CLIP Text Encode +] ──► [KSampler]
            negative ───┼──► [CLIP Text Encode -] ──► [KSampler]
          model_label ──┘──► [Show Text] (debug)
```

![Workflow Example](assets/workflow_example.png)

---

## Installation

### Via ComfyUI Manager (recommended)
Search for **SmartPromptCrafter** in the Manager and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jideka/ComfyUI-SmartPromptCrafter
```
Restart ComfyUI. No pip install required.

---

## Setup — Get a free Groq API key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`) and paste it into the `groq_api_key` widget

---

## Inputs

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Connect directly to **Load Checkpoint** MODEL output |
| `rough_idea` | STRING | Your rough description — the AI expands and adapts it |
| `extra_negative` | STRING | Extra negatives always appended (e.g. `watermark, logo`) |
| `groq_api_key` | STRING | Your free Groq API key |

## Outputs

| Output | Type | Description |
|---|---|---|
| `model` | MODEL | Pass-through — connect to KSampler |
| `positive` | STRING | Optimized positive prompt — connect to CLIP Text Encode |
| `negative` | STRING | Optimized negative prompt — connect to CLIP Text Encode |
| `model_label` | STRING | Detected model family (e.g. `SDXL`, `SD 1.5`, `Flux`) |

---

## Supported Architectures

| Model Family | Detection | Prompt Style |
|---|---|---|
| SD 1.5 / DreamShaper / Realistic Vision | ✅ | Comma-separated tokens |
| SDXL / DreamShaper XL | ✅ | Natural language |
| Pony Diffusion | ✅ | `score_9`, `score_8_up`, rating tags |
| Illustrious | ✅ | `score_9`, anime tags |
| Flux | ✅ | Descriptive natural language, minimal negative |
| SD3 | ✅ | Natural language, moderate length |

---

## How it works

The node reads the internal ComfyUI `MODEL` object at runtime — no filename needed. It extracts:
- `model_type` enum (EPS, V_PREDICTION, FLUX, etc.)
- Inner model class name (SDXL, Flux, etc.)
- Latent format (SD vs SDXL channel layout)
- `adm_channels` and `context_dim` (architecture fingerprints)
- `diffusion_model` class

This technical fingerprint is sent to Groq alongside your rough idea. The LLM identifies the exact model family and generates prompts in the correct style.

---

## Example

**Rough idea:** `a girl in a forest, sunset, fantasy`
**Loaded model:** `dreamshaperXL_v21TurboDPMSDE.safetensors` (detected as SDXL)

**Generated positive:**
> A young woman standing in an ancient mystical forest, golden sunset light filtering through towering trees, fantasy atmosphere, ethereal glow, detailed foliage, cinematic composition, ultra-detailed, sharp focus, dramatic lighting

**Generated negative:**
> deformed, distorted anatomy, extra limbs, blurry, low quality, bad proportions, oversaturated, flat lighting, watermark, signature

---

## License

MIT — see [LICENSE](LICENSE)

---

## Credits

Built with [Groq](https://groq.com) — `llama-3.3-70b-versatile`  
Inspired by the ComfyUI community ❤️

Buy me a coffe : https://ko-fi.com/jideka
