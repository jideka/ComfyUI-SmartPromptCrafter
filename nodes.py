import json
import urllib.request
import urllib.error
import re

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Known class name → model family mapping
# Based on real ComfyUI MODEL object inspection
# ---------------------------------------------------------------------------

CLASS_FAMILY_MAP = {
    # SD 1.5
    "BaseModel":          "SD 1.5",
    "SDXLRefiner":        "SDXL Refiner",
    # SDXL
    "SDXL":               "SDXL",
    # Flux
    "Flux":               "Flux",
    "FluxInpaint":        "Flux",
    # SD3
    "SD3":                "SD3",
    # Wan
    "WAN21":              "Wan 2.1",
    "WAN22":              "Wan 2.2",
    "WAN22_S2V":          "Wan 2.2 (Image-to-Video)",
    "WAN22_T2V":          "Wan 2.2 (Text-to-Video)",
    # Hunyuan
    "HunyuanVideo":       "Hunyuan Video",
    "HunyuanDiT":         "Hunyuan DiT",
    # Mochi
    "Mochi":              "Mochi",
    # CogVideo
    "CogVideoX":          "CogVideoX",
    # LTX
    "LTXVModel":          "LTX Video",
    # Aura
    "AuraFlow":           "AuraFlow",
    # Stable Cascade
    "StableCascade_C":    "Stable Cascade (Stage C)",
    "StableCascade_B":    "Stable Cascade (Stage B)",
}

LATENT_FAMILY_MAP = {
    "SD_0_1":   "SD 1.5",
    "SDXL":     "SDXL",
    "Flux":     "Flux",
    "SD3":      "SD3",
    "Wan21":    "Wan",
    "Mochi":    "Mochi",
    "LTXVideo": "LTX Video",
    "CogVideoX":"CogVideoX",
}

# ---------------------------------------------------------------------------
# Helper - detect model architecture from ComfyUI MODEL object
# ---------------------------------------------------------------------------

def _detect_model_info(model) -> str:
    """Extract rich fingerprint from ComfyUI MODEL object for LLM identification."""
    info = {}

    try:
        inner = getattr(model, "model", None)
        if inner is None:
            return "unknown"

        # 1. model_type enum
        model_type = getattr(inner, "model_type", None)
        if model_type is not None:
            info["model_type"] = str(model_type)

        # 2. Inner class name - most precise identifier
        cls_name = type(inner).__name__
        info["class"] = cls_name

        # 3. Known family lookup from class name
        family = CLASS_FAMILY_MAP.get(cls_name)
        if family:
            info["known_family"] = family

        # 4. config / model_config class
        for attr in ("model_config", "config"):
            cfg = getattr(inner, attr, None)
            if cfg is not None:
                info["config_class"] = type(cfg).__name__
                break

        # 5. Latent format
        latent_format = getattr(inner, "latent_format", None)
        if latent_format is not None:
            latent_name = type(latent_format).__name__
            info["latent_format"] = latent_name
            latent_family = LATENT_FAMILY_MAP.get(latent_name)
            if latent_family and "known_family" not in info:
                info["known_family"] = latent_family

        # 6. adm_channels (0 = Flux/Wan/modern, 2816 = SDXL, 256 = SD1.5)
        for attr in ("adm_channels", "context_dim"):
            val = None
            for obj in [inner, getattr(inner, "model_config", None)]:
                if obj is not None:
                    val = getattr(obj, attr, None)
                    if val is not None:
                        break
            if val is not None:
                info[attr] = val

        # 7. diffusion_model class
        diffusion = getattr(inner, "diffusion_model", None)
        if diffusion is not None:
            info["diffusion_model"] = type(diffusion).__name__

    except Exception as e:
        info["detection_error"] = str(e)

    parts = [f"{k}={v}" for k, v in info.items()]
    return ", ".join(parts) if parts else "unknown"


# ---------------------------------------------------------------------------
# Helper - call Groq with plain urllib (no extra dependencies)
# ---------------------------------------------------------------------------

def _call_groq(api_key: str, system_prompt: str, user_prompt: str) -> str:
    payload = json.dumps({
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Content-Type":    "application/json",
            "Authorization":   f"Bearer {api_key}",
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept":          "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling Groq: {e.reason}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in Stable Diffusion and generative AI prompt engineering.
You receive technical metadata about a loaded ComfyUI model and deduce its exact family.

The metadata includes:
- class: the Python class name of the model in ComfyUI (most reliable)
- known_family: pre-identified family if available (trust this)
- model_type: ComfyUI enum (EPS=SD1.5, V_PREDICTION=SD1.5v, FLOW=Flux/Wan/modern)
- latent_format: latent space type
- diffusion_model: underlying architecture class
- adm_channels: 0=modern DiT, 2816=SDXL, 256=SD1.5

Known class-to-family mappings:
- BaseModel -> SD 1.5
- SDXL -> SDXL
- Flux, FluxInpaint -> Flux
- SD3 -> SD3
- WAN21, WAN22, WAN22_S2V, WAN22_T2V -> Wan video model
- HunyuanVideo -> Hunyuan Video
- HunyuanDiT -> Hunyuan DiT (image)
- Mochi -> Mochi video
- CogVideoX -> CogVideo
- LTXVModel -> LTX Video
- AuraFlow -> AuraFlow
- StableCascade_C/B -> Stable Cascade

For video models (Wan, Hunyuan Video, Mochi, CogVideoX, LTX):
- Positive prompt should describe a scene with motion, camera movement, cinematic style
- Negative prompt should be minimal or empty

Output ONLY a JSON object with exactly these keys:
{
  "model_label": "<human readable family name>",
  "positive": "<optimized positive prompt>",
  "negative": "<optimized negative prompt>"
}

Prompt style rules:
- SD 1.5: comma-separated short tokens, quality boosters like (masterpiece:1.2), best quality
- SDXL: natural language sentences, detailed descriptions
- Pony / Illustrious: score_9, score_8_up, rating:general or rating:explicit
- Flux: rich descriptive natural language, no quality tokens needed, minimal negative
- SD3: natural language, moderate length, minimal negative
- Wan / video models: cinematic scene description with motion, camera moves, lighting
- HunyuanDiT: natural language, Chinese aesthetic often works well
- Unknown/new models: use natural language as default

Do NOT wrap the JSON in markdown code fences.
"""


# ---------------------------------------------------------------------------
# In-memory cache: model_info -> (model_label, positive, negative)
# Avoids repeated Groq API calls during batch runs with the same model
# Cache lives for the duration of the ComfyUI session
# ---------------------------------------------------------------------------

_GROQ_CACHE: dict = {}


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class SmartPromptCrafter:
    CATEGORY = "prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rough_idea": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a girl in a forest, sunset, fantasy",
                        "tooltip": "Describe roughly what you want. The AI will adapt it to the model.",
                    },
                ),
                "extra_negative": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Extra negative keywords always appended (e.g. 'watermark, logo').",
                    },
                ),
                "groq_api_key": (
                    "STRING",
                    {
                        "default": "gsk_...",
                        "tooltip": "Your free Groq API key from console.groq.com",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("MODEL", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("model", "positive", "negative", "model_label")
    FUNCTION      = "craft_prompts"
    OUTPUT_NODE   = False

    def craft_prompts(
        self,
        model,
        rough_idea: str,
        extra_negative: str,
        groq_api_key: str,
    ):
        if not groq_api_key or groq_api_key.startswith("gsk_..."):
            raise ValueError(
                "SmartPromptCrafter: please enter a valid Groq API key. "
                "Get one for free at https://console.groq.com"
            )

        # Detect architecture from the MODEL object
        model_info = _detect_model_info(model)
        print(f"[SmartPromptCrafter] Detected model info: {model_info}")

        # Cache key: model fingerprint + rough idea (idea changes the prompt)
        cache_key = f"{model_info}||{rough_idea}"

        if cache_key in _GROQ_CACHE:
            print(f"[SmartPromptCrafter] Cache hit - skipping Groq API call")
            model_label, positive, negative = _GROQ_CACHE[cache_key]
        else:
            user_prompt = (
                f"Model technical info: {model_info}\n"
                f"Rough idea: {rough_idea}"
            )

            raw = _call_groq(groq_api_key, SYSTEM_PROMPT, user_prompt)

            # Parse JSON, tolerant of minor formatting issues
            try:
                clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
                result = json.loads(clean)
                positive    = result.get("positive",    "").strip()
                negative    = result.get("negative",    "").strip()
                model_label = result.get("model_label", model_info).strip()
            except json.JSONDecodeError:
                positive    = raw
                negative    = ""
                model_label = model_info

            _GROQ_CACHE[cache_key] = (model_label, positive, negative)
            print(f"[SmartPromptCrafter] Model identified as: {model_label} (cached)")

        # Append extra_negative
        if extra_negative.strip():
            separator = ", " if negative.strip() else ""
            negative = negative + separator + extra_negative.strip()

        return (model, positive, negative, model_label)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SmartPromptCrafter": SmartPromptCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartPromptCrafter": "Smart Prompt Crafter ✨",
}
