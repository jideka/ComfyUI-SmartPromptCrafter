import json
import urllib.request
import urllib.error
import re

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Helper – detect model architecture from ComfyUI MODEL object
# ---------------------------------------------------------------------------

def _detect_model_info(model) -> str:
    """Extract as much info as possible from a ComfyUI MODEL object."""
    info_parts = []

    try:
        # ModelPatcher wraps the actual model
        inner = getattr(model, "model", None)

        # 1. model_type enum (most reliable)
        model_type = getattr(inner, "model_type", None)
        if model_type is not None:
            info_parts.append(f"model_type={model_type}")

        # 2. Class name of the inner model
        if inner is not None:
            cls_name = type(inner).__name__
            info_parts.append(f"class={cls_name}")

        # 3. Try to get config / model_config
        for attr in ("model_config", "config"):
            cfg = getattr(inner, attr, None)
            if cfg is not None:
                info_parts.append(f"config_class={type(cfg).__name__}")
                break

        # 4. Latent format (gives SD1.5 vs SDXL hint via channel count)
        latent_format = getattr(inner, "latent_format", None)
        if latent_format is not None:
            info_parts.append(f"latent_format={type(latent_format).__name__}")

        # 5. adm_channels / context_dim hints
        for attr in ("adm_channels", "context_dim"):
            val = None
            for obj in [inner, getattr(inner, "model_config", None)]:
                if obj is not None:
                    val = getattr(obj, attr, None)
                    if val is not None:
                        break
            if val is not None:
                info_parts.append(f"{attr}={val}")

        # 6. Try diffusion_model class name
        diffusion = getattr(inner, "diffusion_model", None)
        if diffusion is not None:
            info_parts.append(f"diffusion_model={type(diffusion).__name__}")

    except Exception as e:
        info_parts.append(f"detection_error={e}")

    return ", ".join(info_parts) if info_parts else "unknown"


# ---------------------------------------------------------------------------
# Helper – call Groq with plain urllib (no extra dependencies)
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
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept":        "application/json",
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

SYSTEM_PROMPT = """You are an expert in Stable Diffusion prompt engineering.
You receive technical information about a loaded model (class names, model_type
enum, latent format, diffusion architecture, etc.) and you deduce which model
family it belongs to (SD 1.5, SDXL, Pony, Illustrious, Flux, SD3, etc.).

Given that technical info and a rough description of what the user wants,
output ONLY a JSON object — no preamble, no explanation — with exactly
these keys:

{
  "model_label": "<human readable model family, e.g. SDXL, SD 1.5, Flux, Pony>",
  "positive": "<optimized positive prompt for that model>",
  "negative": "<optimized negative prompt for that model>"
}

Rules:
- Deduce the model family from the technical metadata provided.
- Match the token style that works best for the model:
    * Pony / Illustrious  → score_9, score_8_up, rating tags
    * SDXL                → natural language, detailed sentences
    * SD 1.5              → comma-separated short tokens
    * Flux                → natural language, very descriptive
    * SD3                 → natural language, moderate length
- Keep prompts in English.
- Positive: expand the user's rough idea with style, lighting, composition,
  and quality boosters appropriate for the model.
- Negative: include artefacts, anatomy issues, and style-specific negatives
  known to help that model. (Flux/SD3 rarely need negatives — keep short.)
- Do NOT wrap the JSON in markdown code fences.
"""


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

        # Append extra_negative
        if extra_negative.strip():
            separator = ", " if negative.strip() else ""
            negative = negative + separator + extra_negative.strip()

        # Pass MODEL through so it can be chained to KSampler etc.
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
