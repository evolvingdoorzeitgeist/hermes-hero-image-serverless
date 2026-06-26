import os
import base64
import io
import uuid

import runpod
import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

_pipe = None

ASPECT_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "21:9": (1536, 640),
}


def get_pipe():
    global _pipe

    if _pipe is None:
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            raise ValueError("HF_TOKEN is missing. Add it to RunPod endpoint environment variables.")

        _pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=hf_token,
        )
        _pipe.to("cuda")

    return _pipe


def health():
    return {
        "status": "ok",
        "worker": "hermes-hero-image-serverless",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "supported_aspect_ratios": list(ASPECT_SIZES.keys()),
    }


def generate(job_input):
    prompt = (job_input.get("prompt") or "").strip()
    aspect_ratio = job_input.get("aspect_ratio") or "16:9"
    filename = job_input.get("filename") or f"hero_{uuid.uuid4().hex}.png"

    if not prompt:
        raise ValueError("Prompt is required")

    if aspect_ratio not in ASPECT_SIZES:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    width, height = ASPECT_SIZES[aspect_ratio]

    # For FLUX schnell, keep steps low.
    num_inference_steps = int(job_input.get("steps") or 4)
    guidance_scale = float(job_input.get("guidance_scale") or 0.0)

    full_prompt = (
        prompt.strip()
        + ", no text, no words, no letters, no logo, no watermark, clean composition"
    )

    pipe = get_pipe()

    image = pipe(
        prompt=full_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "status": "ok",
        "job_id": str(uuid.uuid4()),
        "engine": "flux-schnell",
        "model": MODEL_ID,
        "filename": filename,
        "mime_type": "image/png",
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "size_bytes": len(image_bytes),
        "image_base64": image_b64,
    }


def handler(job):
    job_input = job.get("input", {}) or {}
    action = job_input.get("action")

    try:
        if action == "health":
            return health()

        if action == "generate":
            return generate(job_input)

        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "supported_actions": ["health", "generate"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "action": action,
        }


runpod.serverless.start({"handler": handler})
