# scripts/build_clip_bank.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


# Allow running from repo root:
# python scripts/build_clip_bank.py ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ALLOW_EXT = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

DEFAULT_CAPTION_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"

# LLaVA prompt policy:
# We intentionally ask for exactly one complete sentence. CLIP ViT-B/32 uses a
# 77-token text context. In earlier trials, asking LLaVA for a very detailed
# caption often produced captions that were truncated or semantically unfinished
# after CLIP tokenization/generation limits. A single complete sentence gives a
# stable, compact text condition for CLIP text embedding.
DEFAULT_CAPTION_PROMPT = (
    "Describe this image for image generation in exactly one complete sentence."
)

DEFAULT_IMG_BANK_SUFFIX = "clip_img_bank"
DEFAULT_TEXT_BANK_SUFFIX = "clip_text_bank"

# Path / metadata helpers
def list_image_files(img_dir: str, recursive: bool = True) -> List[str]:
    root = Path(img_dir)

    if not root.exists():
        raise FileNotFoundError(f"img_dir does not exist: {img_dir}")

    if not root.is_dir():
        raise NotADirectoryError(f"img_dir is not a directory: {img_dir}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = [
        str(p)
        for p in iterator
        if p.is_file() and p.suffix.lower() in ALLOW_EXT
    ]
    paths.sort()

    if not paths:
        raise RuntimeError(f"No image files found under img_dir: {img_dir}")

    return paths


def _normalize_loaded_paths(arr) -> List[str]:
    out: List[str] = []
    for p in arr.tolist():
        if isinstance(p, bytes):
            out.append(p.decode("utf-8"))
        else:
            out.append(str(p))
    return out


def _bank_path(out_dir: str, domain_name: str, bank_suffix: str) -> str:
    return str(Path(out_dir) / f"{domain_name}_{bank_suffix}.npz")


def _meta_path(out_dir: str, domain_name: str, bank_suffix: str) -> str:
    return str(Path(out_dir) / f"{domain_name}_{bank_suffix}_meta.json")


def _load_extra_metadata(json_text: str) -> Dict[str, str]:
    if not json_text:
        return {}

    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("--extra_metadata_json must be a JSON object.")

    return {str(k): str(v) for k, v in parsed.items()}


def _clean_caption(text: str) -> str:
    text = str(text).strip()

    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1].strip()

    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


def _resolve_device(device: Optional[str]):
    import torch

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(dtype_name: str, device):
    import torch

    name = str(dtype_name).lower()

    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "full"}:
        return torch.float32

    raise ValueError(f"Unknown torch dtype: {dtype_name}")


def _save_meta(meta_json: str, meta: Dict[str, object]) -> None:
    Path(meta_json).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _print_sample_captions(captions: Sequence[str], n: int) -> None:
    n = max(0, int(n))
    if n <= 0:
        return

    print("\n========== [SAMPLE CAPTIONS] ==========")
    for caption in list(captions)[:n]:
        print(" -", caption)


# OpenCLIP image/text loading
def load_open_clip(
    clip_model_name: str,
    clip_pretrained: str,
    device,
):
    import open_clip

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    return clip_model, preprocess, clip_tokenizer


def encode_images_with_clip(
    paths: Sequence[str],
    clip_model,
    preprocess,
    device,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode images into normalized CLIP image embeddings."""
    import torch

    feats: List[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="CLIP image encode"):
            batch_paths = list(paths[i : i + batch_size])
            images = []

            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                try:
                    images.append(preprocess(img))
                finally:
                    img.close()

            x = torch.stack(images, dim=0).to(device)
            z = clip_model.encode_image(x)
            z = z / z.norm(dim=-1, keepdim=True)

            feats.append(z.detach().cpu().numpy().astype(np.float32))

            del x, z
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return np.concatenate(feats, axis=0).astype(np.float32)


def build_clip_image_bank(
    img_dir: str,
    out_npz: str,
    meta_json: str,
    domain_name: str,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
    image_batch_size: int = 64,
    device_name: Optional[str] = None,
    recursive: bool = True,
    force_rebuild: bool = False,
    extra_metadata: Optional[Dict[str, str]] = None,
):
    """Build an image -> CLIP image embedding bank.

    Output .npz keys:
        paths       : object array of source image paths
        embeddings  : float32 array, shape (N, 512) for ViT-B-32
    """
    paths = list_image_files(img_dir, recursive=recursive)

    if Path(out_npz).exists() and not force_rebuild:
        data = np.load(out_npz, allow_pickle=True)
        cache_paths = _normalize_loaded_paths(data["paths"])

        if cache_paths == list(paths):
            print(f"[clip-img] loaded existing cache: {out_npz}")
            return {
                "paths": cache_paths,
                "embeddings": data["embeddings"].astype(np.float32),
                "cache_path": str(out_npz),
                "meta_path": str(meta_json),
                "loaded_existing": True,
            }

        print(f"[clip-img][warn] path order changed -> rebuilding: {out_npz}")

    device = _resolve_device(device_name)

    print("\n========== [CLIP IMAGE BANK BUILD] ==========")
    print(f"[IMG DIR] {img_dir}")
    print(f"[DOMAIN] {domain_name}")
    print(f"[N IMAGES] {len(paths)}")
    print(f"[OUT NPZ] {out_npz}")
    print(f"[META JSON] {meta_json}")
    print(f"[DEVICE] {device}")
    print(f"[CLIP IMAGE] {clip_model_name} / {clip_pretrained}")

    clip_model, preprocess, _ = load_open_clip(
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        device=device,
    )

    embeddings = encode_images_with_clip(
        paths=paths,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        batch_size=int(image_batch_size),
    )

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        paths=np.array(paths, dtype=object),
        embeddings=embeddings.astype(np.float32),
        domain=np.array([str(domain_name)]),
        bank_type=np.array(["clip_image"]),
        clip_model_name=np.array([str(clip_model_name)]),
        clip_pretrained=np.array([str(clip_pretrained)]),
    )

    meta: Dict[str, object] = {
        "domain": str(domain_name),
        "img_dir": str(img_dir),
        "num_samples": int(len(paths)),
        "cache_path": str(out_npz),
        "meta_path": str(meta_json),
        "bank_type": "clip_image",
        "clip_model_name": str(clip_model_name),
        "clip_pretrained": str(clip_pretrained),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_normalization": "L2",
        "image_batch_size": int(image_batch_size),
        "recursive": bool(recursive),
        "allow_ext": sorted(ALLOW_EXT),
    }

    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    _save_meta(meta_json, meta)

    print("\n========== [SAVED] ==========")
    print(f"[IMAGE BANK] {out_npz}")
    print(f"[META] {meta_json}")
    print(f"[EMBEDDINGS] {embeddings.shape} {embeddings.dtype}")

    return {
        "paths": paths,
        "embeddings": embeddings,
        "cache_path": str(out_npz),
        "meta_path": str(meta_json),
        "loaded_existing": False,
    }


# LLaVA caption + CLIP text bank
def load_llava_model(
    caption_model_name: str,
    device,
    torch_dtype,
):
    """Load the LLaVA captioning model and processor."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(caption_model_name)

    model = LlavaForConditionalGeneration.from_pretrained(
        caption_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()

    # Batched generation is more stable with left padding for decoder-style LMs.
    processor.tokenizer.padding_side = "left"

    # Compatibility settings for newer transformers versions.
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = (
            model.config.vision_feature_select_strategy
        )
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 1

    return processor, model


def build_llava_prompt(processor, prompt_text: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]

    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )


def _open_rgb_images(paths: Sequence[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def _close_images(images: Sequence[Image.Image]) -> None:
    for img in images:
        try:
            img.close()
        except Exception:
            pass


def generate_captions_with_llava(
    paths: Sequence[str],
    processor,
    model,
    device,
    torch_dtype,
    prompt_text: str,
    batch_size: int = 8,
    max_new_tokens: int = 48,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    """Generate one-sentence image captions with LLaVA."""
    import torch

    captions: List[str] = []
    prompt = build_llava_prompt(processor, prompt_text)

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="LLaVA captioning"):
            batch_paths = list(paths[i : i + batch_size])
            images = _open_rgb_images(batch_paths)

            try:
                inputs = processor(
                    images=images,
                    text=[prompt] * len(images),
                    return_tensors="pt",
                    padding=True,
                )

                moved = {}
                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        if torch.is_floating_point(value):
                            moved[key] = value.to(device, dtype=torch_dtype)
                        else:
                            moved[key] = value.to(device)
                    else:
                        moved[key] = value
                inputs = moved

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                )

                # Keep only newly generated tokens, not the prompt tokens.
                gen_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                batch_captions = processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                )
                batch_captions = [_clean_caption(c) for c in batch_captions]
                captions.extend(batch_captions)

                del inputs, output_ids, gen_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            finally:
                _close_images(images)

    return captions


def encode_texts_with_clip(
    texts: Sequence[str],
    clip_model,
    clip_tokenizer,
    device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode captions into normalized CLIP text embeddings."""
    import torch

    feats: List[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode"):
            batch_texts = list(texts[i : i + batch_size])
            tokens = clip_tokenizer(batch_texts).to(device)

            z = clip_model.encode_text(tokens)
            z = z / z.norm(dim=-1, keepdim=True)

            feats.append(z.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)


def build_caption_clip_text_bank(
    img_dir: str,
    out_npz: str,
    meta_json: str,
    domain_name: str,
    caption_model_name: str = DEFAULT_CAPTION_MODEL,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
    caption_prompt: str = DEFAULT_CAPTION_PROMPT,
    caption_batch_size: int = 8,
    text_batch_size: int = 256,
    max_new_tokens: int = 48,
    torch_dtype_name: str = "auto",
    device_name: Optional[str] = None,
    recursive: bool = True,
    force_rebuild: bool = False,
    extra_metadata: Optional[Dict[str, str]] = None,
):
    """Build an image -> LLaVA caption -> CLIP text embedding bank.

    Output .npz keys:
        paths       : object array of source image paths
        captions    : object array of one-sentence captions
        embeddings  : float32 array, shape (N, 512) for ViT-B-32
    """
    import torch

    paths = list_image_files(img_dir, recursive=recursive)

    if Path(out_npz).exists() and not force_rebuild:
        data = np.load(out_npz, allow_pickle=True)
        cache_paths = _normalize_loaded_paths(data["paths"])

        if cache_paths == list(paths):
            print(f"[clip-text] loaded existing cache: {out_npz}")
            return {
                "paths": cache_paths,
                "captions": data["captions"].tolist(),
                "embeddings": data["embeddings"].astype(np.float32),
                "cache_path": str(out_npz),
                "meta_path": str(meta_json),
                "loaded_existing": True,
            }

        print(f"[clip-text][warn] path order changed -> rebuilding: {out_npz}")

    device = _resolve_device(device_name)
    torch_dtype = _resolve_dtype(torch_dtype_name, device)

    print("\n========== [LLaVA CAPTION + CLIP TEXT BANK BUILD] ==========")
    print(f"[IMG DIR] {img_dir}")
    print(f"[DOMAIN] {domain_name}")
    print(f"[N IMAGES] {len(paths)}")
    print(f"[OUT NPZ] {out_npz}")
    print(f"[META JSON] {meta_json}")
    print(f"[DEVICE] {device}")
    print(f"[DTYPE] {torch_dtype}")
    print(f"[CAPTION MODEL] {caption_model_name}")
    print(f"[CLIP TEXT] {clip_model_name} / {clip_pretrained}")
    print(f"[PROMPT] {caption_prompt}")

    processor, llava_model = load_llava_model(
        caption_model_name=caption_model_name,
        device=device,
        torch_dtype=torch_dtype,
    )

    captions = generate_captions_with_llava(
        paths=paths,
        processor=processor,
        model=llava_model,
        device=device,
        torch_dtype=torch_dtype,
        prompt_text=caption_prompt,
        batch_size=int(caption_batch_size),
        max_new_tokens=int(max_new_tokens),
    )

    # Release the 7B captioning model before loading CLIP where possible.
    del llava_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    clip_model, _, clip_tokenizer = load_open_clip(
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        device=device,
    )

    embeddings = encode_texts_with_clip(
        texts=captions,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        device=device,
        batch_size=int(text_batch_size),
    )

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        paths=np.array(paths, dtype=object),
        captions=np.array(captions, dtype=object),
        embeddings=embeddings.astype(np.float32),
        domain=np.array([str(domain_name)]),
        bank_type=np.array(["caption_clip_text"]),
        caption_model=np.array([str(caption_model_name)]),
        clip_model_name=np.array([str(clip_model_name)]),
        clip_pretrained=np.array([str(clip_pretrained)]),
        caption_prompt=np.array([str(caption_prompt)]),
    )

    meta: Dict[str, object] = {
        "domain": str(domain_name),
        "img_dir": str(img_dir),
        "num_samples": int(len(paths)),
        "cache_path": str(out_npz),
        "meta_path": str(meta_json),
        "bank_type": "caption_clip_text",
        "caption_model": str(caption_model_name),
        "caption_prompt": str(caption_prompt),
        "caption_policy": "exactly_one_complete_sentence",
        "caption_policy_reason": (
            "CLIP ViT-B/32 uses a 77-token context. One complete sentence keeps "
            "the LLaVA caption compact and avoids incomplete long captions being "
            "used as text conditions."
        ),
        "max_new_tokens": int(max_new_tokens),
        "text_encoder_family": "CLIP",
        "clip_model_name": str(clip_model_name),
        "clip_pretrained": str(clip_pretrained),
        "clip_context_length": 77,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_normalization": "L2",
        "caption_batch_size": int(caption_batch_size),
        "text_batch_size": int(text_batch_size),
        "recursive": bool(recursive),
        "allow_ext": sorted(ALLOW_EXT),
    }

    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    _save_meta(meta_json, meta)

    print("\n========== [SAVED] ==========")
    print(f"[TEXT BANK] {out_npz}")
    print(f"[META] {meta_json}")
    print(f"[EMBEDDINGS] {embeddings.shape} {embeddings.dtype}")

    return {
        "paths": paths,
        "captions": captions,
        "embeddings": embeddings,
        "cache_path": str(out_npz),
        "meta_path": str(meta_json),
        "loaded_existing": False,
    }


# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build CLIP image banks and/or LLaVA-caption + CLIP text banks "
            "from a prepared image directory."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["img", "text", "all"],
        required=True,
        help=(
            "img: image -> CLIP image embedding. "
            "text: image -> LLaVA one-sentence caption -> CLIP text embedding. "
            "all: build both banks."
        ),
    )
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Prepared image directory. All images under this folder are used.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for .npz banks and metadata JSON files.",
    )
    parser.add_argument(
        "--domain_name",
        required=True,
        help="Domain name used for output filenames and metadata.",
    )
    parser.add_argument(
        "--img_bank_suffix",
        default=DEFAULT_IMG_BANK_SUFFIX,
        help=(
            "Image-bank filename suffix. Default preserves the old convention: "
            "<domain>_clip_bank.npz."
        ),
    )
    parser.add_argument(
        "--text_bank_suffix",
        default=DEFAULT_TEXT_BANK_SUFFIX,
        help="Text-bank filename suffix.",
    )
    parser.add_argument(
        "--out_img_npz",
        default=None,
        help="Optional explicit image-bank .npz path.",
    )
    parser.add_argument(
        "--out_text_npz",
        default=None,
        help="Optional explicit caption-text-bank .npz path.",
    )
    parser.add_argument(
        "--img_meta_json",
        default=None,
        help="Optional explicit image-bank metadata .json path.",
    )
    parser.add_argument(
        "--text_meta_json",
        default=None,
        help="Optional explicit caption-text-bank metadata .json path.",
    )
    parser.add_argument(
        "--caption_model",
        default=DEFAULT_CAPTION_MODEL,
        help="Hugging Face LLaVA caption model name.",
    )
    parser.add_argument(
        "--clip_model",
        default=DEFAULT_CLIP_MODEL,
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--clip_pretrained",
        default=DEFAULT_CLIP_PRETRAINED,
        help="OpenCLIP pretrained tag.",
    )
    parser.add_argument(
        "--caption_prompt",
        default=DEFAULT_CAPTION_PROMPT,
        help=(
            "Prompt used for image captioning. Default requests exactly one "
            "complete sentence to remain compatible with CLIP's 77-token context."
        ),
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=64,
        help="Batch size for CLIP image encoding.",
    )
    parser.add_argument(
        "--caption_batch_size",
        type=int,
        default=8,
        help="Batch size for LLaVA caption generation. A100 used batch_size=8.",
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=256,
        help="Batch size for CLIP text encoding.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Maximum generated caption tokens from LLaVA.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
        help="Torch dtype for LLaVA loading. Default: fp16 on CUDA, fp32 on CPU.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively list images. Enabled by default.",
    )
    parser.add_argument(
        "--non_recursive",
        action="store_false",
        dest="recursive",
        help="Only list images directly under img_dir.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Rebuild even when an existing cache with matching paths exists.",
    )
    parser.add_argument(
        "--extra_metadata_json",
        default="",
        help='Optional JSON object of extra string metadata, e.g. \'{"split":"train"}\'.',
    )
    parser.add_argument(
        "--print_samples",
        type=int,
        default=5,
        help="Print the first N captions after text/all mode.",
    )

    return parser


def _resolve_outputs(args: argparse.Namespace) -> Dict[str, str]:
    out_img_npz = args.out_img_npz or _bank_path(
        args.out_dir,
        args.domain_name,
        args.img_bank_suffix,
    )
    out_text_npz = args.out_text_npz or _bank_path(
        args.out_dir,
        args.domain_name,
        args.text_bank_suffix,
    )

    img_meta_json = args.img_meta_json or _meta_path(
        args.out_dir,
        args.domain_name,
        args.img_bank_suffix,
    )
    text_meta_json = args.text_meta_json or _meta_path(
        args.out_dir,
        args.domain_name,
        args.text_bank_suffix,
    )

    return {
        "out_img_npz": out_img_npz,
        "out_text_npz": out_text_npz,
        "img_meta_json": img_meta_json,
        "text_meta_json": text_meta_json,
    }


def main() -> None:
    args = build_parser().parse_args()
    outputs = _resolve_outputs(args)
    extra_metadata = _load_extra_metadata(args.extra_metadata_json)

    img_bank = None
    text_bank = None

    if args.mode in {"img", "all"}:
        img_bank = build_clip_image_bank(
            img_dir=args.img_dir,
            out_npz=outputs["out_img_npz"],
            meta_json=outputs["img_meta_json"],
            domain_name=args.domain_name,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            image_batch_size=args.image_batch_size,
            device_name=args.device,
            recursive=bool(args.recursive),
            force_rebuild=bool(args.force_rebuild),
            extra_metadata=extra_metadata,
        )

    if args.mode in {"text", "all"}:
        text_bank = build_caption_clip_text_bank(
            img_dir=args.img_dir,
            out_npz=outputs["out_text_npz"],
            meta_json=outputs["text_meta_json"],
            domain_name=args.domain_name,
            caption_model_name=args.caption_model,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            caption_prompt=args.caption_prompt,
            caption_batch_size=args.caption_batch_size,
            text_batch_size=args.text_batch_size,
            max_new_tokens=args.max_new_tokens,
            torch_dtype_name=args.torch_dtype,
            device_name=args.device,
            recursive=bool(args.recursive),
            force_rebuild=bool(args.force_rebuild),
            extra_metadata=extra_metadata,
        )

        n_print = max(0, int(args.print_samples))
        if n_print > 0:
            _print_sample_captions(text_bank["captions"], n_print)

    print("\n========== [DONE] ==========")
    if img_bank is not None:
        print(f"[IMG BANK] {img_bank['cache_path']}")
    if text_bank is not None:
        print(f"[TEXT BANK] {text_bank['cache_path']}")


if __name__ == "__main__":
    main()
