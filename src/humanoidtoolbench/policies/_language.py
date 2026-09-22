"""Pinned frozen CLIP text features shared by ACT/DP training and serving."""

from __future__ import annotations

MODEL_ID = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
PROJECTION_DIM = 768
MAX_LENGTH = 77
CACHE_FILES = (
    "config.json",
    "model.safetensors",
    "merges.txt",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def language_receipt(enabled: bool) -> dict:
    return {
        "enabled": enabled,
        "encoder": "clip",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "projection_dim": PROJECTION_DIM,
        "tokenizer_revision": REVISION,
        "max_length": MAX_LENGTH,
        "preprocessing": "verbatim instruction, native CLIP tokenizer",
        "encoder_dtype": "float32",
        "encoder_device": "cpu",
        "architecture": "zero_init_feature_residual_v1",
    }


def config_receipt(config) -> dict:
    get = (
        config.get
        if isinstance(config, dict)
        else lambda key, default=None: getattr(config, key, default)
    )
    enabled = get("language_conditioning", False)
    if not isinstance(enabled, bool):
        raise ValueError("language_conditioning must be boolean")
    if enabled:
        expected = {
            "language_model_id": MODEL_ID,
            "language_revision": REVISION,
            "language_projection_dim": PROJECTION_DIM,
        }
        for key, value in expected.items():
            if get(key) != value:
                raise ValueError(f"Unsupported ACT/DP language configuration: {key}")
    return language_receipt(enabled)


def make_encoder(config):
    return FrozenCLIPInstructionCache() if config_receipt(config)["enabled"] else None


class FrozenCLIPInstructionCache:
    """Encode each exact instruction once, outside the trainable policy and DDP.

    The encoder stays on CPU in float32 in training and serving. Loading only
    local pinned files keeps model downloads out of data workers and requests.
    """

    def __init__(self):
        self.encoder = None
        self.tokenizer = None
        self.cache = {}

    def _load(self):
        import torch
        from huggingface_hub.constants import HF_HUB_CACHE
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True, cache_dir=HF_HUB_CACHE
        )
        self.encoder = (
            CLIPTextModelWithProjection.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                local_files_only=True,
                cache_dir=HF_HUB_CACHE,
                torch_dtype=torch.float32,
            )
            .eval()
            .requires_grad_(False)
        )
        if self.encoder.config.projection_dim != PROJECTION_DIM:
            raise ValueError("Pinned CLIP projection dimension does not match")
        if self.encoder.config.max_position_embeddings != MAX_LENGTH:
            raise ValueError("Pinned CLIP text context length does not match")

    def encode(self, instructions, *, device):
        import torch

        if not isinstance(instructions, (list, tuple)) or not instructions:
            raise ValueError(
                "Language-conditioned ACT/DP requires a list of instructions"
            )
        if any(not isinstance(text, str) or not text.strip() for text in instructions):
            raise ValueError(
                "Language-conditioned ACT/DP requires nonempty instructions"
            )
        missing = list(
            dict.fromkeys(text for text in instructions if text not in self.cache)
        )
        if missing:
            if self.encoder is None:
                self._load()
            for start in range(0, len(missing), 128):
                texts = missing[start : start + 128]
                tokens = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                with torch.no_grad():
                    embeddings = self.encoder(**tokens).text_embeds.float()
                if (
                    embeddings.shape != (len(texts), PROJECTION_DIM)
                    or not torch.isfinite(embeddings).all()
                ):
                    raise ValueError("Invalid frozen CLIP text embeddings")
                self.cache.update(
                    (text, embedding.detach().clone())
                    for text, embedding in zip(texts, embeddings)
                )
        return torch.stack([self.cache[text] for text in instructions]).to(device)


def zero_projection(input_dim, output_dim):
    import torch

    projection = torch.nn.Linear(input_dim, output_dim)
    torch.nn.init.zeros_(projection.weight)
    torch.nn.init.zeros_(projection.bias)
    return projection
