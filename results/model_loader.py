import torch
from transformer_lens import HookedTransformer

MODELS = {
    "gpt2-xl": {
        "name": "gpt2-xl",
        "n_layers": 48,
        "n_heads": 25,
        "d_model": 1600,
        "n_components": 1248,
    },
    "pythia-1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 2048,
        "n_components": 768,
    },
    "pythia-410m": {
        "name": "EleutherAI/pythia-410m",
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 1024,
        "n_components": 768,
    },
}


def load_model(model_key: str, device: str = "cuda") -> HookedTransformer:
    """Load a single model into the given device."""
    cfg = MODELS[model_key]
    print(f"Loading {model_key} ({cfg['d_model']}d, {cfg['n_layers']}L)...")
    model = HookedTransformer.from_pretrained(
        cfg["name"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    print(f"  Loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model


def unload_model(model) -> None:
    """Unload a model and free GPU memory."""
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print(f"  Freed. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
