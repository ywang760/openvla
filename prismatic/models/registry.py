"""
registry.py

Exhaustive list of pretrained VLMs (with full descriptions / links to corresponding names and sections of paper).
"""

# === Pretrained Model Registry ===
# fmt: off
MODEL_REGISTRY = {
    "llama2+7b": {
        "model_id": "llama2+7b",
        "names": ["Llama-2 7B"],
        "description": {
            "name": "Llama-2 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Letterbox",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        },
    },
    "llama2+13b": {
        "model_id": "llama2+13b",
        "names": ["Llama-2 13B"],
        "description": {
            "name": "Llama-2 13B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Letterbox",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        },
    },

    "llama2-no-cotraining+7b": {
        "model_id": "llama2-no-cotraining+7b",
        "names": ["Llama-2 7B (No Co-training)"],
        "description": {
            "name": "Llama-2 7B (No Co-training)",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Letterbox",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Multimodal-Only"],
            "train_epochs": 1,
        },
    },

    # === CLIP Prism Models ===
    "prism-clip-controlled+7b": {
        "model_id": "prism-clip-controlled+7b",
        "names": ["Prism-CLIP 7B (Controlled)"],
        "description": {
            "name": "CLIP Prism 7B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-clip-controlled+13b": {
        "model_id": "prism-clip-controlled+13b",
        "names": ["Prism-CLIP 13B (Controlled)"],
        "description": {
            "name": "CLIP Prism 13B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-clip+7b": {
        "model_id": "prism-clip+7b",
        "names": ["Prism-CLIP 7B"],
        "description": {
            "name": "CLIP Prism 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
    "prism-clip+13b": {
        "model_id": "prism-clip+13b",
        "names": ["Prism-CLIP 13B"],
        "description": {
            "name": "CLIP Prism 13B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },

    # === SigLIP Prism Models ==
    "prism-siglip-controlled+7b": {
        "model_id": "prism-siglip-controlled+7b",
        "names": ["Prism-SigLIP 7B (Controlled)"],
        "description": {
            "name": "SigLIP Prism 7B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-siglip-controlled+13b": {
        "model_id": "prism-siglip-controlled+7b",
        "names": ["Prism-SigLIP 13B (Controlled)"],
        "description": {
            "name": "SigLIP Prism 13B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-siglip+7b": {
        "model_id": "prism-siglip+7b",
        "names": ["Prism-SigLIP 7B"],
        "description": {
            "name": "SigLIP Prism 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        }
    },
    "prism-siglip+13b": {
        "model_id": "prism-siglip+13b",
        "names": ["Prism-SigLIP 13B"],
        "description": {
            "name": "SigLIP Prism 13B",
            "optimization_procedure": "single-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        }
    },

    # === DINOSigLIP Prism Models ===
    "prism-dinosiglip-controlled+7b": {
        "model_id": "prism-dinosiglip-controlled+7b",
        "names": ["Prism-DINOSigLIP 7B (Controlled)", "Prism 7B (Controlled)"],
        "description": {
            "name": "DINOSigLIP Prism 7B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-dinosiglip-controlled+13b": {
        "model_id": "prism-dinosiglip-controlled+13b",
        "names": ["Prism-DINOSigLIP 13B (Controlled)", "Prism 13B (Controlled)"],
        "description": {
            "name": "DINOSigLIP Prism 13B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-dinosiglip+7b": {
        "model_id": "prism-dinosiglip+7b",
        "names": ["Prism-DINOSigLIP 7B"],
        "description": {
            "name": "DINOSigLIP Prism 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },
    "prism-dinosiglip+13b": {
        "model_id": "prism-dinosiglip+13b",
        "names": ["Prism-DINOSigLIP 13B"],
        "description": {
            "name": "DINOSigLIP Prism 13B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO/14 @ 384px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 13B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        },
    },

    # === DINOSigLIP 224px Prism Models ===
    "prism-dinosiglip-224px-controlled+7b": {
        "model_id": "prism-dinosiglip-224px-controlled+7b",
        "names": ["Prism-DINOSigLIP 224px 7B (Controlled)"],
        "description": {
            "name": "DINOSigLIP 224px 7B (Controlled)",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO 14 @ 224px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "prism-dinosiglip-224px+7b": {
        "model_id": "prism-dinosiglip-224px+7b",
        "names": ["Prism-DINOSigLIP 224px 7B"],
        "description": {
            "name": "DINOSigLIP 224px 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINOv2 ViT-L/14 + SigLIP ViT-SO 14 @ 224px",
            "image_processing": "Naive Resize",
            "language_model": "Llama-2 7B",
            "datasets": ["LLaVa v1.5 Instruct", "LVIS-Instruct-4V", "LRV-Instruct"],
            "train_epochs": 2,
        }
    },

    # === Additional LLM Backbones ===
    "llama2-chat+7b": {
        "model_id": "llama2-chat+7b",
        "names": ["Llama-2 Chat 7B"],
        "description": {
            "name": "Llama-2 Chat 7B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Letterbox",
            "language_model": "Llama-2 Chat 7B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
    "llama2-chat+13b": {
        "model_id": "llama2-chat+13b",
        "names": ["Llama-2 Chat 13B"],
        "description": {
            "name": "Llama-2 Chat 13B",
            "optimization_procedure": "single-stage",
            "visual_representation": "CLIP ViT-L/14 @ 336px",
            "image_processing": "Letterbox",
            "language_model": "Llama-2 Chat 13B",
            "datasets": ["LLaVa v1.5 Instruct"],
            "train_epochs": 1,
        }
    },
}

# Build Global Registry (Model ID, Name) -> Metadata
GLOBAL_REGISTRY = {name: v for k, v in MODEL_REGISTRY.items() for name in [k] + v["names"]}

# fmt: on
