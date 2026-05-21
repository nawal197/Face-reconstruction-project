"""
Expériences définies dans le cahier des charges
"""

EXPERIMENTS = {
    "exp1": {
        "name": "Comparaison GAN classiques",
        "pipelines": ["pix2pix", "pix2pixhd", "esrgan"],
        "scale": 4,
        "mask_type": "surgical",
        "description": "Comparer Pix2Pix vs Pix2PixHD vs ESRGAN pipeline",
    },
    "exp2": {
        "name": "CNN vs Transformers vs Diffusion",
        "pipelines": ["pix2pix", "transformer", "diffusion"],
        "scale": 4,
        "mask_type": "surgical",
        "description": "Paradigmes architecturaux",
    },
    "exp3": {
        "name": "Impact résolution",
        "pipelines": ["pix2pix"],
        "scales": [2, 4, 8],
        "mask_type": "surgical",
        "description": "Tester x2/x4/x8",
    },
    "exp4": {
        "name": "Masques artificiels vs réels",
        "pipelines": ["pix2pix", "identity_gan"],
        "scale": 4,
        "mask_types": ["surgical", "fabric", "n95", "real"],
        "description": "Domain gap entre masques synthétiques et réels",
    },
    "exp5": {
        "name": "Impact Identity Loss",
        "pipelines": ["identity_gan"],
        "scale": 4,
        "ablation": ["with_arcface", "without_arcface"],
        "description": "Avec/sans ArcFace loss",
    },
    "exp6": {
        "name": "Ablation study",
        "pipeline": "identity_gan",
        "scale": 4,
        "ablations": [
            "full",
            "no_perceptual",
            "no_gan",
            "no_arcface",
        ],
        "description": "Contribution de chaque loss",
    },
}
