# =============================================================================
# config.py — Načítání konfigurace ze YAML souboru
#
# Jednoduchý pomocný modul který otevře default.yaml.
# =============================================================================

import yaml


def load_config(path="cfg/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg