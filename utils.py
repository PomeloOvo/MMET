"""Configuration helpers used by the training entry point."""

import yaml


def load_config(path):
    """Load an experiment configuration from YAML."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def save_config(config, path):
    """Serialize an experiment configuration as YAML."""
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
