# Configs

The pipeline currently uses command-line arguments and writes the resolved configuration to JSON in the output directory:

- `training_config.json`
- `adaptive_xai_config.json`

This keeps the experiment self-documenting without adding a YAML dependency.
