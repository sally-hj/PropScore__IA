from __future__ import annotations

import importlib.util
from pathlib import Path


def main():
    module_path = Path(__file__).resolve().with_name("00_prepare_tabular_dataset.py")
    spec = importlib.util.spec_from_file_location("prepare_tabular_dataset_00", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
