"""Keep exported forecasting models in one dedicated module each."""

import ast
from pathlib import Path


def test_exported_models_have_dedicated_files():
    models_dir = Path(__file__).parents[1] / "neuralforecast" / "models"
    init_tree = ast.parse((models_dir / "__init__.py").read_text(encoding="utf-8"))

    imports = [
        (node.module, alias.name)
        for node in init_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module
        and not node.module.startswith("_")
        for alias in node.names
        if not alias.name.startswith("_")
    ]
    modules = [module for module, _ in imports]
    duplicates = sorted({module for module in modules if modules.count(module) > 1})
    assert not duplicates, f"Multiple exported models share modules: {duplicates}"

    for module, model in imports:
        path = models_dir / f"{module}.py"
        assert path.is_file(), f"Missing model module: {path.name}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        assert model in classes, f"{model} must be defined in {path.name}"
