"""Vérifier le notebook de diagnostic et le refus d'export sans objets entraînés."""
from pathlib import Path
import ast
import runpy
import nbformat

root = Path(__file__).resolve().parents[1]
script = root / "scripts/export_existing_xgboost.py"
ast.parse(script.read_text(encoding="utf-8"))
notebook = nbformat.read(root / "notebooks/08_llm_decision_agent.ipynb", as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        assert cell.execution_count is not None
        assert not any(output.output_type == "error" for output in cell.outputs)
try:
    runpy.run_path(str(script))
except RuntimeError as error:
    assert str(error).startswith("Objets absents du noyau 03")
else:
    raise AssertionError("L'export aurait dû être refusé sans objets du noyau 03.")
print("Notebook valide et exécuté ; refus d'export sans objets : OK.")
