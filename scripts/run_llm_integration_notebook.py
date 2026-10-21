"""Exécuter seulement 09 et vérifier la préservation des résultats existants."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[1]
protected = list((root / "notebooks").glob("0[1-8]*.ipynb"))
protected += [p for base in [root / "data", root / "results/model_benchmark", root / "results/decision_agent"]
              for p in base.rglob("*") if p.is_file()]
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
before = {p: digest(p) for p in protected}
path = root / "notebooks/09_llm_integration.ipynb"
notebook = nbformat.read(path, as_version=4)
spec = KernelSpecManager(kernel_dirs=[str(root / ".venv-llm/share/jupyter/kernels")])
manager = AsyncKernelManager(kernel_name="goflow-llm", kernel_spec_manager=spec)
class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return result
client = ProgressClient(notebook, km=manager, kernel_name="goflow-llm", timeout=420,
                        resources={"metadata": {"path": str(root)}})
with client.setup_kernel():
    for index, cell in enumerate(notebook.cells):
        print("Cellule", index, cell.cell_type, flush=True)
        try:
            client.execute_cell(cell, index)
        finally:
            nbformat.write(notebook, path)
nbformat.validate(notebook)
assert all(digest(p) == value for p, value in before.items()), "Fichier protégé modifié"
print("Notebook 09 exécuté ; notebooks 01–08, sources et résultats historiques inchangés.")
