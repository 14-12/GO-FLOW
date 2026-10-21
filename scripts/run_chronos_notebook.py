"""Exécuter uniquement le benchmark Chronos avec son noyau isolé local au projet."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

class ClientAvecProgression(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return result

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/07_chronos_benchmark.ipynb'
notebook = nbformat.read(path, as_version=4)
# Résoudre le kernelspec dans le projet, sans modifier les noyaux de l’environnement principal.
spec = KernelSpecManager(kernel_dirs=[str(root / '.venv-chronos/share/jupyter/kernels')])
manager = AsyncKernelManager(kernel_name='goflow-chronos', kernel_spec_manager=spec)
client = ClientAvecProgression(notebook, km=manager, kernel_name='goflow-chronos', timeout=None,
    resources={'metadata': {'path': str(root)}})
with client.setup_kernel():
    for i, cell in enumerate(notebook.cells):
        print(f'Cellule {i} : {cell.cell_type}', flush=True)
        try:
            client.execute_cell(cell, i)
        finally:
            nbformat.write(notebook, path)
nbformat.validate(notebook)
print('Benchmark Chronos terminé et sauvegardé.', flush=True)
