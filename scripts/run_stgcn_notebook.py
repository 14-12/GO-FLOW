"""Exécuter uniquement le notebook 06 et sauvegarder chaque cellule terminée."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

class ClientAvecProgression(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        # Afficher les messages au fil de l’entraînement sans réexécuter les cellules.
        result = super().process_message(msg, cell, cell_index)
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return result

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/06_stgcn_benchmark.ipynb'
notebook = nbformat.read(path, as_version=4)
client = ClientAvecProgression(notebook, timeout=None, kernel_name='python3',
    resources={'metadata': {'path': str(root)}})
with client.setup_kernel():
    for i, cell in enumerate(notebook.cells):
        print(f'Cellule {i} : {cell.cell_type}', flush=True)
        try:
            client.execute_cell(cell, i)
        finally:
            nbformat.write(notebook, path)
nbformat.validate(notebook)
print('Benchmark STGCN terminé et sauvegardé.', flush=True)
