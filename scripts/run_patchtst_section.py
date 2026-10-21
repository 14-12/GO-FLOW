"""Exécuter seulement les nouvelles cellules et conserver leurs sorties."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('notebooks/03_model_benchmark.ipynb')
notebook = nbformat.read(path, as_version=4)
section = nbformat.v4.new_notebook(cells=notebook.cells[42:], metadata=notebook.metadata)
client = NotebookClient(section, timeout=None, kernel_name='python3',
                        resources={'metadata': {'path': str(Path.cwd())}})
with client.setup_kernel():
    for i, cell in enumerate(section.cells):
        print(f'Cellule {42 + i} : {cell.cell_type}', flush=True)
        try:
            client.execute_cell(cell, i)
        finally:
            notebook.cells[42:] = section.cells
            nbformat.write(notebook, path)
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, flush=True)
print('Section PatchTST exécutée et sauvegardée.', flush=True)
