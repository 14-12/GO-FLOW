"""Exporter les figures réellement produites, sans réentraîner ni refaire l’inférence."""
import base64
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/06_stgcn_benchmark.ipynb'
notebook = nbformat.read(path, as_version=4)
folder = root / 'results/model_benchmark'
figures = {
    5: ['stgcn_adjacency_heatmap.png'],
    6: ['stgcn_degree_distribution.png'],
    15: ['stgcn_learning_curve.png'],
    23: [f'stgcn_actual_vs_predicted_sensor_{node}.png' for node in [0, 10, 100]],
    24: ['stgcn_models_rmse.png'],
    25: ['stgcn_models_r2.png'],
}
for index, names in figures.items():
    outputs = [o['data']['image/png'] for o in notebook.cells[index].get('outputs', [])
               if 'image/png' in o.get('data', {})]
    assert len(outputs) == len(names), f'Figures de la cellule {index} non disponibles.'
    for data, name in zip(outputs, names):
        # Copier le rendu exact du notebook comme fichier PNG autonome.
        if isinstance(data, list):
            data = ''.join(data)
        (folder / name).write_bytes(base64.b64decode(data))
    # Prévoir aussi l’export lors de futures exécutions des cellules graphiques.
    cell = notebook.cells[index]
    if 'plt.savefig(' not in cell.source:
        filename = 'f"stgcn_actual_vs_predicted_sensor_{node}.png"' if index == 23 else repr(names[0])
        indent = '    ' if index == 23 else ''
        cell.source = cell.source.replace(indent + 'plt.show()',
            indent + f'plt.savefig(dossier_resultats / {filename}, dpi=150, bbox_inches="tight")\n' + indent + 'plt.show()')
nbformat.validate(notebook)
nbformat.write(notebook, path)
print('8 figures autonomes exportées ; aucun entraînement ni aucune inférence supplémentaire.')
