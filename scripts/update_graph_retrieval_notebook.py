"""Actualiser le notebook 05 avec les fichiers sources récupérés et les ambiguïtés constatées."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/05_graph_preparation.ipynb'
notebook = nbformat.read(path, as_version=4)
if any(c.cell_type == 'markdown' and c.source.startswith('## Résultat de la récupération') for c in notebook.cells):
    raise SystemExit('La mise à jour existe déjà ; exécuter directement le notebook pour relire les données.')
meta_path = root / 'data/source/pems08/validated_provenance.json'
meta = json.loads(meta_path.read_text(encoding='utf-8'))
relative_folder = 'data/source/pems08/' + meta['commit']
report = meta['validation']

# Réviser uniquement le notebook de préparation, sans ouvrir les modèles à l’exécution.
notebook.cells[0].source = '''# Préparer le graphe PEMS08 — récupération et validation

Les fichiers `data/raw/distance.csv` et `data/raw/pems08.npz` ont été contrôlés : **0 octet chacun**. Ils ne sont pas lus. Une paire de données authentiques a été récupérée depuis le dépôt des auteurs ASTGCN, à une révision Git fixée, dans `data/source/pems08/`.

Le téléchargement a été exécuté séparément par `scripts/fetch_pems08.py --download`. Ce notebook ne fait aucune requête réseau et n’entraîne aucun modèle. Les deux fichiers téléchargés sont inspectés, comparés au CSV existant, et leur provenance est conservée avec URLs et SHA-256. Le remplacement des fichiers vides et l’export d’adjacence restent bloqués si le mapping n’est pas identifié de manière univoque.

**Constat à expliquer :** les 170 séries correspondent à leurs indices NPZ, mais 25 capteurs appartiennent à quatre groupes de séries entièrement identiques. Le CSV ne permet donc pas, à lui seul, de distinguer leurs IDs physiques. Aucune correspondance n’est inventée pour lever cette ambiguïté.'''

for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    s = cell.source
    if s.startswith('# Importer les outils de lecture'):
        s = s.replace('DISTANCE_PATH = None', f'DISTANCE_PATH = racine / "{relative_folder}/distance.csv"')
        s = s.replace('NPZ_PATH = None', f'NPZ_PATH = racine / "{relative_folder}/pems08.npz"')
        s = s.replace('PROVENANCE_PATH = None', 'PROVENANCE_PATH = racine / "data/source/pems08/validated_provenance.json"')
        s = s.replace('A = None', 'A = None\ngroupes_identiques = []')
        s = s.replace('if chemin.is_file():', 'if chemin.is_file() and chemin.stat().st_size > 0:')
        s += '''

# Montrer les fichiers vides sans les charger ; consulter la provenance de la récupération.
display(pd.DataFrame([{"chemin": str(racine / "data/raw" / nom),
    "taille_octets": (racine / "data/raw" / nom).stat().st_size}
    for nom in ["distance.csv", "pems08.npz"]]))
recuperation = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
print("Dépôt :", recuperation["repository"], "| révision :", recuperation["commit"])
print("URLs :", {nom: info["url"] for nom, info in recuperation["files"].items()})
print("SHA-256 :", {nom: info["sha256"] for nom, info in recuperation["files"].items()})'''
    elif s.startswith('tenseur = None'):
        insertion = '''        # Une égalité de valeurs ne suffit pas à identifier un capteur aux séries dupliquées.
        validations["csv_npz_values"] = True
        classes = {}
        for capteur in range(170):
            empreinte = hashlib.sha256(np.ascontiguousarray(tenseur[:, capteur, :]).tobytes()).hexdigest()
            classes.setdefault(empreinte, []).append(capteur)
        groupes_identiques = [groupe for groupe in classes.values() if len(groupe) > 1]
        print("Séries distinctes :", len(classes), "| groupes identiques :", groupes_identiques)
        print("Séries numériquement compatibles pour les 170 indices ; identité non univoque dans ces groupes.")
        assert not groupes_identiques, "Mapping CSV/NPZ non univoque : preuve de l’ordre d’export nécessaire pour les séries identiques."
'''
        s = s.replace('        validations["csv_npz"] = True', insertion + '        validations["csv_npz"] = True')
        s = s.replace('NPZ/CSV incompatibles : ', 'Validation du mapping CSV/NPZ bloquée : ')
    elif s.startswith('mapping_graphe = None'):
        s = s.replace('validations.get("csv_npz")', 'validations.get("csv_npz_values", validations.get("csv_npz"))')
    elif s.startswith('# Enregistrer l’adjacence'):
        # Distinguer une récupération réussie d’un mapping incomplet : conclusion B, pas C.
        s = s.replace('else:\n    conclusion = "C.',
            'elif validations.get("csv_npz_values") and groupes_identiques:\n'
            '    conclusion = "B. Vrais fichiers PEMS08 récupérés mais mapping encore incertain pour les séries identiques."\n'
            'else:\n    conclusion = "C.')
        s = s.replace('if validations.get("csv_npz") else "non évalué ou non validé"',
                      'if validations.get("csv_npz_values") else "non évalué ou non validé"')
        s = s.replace('"non confirmée"),', '"valeurs compatibles mais IDs ambigus dans quatre groupes" if validations.get("csv_npz_values") else "non confirmée"),')
        s = s.replace('lignes += ["- " + probleme',
            'lignes += ["", "Groupes de séries identiques : `" + str(groupes_identiques) + "`.",\n'
            '    "Séries comparées : 170 capteurs × 17856 timesteps × 3 variables ; détail dans pems08_series_comparison.csv.",\n'
            '    "Ordre réel des canaux : flow, occupy, speed. La numérotation du graphe vise les indices NPZ (chargeur des auteurs vérifié).",\n'
            '    "Originaux téléchargés conservés dans data/source/pems08/ ; fichiers vides de data/raw/ non remplacés.",\n'
            '    "Pour lever l’ambiguïté : obtenir le code ou manifeste de conversion qui prouve que location a été défini comme l’index capteur du NPZ, sans permutation."]\n'
            'lignes += ["- " + probleme')
    cell.source = s

# Afficher les validations de récupération et les aperçus, indépendamment du blocage d’adjacence.
notebook.cells.append(nbformat.v4.new_markdown_cell('''## Résultat de la récupération

**B. Vrais fichiers PEMS08 récupérés mais mapping encore incertain.**

La clé du NPZ est `data`, de shape **(17856, 170, 3)** : temps, capteurs, variables. L’ordre réel est **flow, occupy, speed**, déterminé par comparaison des données. Le fichier spatial contient **295 lignes**, colonnes **from, to, cost**, et couvre les **170 IDs de 0 à 169**. Les distances sont les valeurs publiées ; leur unité n’est pas supposée. Les doublons d’arêtes sont contrôlés et présentés dans les cellules précédentes.

Sur les 170 capteurs, les valeurs correspondent au même index, avec seulement des écarts de représentation décimale pour occupy. Cependant, quatre groupes comprennent **25 capteurs aux séries intégralement identiques** : 145 capteurs sont distinguables individuellement par leurs données. Pour les autres, une permutation au sein du groupe est indétectable à partir du CSV.

Il manque le **code ou manifeste de conversion ayant créé traffic_dataset.csv**, prouvant que `location` a été attribué directement à l’index capteur du NPZ. Les fichiers téléchargés sont conservés dans `data/source/pems08/`, avec URLs, révision et SHA-256. Les deux fichiers vides dans `data/raw/` restent inchangés. Aucun graphe n’est exporté et aucun STGCN n’est lancé.'''))
notebook.cells.append(nbformat.v4.new_code_cell('''# Afficher les résultats réellement mesurés lors de la récupération.
resume_recuperation = json.loads((racine / "results/spatial_audit/pems08_retrieval_validation.json").read_text(encoding="utf-8"))
display(pd.DataFrame([resume_recuperation]))
apercu = pd.read_csv(racine / "results/spatial_audit/pems08_comparison_preview.csv")
display(apercu)
print("Les fichiers vides n’ont pas été remplacés ; aucune adjacence n’a été sauvegardée.")'''))

nbformat.write(notebook, path)
NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print('Notebook 05 actualisé et exécuté ; conclusion B.')
