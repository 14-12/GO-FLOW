"""Créer et exécuter uniquement le nouveau notebook d’audit spatial."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
target = root / 'notebooks/04_spatial_data_audit.ipynb'
assert not target.exists() or nbformat.read(target, as_version=4).cells[0].source.startswith(
    '# Audit spatial — préparation STGCN'), 'Un autre notebook existe à cette destination.'

# Mémoriser tous les fichiers existants avant la création du nouveau notebook.
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

before = {p: digest(p) for p in root.rglob('*') if p.is_file() and p != target}
cells = []
def md(source):
    cells.append(nbformat.v4.new_markdown_cell(source.strip()))
def code(source):
    cells.append(nbformat.v4.new_code_cell(source.strip()))

md('''# Audit spatial — préparation STGCN sur PEMS08

Cet audit lit les fichiers du projet pour déterminer si un graphe **PEMS08** est disponible et si ses nœuds correspondent aux **170 locations** du CSV. Aucun modèle n’est entraîné, aucun notebook existant n’est exécuté, aucune donnée n’est téléchargée et aucune adjacence n’est créée.

Le contexte déclaré est : flow, speed, occupy, intervalles de 5 minutes et prévision à +15 minutes. Une matrice 170 × 170 ou des identifiants consécutifs ne suffisent pas à prouver la provenance ou le voisinage. Les documents de l’audit précédent constituent des comptes rendus, pas des données spatiales.''')
code('''# Importer uniquement les outils de lecture, d’inventaire et d’affichage.
from pathlib import Path
import hashlib
import json
import re
import pickletools
from collections import Counter
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

# Accepter une exécution depuis la racine ou depuis notebooks.
racine = Path.cwd()
if not (racine / "data/raw/traffic_dataset.csv").is_file():
    racine = racine.parent
assert (racine / "data/raw/traffic_dataset.csv").is_file()
chemin_audit = racine / "notebooks/04_spatial_data_audit.ipynb"
fichiers = sorted(p for p in racine.rglob("*") if p.is_file() and p != chemin_audit)

def empreinte(chemin):
    # Lire par blocs et ne modifier aucun fichier.
    with chemin.open("rb") as flux:
        return hashlib.file_digest(flux, "sha256").hexdigest()

empreintes_avant = {p: empreinte(p) for p in fichiers}
inventaire = pd.DataFrame([{
    "chemin": p.relative_to(racine).as_posix(), "format": p.suffix.lower(),
    "taille_octets": p.stat().st_size
} for p in fichiers])
with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
    display(inventaire)

# Afficher aussi les dossiers absents explicitement demandés dans l’audit.
display(pd.DataFrame([{"dossier": nom, "présent": (racine / nom).is_dir()}
    for nom in ["data", "data/raw", "data/processed", "datasets", "results"]]))
print("Arborescence complète :")
for chemin in sorted(racine.rglob("*")):
    relatif = chemin.relative_to(racine)
    print("  " * (len(relatif.parts) - 1) + relatif.as_posix() + ("/" if chemin.is_dir() else ""))
comptes = Counter(p.suffix.lower() for p in fichiers)
display(pd.DataFrame([{"format": ext, "nombre": comptes[ext]}
    for ext in sorted(set(comptes) | {".csv", ".npz", ".npy", ".pkl", ".txt", ".json"})]))''')
md('''## Vérifier les séries et les identifiants locaux

Les valeurs de `location` sont examinées comme des identifiants locaux. Leur nombre, leur ordre numérique et les corrélations de trafic ne constituent pas un graphe routier. On vérifie aussi la continuité temporelle et l’unicité des observations, sans recalculer de features ni de split.''')
code('''# Lire la source sans reconstruire les ensembles du benchmark.
trafic = pd.read_csv(racine / "data/raw/traffic_dataset.csv")
locations = set(trafic["location"].unique().tolist())
assert locations == set(range(170)), "Les locations ne correspondent pas à 0..169."
assert list(trafic.columns) == ["timestep", "location", "flow", "occupy", "speed"]
ordre = trafic.sort_values(["location", "timestep"])
assert ordre.groupby("location")["timestep"].diff().dropna().eq(1).all()
assert not trafic.duplicated(["location", "timestep"]).any()
display(trafic.head(5))
display(pd.DataFrame([{
    "lignes": len(trafic), "colonnes": len(trafic.columns), "locations": len(locations),
    "location_min": min(locations), "location_max": max(locations),
    "timesteps": trafic.timestep.nunique(), "timestep_min": trafic.timestep.min(),
    "timestep_max": trafic.timestep.max()
}]))
display(trafic.groupby("location").size().describe().to_frame("observations par capteur"))
print("Pas de coordonnées, distances, arêtes ou IDs physiques dans les colonnes du CSV.")''')
md('''## Examiner les fichiers susceptibles de contenir un graphe

L’inspection porte sur tous les CSV, tableaux NumPy, JSON, TXT et PKL, même si leur nom ne contient pas « graph ». Chaque chemin, format, dimension et aperçu est affiché. Les CSV de scores et les inventaires d’audit sont distingués des données brutes.

Pour un PKL éventuel, les opérations pickle sont examinées sans désérialisation exécutable ; les dimensions demanderaient alors une inspection contrôlée de sa provenance. Les chargements NumPy utilisent `allow_pickle=False`. Les checkpoints existants sont lus avec `weights_only=True`, sans instancier les modèles ni effectuer d’inférence.''')
code('''# Repérer les noms évocateurs et les colonnes décrivant des relations spatiales.
motif_spatial = re.compile(r"adj_mx|adjacency|adjacence|distance|edges?|graph|sensor[_ -]?(ids?|locations?)|coordinates?|latitude|longitude", re.I)
colonnes_spatiales = {"from", "to", "source", "target", "source_sensor_id", "target_sensor_id",
    "sensor_id", "sensor_ids", "latitude", "longitude", "lat", "lon", "distance", "weight"}
candidats_spatiaux = []
resumes = []
descriptions = {
    "traffic_dataset.csv": "Mesures temporelles ; identifiants locaux seulement.",
    "benchmark_results.csv": "Métriques et temps des modèles.",
    "lstm_normalisation.csv": "Statistiques de normalisation sur train.",
    "lstm_training_history.csv": "Historique d’apprentissage LSTM.",
    "patchtst_training_history.csv": "Historique d’apprentissage PatchTST.",
    "xgboost_feature_importance.csv": "Importance des features XGBoost.",
    "xgboost_validation_results.csv": "Scores de validation et configurations XGBoost.",
    "file_inventory.csv": "Inventaire de l’audit précédent ; les chemins ne sont pas des arêtes."
}

def afficher_tableau(chemin, tableau, interpretation):
    # Montrer les dimensions et les premières lignes réellement lues.
    display(Markdown(f"### `{chemin.relative_to(racine).as_posix()}`"))
    print("Chemin absolu :", chemin)
    print("Format :", chemin.suffix, "| dimensions :", tableau.shape)
    print(interpretation)
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 120):
        display(tableau.head(5))

for chemin in fichiers:
    extension = chemin.suffix.lower()
    nom_spatial = bool(motif_spatial.search(chemin.name))
    # Les rapports d’audit sont des documents et non des fichiers de graphe.
    document_audit = "spatial_audit" in chemin.parts or chemin.name in {"audit_spatial_data.py", "create_spatial_audit_notebook.py"}
    potentiel = nom_spatial and not document_audit
    if extension == ".csv":
        tableau = trafic if chemin.name == "traffic_dataset.csv" else pd.read_csv(chemin)
        colonnes = {str(c).lower().strip() for c in tableau.columns}
        # Une cible de benchmark nommée target n’est pas une extrémité d’arête.
        schema_spatial = bool(colonnes & (colonnes_spatiales - {"from", "to", "source", "target", "weight"}))
        schema_spatial |= {"from", "to"}.issubset(colonnes) or {"source", "target"}.issubset(colonnes)
        schema_spatial |= {"source_sensor_id", "target_sensor_id"}.issubset(colonnes)
        potentiel |= schema_spatial and not document_audit
        interpretation = descriptions.get(chemin.name, "Contenu à interpréter ; provenance non attestée.")
        afficher_tableau(chemin, tableau, interpretation)
        if potentiel:
            # Comparer uniquement les colonnes d’identifiants explicites, sans créer d’arêtes.
            for nom in tableau.columns:
                if str(nom).lower() in {"from", "to", "source", "target", "sensor_id", "source_sensor_id", "target_sensor_id"}:
                    ids = pd.to_numeric(tableau[nom], errors="coerce")
                    print("Colonne", nom, "| valeurs communes avec location :", len(set(ids.dropna()) & locations),
                          "| toutes dans 0..169 :", ids.notna().all() and ids.isin(locations).all())
            print("Une correspondance numérique seule ne prouve pas l’identité des capteurs ni PEMS08.")
        resumes.append({"chemin": str(chemin.relative_to(racine)), "format": extension,
                        "dimensions": str(tableau.shape), "interprétation": interpretation})
    elif extension in {".npy", ".npz"}:
        potentiel = True
        print("Chemin :", chemin, "| format :", extension)
        objet = np.load(chemin, allow_pickle=False)
        tableaux = {"tableau": objet} if extension == ".npy" else {k: objet[k] for k in objet.files}
        for cle, tableau in tableaux.items():
            print("Clé :", cle, "| dimensions :", tableau.shape, "| type :", tableau.dtype)
            print("Premières valeurs :", tableau.ravel()[:12])
            print("Forme et nom à vérifier ; aucune provenance ni relation spatiale supposée.")
        if extension == ".npz":
            objet.close()
    elif extension in {".txt", ".json"}:
        contenu = chemin.read_text(encoding="utf-8-sig")
        print("Chemin :", chemin, "| format :", extension)
        if extension == ".json":
            objet = json.loads(contenu)
            print("Type :", type(objet).__name__, "| éléments de premier niveau :", len(objet) if isinstance(objet, (dict, list)) else 1)
            if isinstance(objet, dict):
                potentiel |= bool(motif_spatial.search(" ".join(objet.keys())))
            print("Aperçu :", json.dumps(objet, ensure_ascii=False)[:1200])
        else:
            print("Dimensions :", len(contenu.splitlines()), "lignes")
            print("Aperçu :", "\\n".join(contenu.splitlines()[:5]))
        print("La provenance et la signification du contenu doivent être documentées.")
    elif extension in {".pkl", ".pickle"}:
        potentiel = True
        print("Chemin :", chemin, "| format :", extension, "| dimensions : non déterminées sans désérialisation")
        operations = pickletools.genops(chemin.read_bytes())
        for numero, (operation, argument, position) in enumerate(operations):
            print(position, operation.name, str(argument)[:120])
            if numero >= 11:
                break
        print("Conteneur à vérifier ; aucun chargement exécutable ni provenance présumée.")
    elif extension == ".pt":
        import torch
        objet = torch.load(chemin, map_location="cpu", weights_only=True)
        print("Chemin :", chemin, "| format : checkpoint PyTorch")
        print("Clés :", list(objet), "| variables :", objet.get("feature_names"))
        etat = objet.get("state_dict", {})
        print("Nombre de tenseurs :", len(etat))
        for nom, poids in list(etat.items())[:3]:
            print(nom, tuple(poids.shape), "| aperçu :", poids.flatten()[:6].tolist())
        potentiel |= any(motif_spatial.search(str(k)) for k in objet if k != "state_dict")
        print("Poids et paramètres de prétraitement, pas de modèle exécuté.")
    elif extension == ".xlsx":
        with pd.ExcelFile(chemin) as classeur:
            for feuille in classeur.sheet_names:
                tableau = pd.read_excel(classeur, sheet_name=feuille)
                afficher_tableau(chemin, tableau, "Feuille " + feuille + " : tableau de résultats.")
    if potentiel:
        candidats_spatiaux.append(chemin)

display(pd.DataFrame(resumes))
print("Fichiers spatiaux potentiels :", [str(p.relative_to(racine)) for p in candidats_spatiaux])''')
md('''## Provenance : PEMS08, METR-LA ou PEMS-BAY ?

On recherche les noms de datasets, les chemins de chargement et une éventuelle conversion depuis un tenseur original dans les sources locales. Les mentions « PEMS08 » dans les notebooks sont des **déclarations**, pas une vérification d’origine. Les rapports d’audit et ce notebook sont exclus de cette recherche pour éviter de prendre leurs propres références pour des preuves.

Sans archive source, manifeste, liste ordonnée de capteurs ou code de conversion, le CSV ne peut pas être authentifié uniquement grâce à ses 170 locations. De même, un fichier appelé `adj_mx.pkl` ne serait pas automatiquement accepté : son dataset et l’ordre de ses nœuds devraient être vérifiés.''')
code('''# Lire le texte des notebooks et scripts existants, sans les exécuter.
motif_origine = re.compile(r"PEMS[-_ ]?0?8|METR[-_ ]?LA|PEMS[-_ ]?BAY|read_csv|np\\.load|numpy\\.load|sensor_ids|adj_mx", re.I)
traces = []
for chemin in fichiers:
    if chemin.suffix.lower() not in {".ipynb", ".py", ".md", ".txt", ".json"}:
        continue
    if "spatial_audit" in chemin.parts or chemin.name in {"audit_spatial_data.py", "create_spatial_audit_notebook.py"}:
        continue
    texte = chemin.read_text(encoding="utf-8")
    if chemin.suffix.lower() == ".ipynb":
        texte = "\\n".join("".join(c["source"]) for c in json.loads(texte)["cells"])
    for numero, ligne in enumerate(texte.splitlines(), 1):
        if motif_origine.search(ligne):
            traces.append({"chemin": str(chemin.relative_to(racine)), "ligne": numero,
                           "trace": ligne.strip()[:240]})
with pd.option_context("display.max_rows", None, "display.max_colwidth", 240):
    display(pd.DataFrame(traces))
mentions_autres = [t for t in traces if re.search(r"METR[-_ ]?LA|PEMS[-_ ]?BAY", t["trace"], re.I)]
print("Mentions d’autres datasets dans les sources préexistantes :", len(mentions_autres))
print("Provenance du CSV : PEMS08 déclaré ; origine et mapping non authentifiés faute de source originale.")
print("Mapping spatial : non vérifiable, aucun fichier spatial local n’est présent dans cet audit.")''')
md('''## Fichiers exacts manquants et référence des auteurs

La [configuration PEMS08 du dépôt original ASTGCN](https://github.com/Davidham3/ASTGCN-2019-mxnet/blob/master/configurations/PEMS08.conf) associe explicitement :

- **`data/PEMS08/distance.csv`** : fichier désigné par `adj_filename`, pour le graphe de 170 capteurs ;
- **`data/PEMS08/pems08.npz`** : fichier de mesures désigné par `graph_signal_matrix_filename` ;
- **170 sommets et 12 mesures par heure**, donc des pas de 5 minutes.

Le [dépôt ASTGNN des auteurs](https://github.com/guoshnBJTU/ASTGNN/blob/main/configurations/PEMS08.conf) emploie les noms **`data/PEMS08/PEMS08.csv`** et **`data/PEMS08/PEMS08.npz`**. Il s’agit d’une autre distribution publiée ; ses fichiers doivent être utilisés comme un ensemble cohérent, après vérification de compatibilité avec le CSV GoFlow.

Ces références sont issues d’une consultation de documentation, pas d’un téléchargement des données. Le notebook ne contient aucune requête réseau.

Pour construire STGCN correctement, il manque **le fichier de relations PEMS08 associé aux mesures**, et **la preuve de correspondance entre ses nœuds et les locations 0 à 169**. Récupérer aussi le code de conversion ou un manifeste `location → sensor_id/index_source`, puis confronter les séries du tenseur original au CSV pour vérifier les 170 capteurs, l’ordre temporel et l’ordre des variables. La présence de 170 nœuds ou d’IDs 0..169 seule ne suffit pas. Il faut documenter la direction des arêtes, les unités de distance et la convention des poids ; aucune de ces informations n’est inventée ici.''')
code('''# Refuser de réutiliser une conclusion négative si de nouveaux candidats apparaissent.
assert not candidats_spatiaux, (
    "De nouveaux candidats spatiaux sont présents : examiner leur provenance et leur mapping, "
    "puis réviser la conclusion. Ne pas valider A automatiquement à partir des dimensions."
)
# Contrôler la conservation des fichiers existants après toutes les lectures.
assert all(empreinte(p) == valeur for p, valeur in empreintes_avant.items())
display(Markdown("**Vérification réussie : fichiers existants conservés (SHA-256 identiques), "
                 "aucun entraînement, aucune adjacence générée.**"))''')
md('''## Conclusion de l’audit local

**C. Aucune donnée spatiale PEMS08 disponible dans le projet au moment de cet audit.**

Le seul fichier brut est `data/raw/traffic_dataset.csv` : **3 035 520 × 5**, 170 locations, 17 856 timesteps par capteur. Aucun graphe, fichier de distances, coordonnées ou liste d’IDs physiques n’est présent. `data/processed/` et `datasets/` sont absents. Les fichiers sous `results/` décrivent les benchmarks et l’audit précédent.

Les sources existantes déclarent PEMS08, sans trace de METR-LA ou PEMS-BAY. Cette absence ne prouve pas l’origine du CSV : le tenseur initial et le mapping ne sont pas disponibles. Aucun mapping vers un fichier spatial ne peut donc être confirmé.

Avant STGCN, récupérer le **`distance.csv` PEMS08** et son **`pems08.npz` associé** (ou le couple `PEMS08.csv` / `PEMS08.npz` de la distribution des auteurs), ainsi que la documentation ou le code établissant la correspondance avec `location`. Aucun voisinage n’est déduit de la numérotation des capteurs.''')

# Exécuter le nouveau notebook seul, sans réseau ni appel aux modèles existants.
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'}
})
nbformat.write(notebook, target)
client = NotebookClient(notebook, timeout=180, kernel_name='python3',
    resources={'metadata': {'path': str(root)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, target)
assert all(digest(p) == valeur for p, valeur in before.items())
print('Notebook créé, exécuté et validé :', target)
print('Fichiers préexistants conservés :', len(before))
