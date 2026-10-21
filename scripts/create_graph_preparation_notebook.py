"""Créer un notebook de préparation du graphe avec blocage des sauvegardes non validées."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
destination = root / 'notebooks/05_graph_preparation.ipynb'
assert not destination.exists(), 'Le notebook existe déjà : réviser ses cellules directement.'
def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

# Protéger les fichiers déjà présents, dont les modèles et les notebooks précédents.
before = {p: checksum(p) for p in root.rglob('*') if p.is_file()}
cells = []
def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip()))

md('''# Préparer le graphe PEMS08 — validation avant sauvegarde

Ce notebook lit les fichiers locaux, compare les séries brutes et prépare une adjacency **uniquement si tous les contrôles passent**. Aucun STGCN n’est construit ; XGBoost, LSTM et PatchTST ne sont ni exécutés ni modifiés. Aucun fichier n’est téléchargé.

La présence de 170 capteurs ou d’IDs 0..169 ne suffit pas à établir le mapping. Les contrôles de contenu NPZ/CSV et la preuve du sens des IDs du graphe sont distincts. En cas de fichier manquant, d’ambiguïté ou de provenance non vérifiable, aucune matrice n’est exportée.''')
code('''# Importer les outils de lecture et de validation, sans dépendance aux modèles.
from pathlib import Path
from itertools import permutations
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

racine = Path.cwd()
if not (racine / "data/raw/traffic_dataset.csv").is_file():
    racine = racine.parent
assert (racine / "data/raw/traffic_dataset.csv").is_file()

# Renseigner des chemins explicites si plusieurs versions existent ou si les fichiers sont hors projet.
DISTANCE_PATH = None
NPZ_PATH = None
# Ce manifeste doit fournir une preuve vérifiable du mapping du graphe, pas une simple hypothèse.
PROVENANCE_PATH = None
problemes = []
validations = {}
statistiques = None
A = None
fichiers = sorted(p for p in racine.rglob("*") if p.is_file())
distances_candidates = [p for p in fichiers if p.name.lower() in {"distance.csv", "pems08.csv"}]
npz_candidates = [p for p in fichiers if p.suffix.lower() == ".npz"]
display(pd.DataFrame([{"chemin": str(p), "format": p.suffix, "taille_octets": p.stat().st_size}
                     for p in fichiers if p.suffix.lower() in {".csv", ".npz", ".npy", ".txt", ".json", ".pkl", ".zip"}]))
print("Fichiers de distances candidats :", [str(p) for p in distances_candidates])
print("Archives NPZ candidates :", [str(p) for p in npz_candidates])

def choisir(chemin_explicite, candidats, description):
    # Refuser de choisir arbitrairement entre plusieurs versions.
    if chemin_explicite is not None:
        chemin = Path(chemin_explicite)
        if chemin.is_file():
            return chemin
        problemes.append(description + " : chemin introuvable : " + str(chemin))
    elif len(candidats) == 1:
        return candidats[0]
    else:
        problemes.append(description + (" : fichier absent." if not candidats else " : plusieurs candidats, chemin explicite requis."))
    return None

distance_path = choisir(DISTANCE_PATH, distances_candidates, "distance.csv PEMS08")
npz_path = choisir(NPZ_PATH, npz_candidates, "pems08.npz PEMS08")
def sha256(chemin):
    # Relever les empreintes locales sans les présenter comme une authentification officielle.
    with chemin.open("rb") as flux:
        return hashlib.file_digest(flux, "sha256").hexdigest()

trafic = pd.read_csv(racine / "data/raw/traffic_dataset.csv")
display(trafic.head(5))
print("CSV actuel :", trafic.shape, "| locations :", trafic.location.nunique(), "| timesteps :", trafic.timestep.nunique())''')
md('''## 1. Arêtes, identifiants et distances

Le fichier attendu comporte deux identifiants d’extrémités et une distance. Les noms de colonnes sont examinés ; les colonnes ne sont pas réaffectées uniquement par leur position. On vérifie les valeurs finies, les IDs entiers, les distances non négatives et les doublons. Deux distances contradictoires pour une même arête bloquent la préparation.

Si les IDs sont hors de 0..169, une table `graph_sensor_id,npz_index` est nécessaire. Même des IDs dans cette plage demandent une preuve documentaire qu’il s’agit des indices du tenseur associé.''')
code('''aretes = None
if distance_path is None:
    print("Inspection des distances non exécutée : fichier non localisé.")
else:
    try:
        brut = pd.read_csv(distance_path)
        display(brut.head(10))
        print("Chemin :", distance_path, "| lignes :", len(brut), "| colonnes :", list(brut.columns))
        alias = {"from": "source", "source": "source", "source_sensor_id": "source", "src": "source",
                 "to": "destination", "target": "destination", "destination": "destination", "dst": "destination",
                 "target_sensor_id": "destination", "distance": "distance", "cost": "distance"}
        brut = brut.rename(columns={c: alias.get(str(c).strip().lower(), c) for c in brut.columns})
        assert not brut.columns.duplicated().any(), "Colonnes ambiguës après identification."
        assert {"source", "destination", "distance"}.issubset(brut.columns), "Schéma d’arêtes non reconnu."
        aretes = brut[["source", "destination", "distance"]].apply(pd.to_numeric, errors="raise")
        assert not aretes.empty
        assert np.isfinite(aretes.to_numpy()).all()
        assert (aretes[["source", "destination"]] % 1 == 0).all().all()
        aretes[["source", "destination"]] = aretes[["source", "destination"]].astype("int64")
        assert (aretes.distance >= 0).all(), "Distance négative."
        ids = set(aretes.source) | set(aretes.destination)
        print("IDs source min/max :", aretes.source.min(), aretes.source.max())
        print("IDs destination min/max :", aretes.destination.min(), aretes.destination.max())
        print("Nœuds distincts observés dans les arêtes :", len(ids), "| IDs :", sorted(ids))
        print("IDs hors 0..169 :", sorted(ids - set(range(170))))
        display(aretes.distance.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_frame())
        doublons = aretes.duplicated(["source", "destination"]).sum()
        print("Arêtes dupliquées :", int(doublons), "| boucles explicites :", int((aretes.source == aretes.destination).sum()))
        assert aretes.groupby(["source", "destination"]).distance.nunique().le(1).all(), "Distances contradictoires."
        aretes = aretes.drop_duplicates(["source", "destination"])
        validations["aretes"] = True
    except (ValueError, AssertionError, OSError) as erreur:
        aretes = None
        problemes.append("Arêtes invalides : " + str(erreur))''')
md('''## 2. Tenseur NPZ et comparaison avec le CSV

Les clés et formes de **tous les tableaux** sont affichées. Les axes sont identifiés par leurs tailles (17 856, 170, 3), sans présumer leur position. L’ordre des trois variables est recherché parmi les six permutations possibles.

Les premières observations des capteurs **0, 1, 10, 50, 100, 169** sont affichées. Leurs séries complètes servent à identifier une permutation unique, puis la correspondance est vérifiée sur **tous les capteurs et tous les timesteps**, avec `rtol=1e-5`, `atol=1e-6` pour les écarts de représentation numérique. Ces tolérances sont fixées avant lecture ; elles ne servent pas à rapprocher des séries incompatibles.''')
code('''tenseur = None
ordre_variables = None
comparaisons = None
if npz_path is None:
    print("Inspection du NPZ et comparaison des séries non exécutées : archive absente.")
else:
    try:
        tableaux = []
        with np.load(npz_path, allow_pickle=False) as archive:
            print("Chemin :", npz_path, "| clés :", archive.files)
            for cle in archive.files:
                tableau = archive[cle]
                print("Clé :", cle, "| dimensions :", tableau.shape, "| type :", tableau.dtype)
                print("Premières valeurs :", tableau.ravel()[:12])
                if tableau.ndim == 3 and sorted(tableau.shape) == [3, 170, 17856]:
                    axes = tuple(tableau.shape.index(taille) for taille in (17856, 170, 3))
                    tableaux.append((cle, axes, tableau.transpose(axes).copy()))
        assert len(tableaux) == 1, "Aucun tenseur unique compatible avec 17856 temps × 170 capteurs × 3 variables."
        cle_tenseur, axes_tenseur, tenseur = tableaux[0]
        print("Axes identifiés (temps, capteurs, variables) :", axes_tenseur)
        assert np.isfinite(tenseur).all()
        assert set(trafic.location) == set(range(170))
        assert not trafic.duplicated(["timestep", "location"]).any()
        assert sorted(trafic.timestep.unique()) == list(range(1, 17857))
        assert len(trafic) == 17856 * 170
        features = ["flow", "occupy", "speed"]
        csv_tenseur = trafic.sort_values(["timestep", "location"])[features].to_numpy().reshape(17856, 170, 3)
        capteurs = [0, 1, 10, 50, 100, 169]
        lignes = []
        for capteur in capteurs:
            for t in range(5):
                ligne = {"location": capteur, "timestep_csv": t + 1}
                ligne.update({f"NPZ_canal_{c}": tenseur[t, capteur, c] for c in range(3)})
                ligne.update({f"CSV_{f}": csv_tenseur[t, capteur, c] for c, f in enumerate(features)})
                lignes.append(ligne)
        display(pd.DataFrame(lignes))
        compatibles = [permutation for permutation in permutations(range(3)) if
            np.allclose(tenseur[:, capteurs, :][:, :, permutation], csv_tenseur[:, capteurs, :], rtol=1e-5, atol=1e-6)]
        assert len(compatibles) == 1, "Ordre des variables incompatible ou ambigu."
        ordre_variables = dict(zip(features, compatibles[0]))
        print("Ordre confirmé : feature CSV → canal NPZ :", ordre_variables)
        lignes = []
        for capteur in range(170):
            for feature, canal in ordre_variables.items():
                valeurs_csv = csv_tenseur[:, capteur, features.index(feature)]
                valeurs_npz = tenseur[:, capteur, canal]
                lignes.append({"location": capteur, "variable": feature, "canal_npz": canal,
                    "écart_max": float(np.max(np.abs(valeurs_npz - valeurs_csv))),
                    "correspondance": bool(np.allclose(valeurs_npz, valeurs_csv, rtol=1e-5, atol=1e-6))})
        comparaisons = pd.DataFrame(lignes)
        display(comparaisons[comparaisons.location.isin(capteurs)])
        assert comparaisons.correspondance.all(), "Au moins une série ne correspond pas à son index NPZ."
        validations["csv_npz"] = True
        print("Correspondance vérifiée sur les 170 capteurs : location = index NPZ ; timestep CSV = index temps NPZ + 1.")
    except (ValueError, AssertionError, OSError, KeyError) as erreur:
        problemes.append("NPZ/CSV incompatibles : " + str(erreur))''')
md('''## 3. Provenance et mapping des IDs du graphe

La [configuration des auteurs ASTGCN](https://github.com/Davidham3/ASTGCN-2019-mxnet/blob/master/configurations/PEMS08.conf) associe `distance.csv` et `pems08.npz` à PEMS08 et à 170 sommets. Leur [chargeur](https://github.com/Davidham3/ASTGCN-2019-mxnet/blob/master/lib/utils.py) utilise les deux premiers IDs comme indices de matrice, sans mapping intermédiaire. Cette documentation décrit la distribution publiée ; elle n’authentifie pas des fichiers locaux simplement parce qu’ils portent le même nom.

Pour autoriser l’export, renseigner `PROVENANCE_PATH` vers un manifeste réellement établi contenant : `dataset` (PEMS08), `distance_sha256`, `npz_sha256`, `source_url`, `evidence` (justification vérifiable de la provenance et du mapping), et `id_semantics` (`npz_index` si prouvé). Si les IDs sont physiques, ajouter `mapping_csv` vers une table **graph_sensor_id,npz_index**, documentée et couvrant les 170 capteurs. Ce notebook ne crée ni ce manifeste ni cette correspondance.

Les empreintes locales permettent de rattacher la preuve aux fichiers inspectés, mais ne constituent pas seules une preuve d’origine officielle. Ne pas utiliser un graphe METR-LA ou PEMS-BAY ni remplir les champs de preuve par une supposition.''')
code('''mapping_graphe = None
provenance = None
if aretes is not None and validations.get("csv_npz"):
    try:
        assert PROVENANCE_PATH is not None, "Preuve du mapping/provenance non fournie ; renseigner PROVENANCE_PATH après vérification documentaire."
        provenance_path = Path(PROVENANCE_PATH)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert provenance["dataset"].upper().replace("-", "") == "PEMS08"
        assert provenance["distance_sha256"] == sha256(distance_path)
        assert provenance["npz_sha256"] == sha256(npz_path)
        assert provenance.get("source_url") and provenance.get("evidence"), "Source et justification manquantes."
        ids = set(aretes.source) | set(aretes.destination)
        if provenance["id_semantics"] == "npz_index":
            assert ids.issubset(set(range(170)))
            mapping_graphe = {i: i for i in range(170)}
        else:
            chemin_mapping = Path(provenance["mapping_csv"])
            if not chemin_mapping.is_absolute():
                chemin_mapping = provenance_path.parent / chemin_mapping
            table_mapping = pd.read_csv(chemin_mapping)
            display(table_mapping.head(10))
            assert {"graph_sensor_id", "npz_index"}.issubset(table_mapping.columns)
            assert len(table_mapping) == 170
            assert table_mapping.graph_sensor_id.is_unique and table_mapping.npz_index.is_unique
            assert set(table_mapping.npz_index) == set(range(170))
            mapping_graphe = dict(zip(table_mapping.graph_sensor_id, table_mapping.npz_index))
            assert ids.issubset(mapping_graphe)
        validations["mapping_graphe"] = True
        print("Preuve documentaire fournie :", provenance["source_url"], provenance["evidence"])
    except (ValueError, AssertionError, OSError, KeyError) as erreur:
        mapping_graphe = None
        problemes.append("Mapping du graphe non validé : " + str(erreur))
else:
    print("Mapping du graphe non évalué : arêtes ou comparaison CSV/NPZ non validées.")''')
md('''## 4. Stratégie de pondération et contrôles du graphe

**Méthode prévue, à exécuter après validation :** conserver les orientations et uniquement les arêtes explicitement fournies. Ne pas ajouter l’arête inverse, de voisins consécutifs ou de boucles artificielles. Si le fichier est symétrique, la matrice le sera naturellement ; sinon, le futur STGCN devra accepter un support dirigé ou documenter séparément une symétrisation.

Pour une distance d, utiliser **exp(−d² / σ²)** avec σ égal à l’écart-type (ddof=0) des distances des arêtes uniques. Ce choix de noyau et d’échelle suit le [générateur de graphe des auteurs DCRNN](https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py). C’est une méthode de pondération publiée, **pas une prétendue matrice officielle PEMS08**. Contrairement à son seuil de parcimonie, aucun seuil supplémentaire n’est appliqué ici : on conserve le support des arêtes fourni. σ non positif ou un poids numériquement nul bloque l’export plutôt que de choisir un autre paramètre arbitraire.

Les métriques incluent les arcs orientés, le degré entrant/sortant moyen, les forces pondérées, les nœuds isolés et la symétrie. La diagonale reflète seulement les boucles explicites. Une matrice dirigée ne doit pas être utilisée silencieusement avec un opérateur spectral exigeant un graphe symétrique.''')
code('''if all(validations.get(cle, False) for cle in ["aretes", "csv_npz", "mapping_graphe"]):
    try:
        sigma = float(aretes.distance.to_numpy(dtype="float64").std(ddof=0))
        assert np.isfinite(sigma) and sigma > 0, "Écart-type des distances nul ou invalide."
        poids = np.exp(-np.square(aretes.distance.to_numpy(dtype="float64") / sigma))
        assert np.isfinite(poids).all() and (poids > 0).all(), "Poids non finis ou sous-flux numérique."
        A = np.zeros((170, 170), dtype="float64")
        sources = aretes.source.map(mapping_graphe).to_numpy(dtype="int64")
        destinations = aretes.destination.map(mapping_graphe).to_numpy(dtype="int64")
        A[sources, destinations] = poids
        assert A.shape == (170, 170) and np.isfinite(A).all()
        assert (A >= 0).all() and (A <= 1).all()
        assert np.count_nonzero(A) == len(aretes)
        support = A > 0
        hors_diagonale = support.copy()
        np.fill_diagonal(hors_diagonale, False)
        entrants = hors_diagonale.sum(axis=0)
        sortants = hors_diagonale.sum(axis=1)
        statistiques = {
            "shape": list(A.shape), "sigma": sigma, "arcs_total": int(support.sum()),
            "arcs_hors_diagonale": int(hors_diagonale.sum()), "boucles": int(np.trace(support)),
            "degre_sortant_moyen": float(sortants.mean()), "degre_entrant_moyen": float(entrants.mean()),
            "force_sortante_moyenne": float(A.sum(axis=1).mean()),
            "noeuds_isoles": np.flatnonzero((entrants + sortants) == 0).tolist(),
            "sans_sortie": np.flatnonzero(sortants == 0).tolist(),
            "sans_entree": np.flatnonzero(entrants == 0).tolist(),
            "support_symetrique": bool(np.array_equal(support, support.T)),
            "poids_symetriques": bool(np.allclose(A, A.T, rtol=1e-12, atol=1e-12)),
            "ecart_symetrie_max": float(np.abs(A - A.T).max()),
            "poids_min": float(poids.min()), "poids_max": float(poids.max()), "nan": int(np.isnan(A).sum())
        }
        display(pd.DataFrame([statistiques]))
        validations["adjacence"] = True
    except (ValueError, AssertionError) as erreur:
        A = None
        problemes.append("Construction du graphe bloquée : " + str(erreur))
else:
    print("Matrice non construite : prérequis non validés.")''')
md('''## 5. Export conditionnel et conclusion unique

L’export `data/processed/pems08_adjacency.npy` est autorisé seulement si les arêtes, toutes les séries CSV/NPZ, la preuve de mapping et les contrôles d’adjacence sont validés. Le fichier est relu pour vérifier son intégrité. Un Markdown décrit les entrées, la provenance, les comparaisons et la stratégie ; il indique explicitement les étapes non exécutées.

Si les fichiers manquent ou sont incompatibles, conclusion **C : STGCN à ne pas lancer**. Le notebook ne construit pas de graphe au mapping incertain (cas B) : il préfère bloquer. Le cas A concerne le support spatial validé ; le choix du futur opérateur STGCN, notamment sa compatibilité avec l’asymétrie éventuelle, reste à documenter.''')
code('''# Enregistrer l’adjacence uniquement après tous les contrôles.
valide = all(validations.get(cle, False) for cle in ["aretes", "csv_npz", "mapping_graphe", "adjacence"])
sortie = racine / "data/processed/pems08_adjacency.npy"
if valide:
    assert not sortie.exists(), "Une adjacence existe déjà : la comparer avant de la remplacer."
    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.save(sortie, A, allow_pickle=False)
    np.testing.assert_array_equal(np.load(sortie, allow_pickle=False), A)
    conclusion = "A. Graphe validé et prêt pour STGCN (support dirigé à respecter si non symétrique)."
else:
    conclusion = "C. Données nécessaires absentes, incompatibles ou non validées : STGCN à ne pas lancer."

# Produire un rapport même lorsque les données ne sont pas disponibles.
lignes = ["# Préparation du graphe PEMS08", "", "## Conclusion", "", conclusion, "",
    "## Entrées et provenance", "", f"- CSV : `{racine / 'data/raw/traffic_dataset.csv'}`.",
    f"- Fichier d’arêtes localisé : `{distance_path}`.", f"- NPZ localisé : `{npz_path}`.",
    "- Les noms de fichiers et les 170 locations ne prouvent pas l’origine officielle ni le mapping."]
for chemin in [distance_path, npz_path]:
    if chemin is not None:
        lignes.append(f"- SHA-256 local de `{chemin}` : `{sha256(chemin)}`.")
lignes += ["", "## Vérifications", "", "```json", json.dumps(validations, ensure_ascii=False, indent=2), "```",
    "", "## Mapping", "", "Ordre des variables confirmé : " + (str(ordre_variables) if validations.get("csv_npz") else "non évalué ou non validé"),
    "Correspondance location/index NPZ : " + ("vérifiée sur toutes les séries" if validations.get("csv_npz") else "non confirmée"),
    "Preuve du mapping des IDs du graphe : " + (json.dumps(provenance, ensure_ascii=False) if validations.get("mapping_graphe") else "non disponible ou non validée"),
    "", "## Méthode de création", "",
    "Méthode prévue : orientations du fichier conservées, poids exp(-d²/sigma²), sigma = écart-type des distances uniques (ddof=0), aucun seuil supplémentaire, aucune arête ajoutée.",
    "Méthode exécutée : " + ("oui" if valide else "non : aucun graphe construit/sauvegardé"),
    "Référence du noyau : https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py", "",
    "## Statistiques mesurées", "", "```json", json.dumps(statistiques, ensure_ascii=False, indent=2), "```",
    "", "## Blocages", ""]
lignes += ["- " + probleme for probleme in problemes] or ["Aucun."]
lignes += ["", "## Fichiers manquants ou preuves à obtenir", "",
    "Le couple PEMS08 `distance.csv` / `pems08.npz`, puis la documentation prouvant le sens des IDs et leur correspondance avec le tenseur. Si les IDs sont physiques, ajouter une table documentée graph_sensor_id,npz_index.",
    "Configuration des auteurs : https://github.com/Davidham3/ASTGCN-2019-mxnet/blob/master/configurations/PEMS08.conf", "",
    "Aucun téléchargement, aucun modèle entraîné, aucune relation spatiale déduite des IDs consécutifs."]
rapport = racine / "results/spatial_audit/graph_preparation.md"
rapport.parent.mkdir(parents=True, exist_ok=True)
rapport.write_text("\\n".join(lignes) + "\\n", encoding="utf-8")
display(Markdown("### Conclusion\\n\\n**" + conclusion + "**\\n\\n" + "\\n".join("- " + p for p in problemes)))
print("Rapport :", rapport)
print("Adjacence sauvegardée :", valide)''')

# Exécuter uniquement ce nouveau notebook et enregistrer ses sorties réelles.
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'}
})
nbformat.write(notebook, destination)
NotebookClient(notebook, timeout=180, kernel_name='python3',
    resources={'metadata': {'path': str(root)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, destination)
assert all(checksum(p) == value for p, value in before.items())
print('Notebook créé et validé :', destination)
print('Tous les fichiers préexistants sont conservés.')
