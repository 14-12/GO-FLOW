"""Auditer les fichiers locaux pour préparer STGCN, sans exécuter les notebooks."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
import torch

racine = Path(__file__).resolve().parents[1]
dossier = racine / 'results' / 'spatial_audit'

# Inventorier aussi les fichiers cachés ; exclure uniquement les sorties de cet audit.
fichiers = sorted(p for p in racine.rglob('*') if p.is_file() and dossier not in p.parents)
def empreinte(chemin):
    calcul = hashlib.sha256()
    with chemin.open('rb') as flux:
        for bloc in iter(lambda: flux.read(1024 * 1024), b''):
            calcul.update(bloc)
    return calcul.hexdigest()

# Mémoriser les empreintes pour contrôler que tous les fichiers existants restent intacts.
avant = {p: empreinte(p) for p in fichiers}
rapport = ['# Audit spatial local — GoFlow / PEMS08', '',
    'Périmètre : tous les fichiers du projet, y compris les fichiers cachés. '
    'Les notebooks sont lus comme documents JSON et ne sont jamais exécutés. '
    'Aucun entraînement ni aucune construction de matrice d’adjacence.', '',
    '## Arborescence complète', '', '```text', 'GoFlow/']
for chemin in sorted(racine.rglob('*')):
    if chemin == dossier or dossier in chemin.parents:
        continue
    relatif = chemin.relative_to(racine)
    rapport.append('  ' * len(relatif.parts) + chemin.name + ('/' if chemin.is_dir() else ''))
rapport.extend(['```', '', '## Inventaire des formats', '',
    '| Format | Nombre |', '| --- | ---: |'])
comptes = Counter(p.suffix.lower() for p in fichiers)
for suffixe, nombre in sorted(comptes.items()):
    rapport.append(f'| {suffixe or "sans extension"} | {nombre} |')
for suffixe in ['.npz', '.npy', '.pkl', '.pickle', '.txt', '.json']:
    if comptes[suffixe] == 0:
        rapport.append(f'| {suffixe} | 0 |')

# Examiner chaque CSV, même les résultats, pour ne pas se fier uniquement aux noms.
rapport.extend(['', '## Contenu des fichiers tabulaires', ''])
descriptions = {
    'traffic_dataset.csv': 'Mesures temporelles de trafic ; aucune colonne spatiale ni identifiant routier externe.',
    'benchmark_results.csv': 'Scores et temps des modèles ; ne représente pas les relations entre capteurs.',
    'lstm_normalisation.csv': 'Moyennes et écarts-types des trois variables sur train ; aucune relation spatiale.',
    'lstm_training_history.csv': 'Historique des époques LSTM ; aucune relation spatiale.',
    'patchtst_training_history.csv': 'Historique des époques PatchTST ; aucune relation spatiale.',
    'xgboost_feature_importance.csv': 'Importance des features XGBoost ; location est une feature et non une arête.',
    'xgboost_validation_results.csv': 'Configurations XGBoost et scores de validation ; aucune relation spatiale.',
}
donnees = None
for chemin in fichiers:
    if chemin.suffix.lower() != '.csv':
        continue
    tableau = pd.read_csv(chemin)
    if chemin.name == 'traffic_dataset.csv':
        donnees = tableau
    rapport.extend([f'### `{chemin.relative_to(racine).as_posix()}`', '',
        f'Chemin absolu : `{chemin}`. Format : CSV. Dimensions : **{tableau.shape[0]:,} × {tableau.shape[1]}**.', '',
        descriptions.get(chemin.name, 'Fichier à interpréter manuellement.'), '',
        'Colonnes : ' + ', '.join(f'`{c}`' for c in tableau.columns) + '.', '',
        'Premières lignes (valeurs du fichier) :', '', '```csv',
        tableau.head(5).to_csv(index=False).rstrip(), '```', ''])

# Contrôler les identifiants sans leur attribuer de proximité géographique.
assert donnees is not None
identifiants = sorted(donnees['location'].unique().tolist())
tailles = donnees.groupby('location').size()
rapport.extend(['## Correspondance des 170 locations', '',
    f'- Nombre de locations : **{len(identifiants)}**.',
    f'- Identifiants locaux observés : `{identifiants}`.',
    f'- Timesteps distincts : **{donnees.timestep.nunique()}**, de **{donnees.timestep.min()}** à **{donnees.timestep.max()}**.',
    f'- Lignes par location : minimum **{tailles.min()}**, maximum **{tailles.max()}**.',
    f'- Couples location/timestep dupliqués : **{int(donnees.duplicated(["location", "timestep"]).sum())}**.',
    '- Les valeurs 0 à 169 sont des identifiants locaux. Leur origine et leur correspondance avec des IDs physiques ne sont pas documentées dans les fichiers présents.',
    '- Aucun fichier spatial indépendant ne fournit des IDs à comparer : le mapping est **invérifiable à ce stade**.',
    '- Des IDs consécutifs ne constituent pas une preuve de voisinage. Aucune arête n’est déduite de location.', ''])

# Lire uniquement les métadonnées et formes des checkpoints avec le chargement restreint.
rapport.extend(['## Vérification des autres conteneurs', ''])
for chemin in fichiers:
    if chemin.suffix.lower() == '.pt':
        objet = torch.load(chemin, map_location='cpu', weights_only=True)
        etat = objet['state_dict']
        formes = {nom: list(valeur.shape) for nom, valeur in etat.items()}
        rapport.extend([f'### `{chemin.relative_to(racine).as_posix()}`', '',
            'Format : checkpoint PyTorch. Contenu : poids du modèle et paramètres de prétraitement.', '',
            f'Clés : `{list(objet)}`.',
            f'Variables : `{objet.get("feature_names")}` ; longueur de séquence : `{objet.get("sequence_length")}`.',
            f'Nombre de tenseurs de poids : **{len(etat)}**. Premières dimensions :', '',
            '```json', json.dumps(dict(list(formes.items())[:6]), ensure_ascii=False, indent=2), '```', '',
            'Aucune matrice d’adjacence, distance, liste de capteurs physiques ou coordonnées dans ce checkpoint.', ''])
    elif chemin.suffix.lower() == '.xlsx':
        classeur = pd.ExcelFile(chemin)
        rapport.extend([f'### `{chemin.relative_to(racine).as_posix()}`', '', 'Format : Excel ; tableaux de benchmark.', ''])
        for feuille in classeur.sheet_names:
            tableau = pd.read_excel(classeur, sheet_name=feuille)
            rapport.extend([f'Feuille `{feuille}` : **{tableau.shape[0]} × {tableau.shape[1]}**.', '',
                '```csv', tableau.head(5).to_csv(index=False).rstrip(), '```', ''])

# Rechercher les noms et les références explicites à des fichiers spatiaux dans le code source.
motif = re.compile(r'adjacency|adj_mx|distance\.csv|sensor[_ -]?(?:ids?|locations?)|latitude|longitude|coordinates?|road_graph|edge_list|\.npz|\.npy|\.pkl|\.pickle', re.I)
noms_candidats = [str(p.relative_to(racine)) for p in fichiers if motif.search(p.name)]
references = []
for chemin in fichiers:
    if chemin.suffix.lower() != '.ipynb':
        continue
    document = json.loads(chemin.read_text(encoding='utf-8'))
    for numero, cellule in enumerate(document.get('cells', [])):
        for ligne in ''.join(cellule.get('source', [])).splitlines():
            if motif.search(ligne):
                references.append(f'{chemin.relative_to(racine)} cellule {numero} : {ligne}')
rapport.extend(['## Candidats spatiaux et provenance', '',
    f'Fichiers dont le nom indique un contenu spatial : **{len(noms_candidats)}**. {noms_candidats}',
    f'Références spatiales explicites dans les sources des notebooks : **{len(references)}**. {references}',
    'Le notebook 01 charge directement traffic_dataset.csv ; aucun code local de conversion d’un NPZ vers le CSV ne documente l’ordre original des capteurs.', '',
    'Les PNG sont des figures de benchmark, le Markdown est un tableau de scores et le script préexistant exécute la section PatchTST. Ils ne fournissent pas de graphe routier.', '',
    '## Conclusion : C — aucune donnée spatiale trouvée', '',
    'Dans les fichiers locaux audités, aucune adjacence 170 × 170, matrice de distance, liste d’arêtes, coordonnée ou liste d’IDs physiques n’a été trouvée. '
    'Il n’y a donc aucun fichier candidat spatial dont les dimensions ou premières valeurs puissent être présentées. '
    'Le CSV suffit pour décrire les séries temporelles, mais ne permet pas de reconstruire le graphe physique.', '',
    '## Éléments exacts à récupérer avant STGCN', '',
    '1. **Le graphe associé à cette version de PEMS08** : soit une matrice d’adjacence **170 × 170**, soit une liste d’arêtes avec `source_sensor_id`, `target_sensor_id` et, si disponibles, `distance` ou `weight`.',
    '2. **La liste ordonnée des 170 capteurs et leur mapping** : table `location,sensor_id`, ou preuve documentée que location correspond à la position du capteur dans le tenseur source et dans les lignes/colonnes du graphe. La seule égalité du nombre de capteurs ne suffit pas.',
    '3. **La provenance de la conversion** : le tenseur PEMS08 original et le code ou manifeste ayant produit traffic_dataset.csv, pour vérifier l’ordre des capteurs et les mesures correspondantes.',
    '4. **La signification du graphe** : arêtes dirigées ou non, unité et définition des distances, sens de circulation, convention des poids et traitement des capteurs sans voisin. Pour une matrice, documenter aussi l’ordre des lignes et colonnes et le sens de la valeur zéro.',
    '5. Si seul un fichier de coordonnées est fourni : récupérer les **IDs physiques, coordonnées et connectivité du réseau routier** (ou un graphe dérivé documenté). Les coordonnées seules ne prouvent pas la connexion routière entre deux capteurs.', '',
    'Ne créer aucune matrice d’adjacence avant d’avoir vérifié cette correspondance. Aucun fichier n’est téléchargé dans cet audit.', ''])

# Garantir que l’audit n’a modifié aucun fichier déjà présent.
assert all(empreinte(p) == valeur for p, valeur in avant.items())
rapport.extend(['## Vérification de conservation', '',
    'Empreintes SHA-256 identiques avant et après lecture pour tous les fichiers inventoriés : notebooks, checkpoints et résultats conservés. Aucun modèle exécuté.', ''])
dossier.mkdir(parents=True, exist_ok=True)
(dossier / 'spatial_audit.md').write_text('\n'.join(rapport), encoding='utf-8')
inventaire = [{'path': p.relative_to(racine).as_posix(), 'format': p.suffix.lower(),
    'size_bytes': p.stat().st_size, 'sha256': avant[p]} for p in fichiers]
pd.DataFrame(inventaire).to_csv(dossier / 'file_inventory.csv', index=False)
print(f'Fichiers audités : {len(fichiers)} ; capteurs : {len(identifiants)} ; shape CSV : {donnees.shape}')
print('Conclusion C : aucune donnée spatiale trouvée localement.')
print('Rapport :', dossier / 'spatial_audit.md')
print(donnees.head(5).to_string(index=False))
