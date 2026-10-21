"""Récupérer la paire PEMS08 des auteurs et remplacer les fichiers vides après validation."""
import argparse
import hashlib
import json
import os
from itertools import permutations
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Davidham3/ASTGCN-2019-mxnet'
STAGE = ROOT / 'data/source/pems08'
RTOL, ATOL = 1e-5, 1e-6

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def request(url):
    # Utiliser une connexion HTTPS et ne transmettre aucun secret.
    return urlopen(Request(url, headers={'User-Agent': 'GoFlow-PEMS08-validation'}), timeout=120)

def download():
    # Fixer la révision pour que les deux données et leur documentation restent cohérentes.
    with request(f'https://api.github.com/repos/{REPO}/commits/master') as response:
        commit = json.load(response)['sha']
    base = f'https://raw.githubusercontent.com/{REPO}/{commit}'
    with request(base + '/configurations/PEMS08.conf') as response:
        configuration = response.read().decode('utf-8')
    with request(base + '/lib/utils.py') as response:
        chargeur = response.read().decode('utf-8')
    assert 'adj_filename = data/PEMS08/distance.csv' in configuration
    assert 'graph_signal_matrix_filename = data/PEMS08/pems08.npz' in configuration
    assert 'num_of_vertices = 170' in configuration
    assert 'A[i, j] = 1' in chargeur and 'int(i[0]), int(i[1])' in chargeur
    dossier = STAGE / commit
    dossier.mkdir(parents=True, exist_ok=True)
    provenance = {
        'dataset': 'PEMS08', 'repository': REPO, 'commit': commit,
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration_url': base + '/configurations/PEMS08.conf',
        'loader_url': base + '/lib/utils.py',
        'configuration': configuration,
        'id_semantics': 'npz_index',
        'evidence': 'La configuration de cette révision associe ces deux fichiers à PEMS08 (170 sommets). '
                    'Le chargeur des auteurs utilise les IDs entiers des arêtes directement comme indices A[i,j], sans remapping.',
        'files': {}
    }
    for filename in ['distance.csv', 'pems08.npz']:
        url = base + '/data/PEMS08/' + filename
        path = dossier / filename
        temporary = path.with_suffix(path.suffix + '.part')
        with request(url) as response, temporary.open('wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
        assert temporary.stat().st_size > 0, 'Téléchargement vide.'
        os.replace(temporary, path)
        provenance['files'][filename] = {'url': url, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        print(filename, path.stat().st_size, 'octets', flush=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / 'download_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Révision source :', commit)

def validate_install():
    # Comparer avant de modifier les deux fichiers actuels.
    provenance = json.loads((STAGE / 'download_provenance.json').read_text(encoding='utf-8'))
    dossier = STAGE / provenance['commit']
    npz_path, distance_path = dossier / 'pems08.npz', dossier / 'distance.csv'
    for name in ['distance.csv', 'pems08.npz']:
        assert sha256(dossier / name) == provenance['files'][name]['sha256']
    with np.load(npz_path, allow_pickle=False) as archive:
        shapes = {k: list(archive[k].shape) for k in archive.files}
        print('Clés et dimensions NPZ :', shapes)
        compatibles = [(k, archive[k].copy()) for k in archive.files
                       if archive[k].ndim == 3 and sorted(archive[k].shape) == [3, 170, 17856]]
    assert len(compatibles) == 1, 'Tenseur unique PEMS08 non identifié.'
    key, array = compatibles[0]
    axes = tuple(array.shape.index(size) for size in (17856, 170, 3))
    array = array.transpose(axes)
    assert array.shape == (17856, 170, 3) and np.isfinite(array).all()
    csv_path = ROOT / 'data/raw/traffic_dataset.csv'
    frame = pd.read_csv(csv_path)
    features = ['flow', 'occupy', 'speed']
    assert len(frame) == 17856 * 170
    assert set(frame.location) == set(range(170))
    assert sorted(frame.timestep.unique()) == list(range(1, 17857))
    assert not frame.duplicated(['location', 'timestep']).any()
    expected = frame.sort_values(['timestep', 'location'])[features].to_numpy().reshape(17856, 170, 3)
    sensors = [0, 1, 10, 50, 100, 169]
    orders = [p for p in permutations(range(3)) if np.allclose(
        array[:, sensors, :][:, :, p], expected[:, sensors, :], rtol=RTOL, atol=ATOL)]
    assert len(orders) == 1, 'Ordre des variables ambigu ou séries incompatibles.'
    order = orders[0]
    rows = []
    for sensor in range(170):
        for col, feature in enumerate(features):
            actual, target = array[:, sensor, order[col]], expected[:, sensor, col]
            valid = bool(np.allclose(actual, target, rtol=RTOL, atol=ATOL))
            rows.append({'location': sensor, 'variable': feature, 'npz_channel': order[col],
                         'max_absolute_error': float(np.abs(actual - target).max()), 'matches': valid})
    comparisons = pd.DataFrame(rows)
    assert comparisons.matches.all(), 'Le mapping location/index NPZ ne correspond pas pour tous les capteurs.'
    # Unicité des séries : ne pas confondre des capteurs aux séries entièrement identiques.
    fingerprints = [hashlib.sha256(np.ascontiguousarray(array[:, i, :]).tobytes()).hexdigest() for i in range(170)]
    groupes = {}
    for sensor, fingerprint in enumerate(fingerprints):
        groupes.setdefault(fingerprint, []).append(sensor)
    doublons_capteurs = [group for group in groupes.values() if len(group) > 1]
    mapping_unique = not doublons_capteurs
    print('Groupes de capteurs indiscernables par les séries :', doublons_capteurs)
    print('Axes temps/capteurs/features :', axes, '| mapping variables :', dict(zip(features, order)))
    print(comparisons[comparisons.location.isin(sensors)].to_string(index=False))
    # Montrer les premières valeurs effectivement comparées.
    preview = []
    for sensor in sensors:
        for t in range(5):
            row = {'location': sensor, 'timestep': t + 1}
            row.update({f'npz_channel_{i}': array[t, sensor, i] for i in range(3)})
            row.update({f'csv_{feature}': expected[t, sensor, i] for i, feature in enumerate(features)})
            preview.append(row)
    edges = pd.read_csv(distance_path)
    print('Colonnes arêtes :', list(edges.columns), '| dimensions :', edges.shape)
    print(edges.head(10).to_string(index=False))
    assert list(edges.columns) in [['from', 'to', 'cost'], ['from', 'to', 'distance']], 'Schéma spatial inattendu.'
    numeric = edges.apply(pd.to_numeric, errors='raise')
    assert np.isfinite(numeric.to_numpy()).all() and len(numeric) > 0
    endpoints = numeric.iloc[:, :2]
    assert (endpoints % 1 == 0).all().all()
    ids = {int(value) for value in endpoints.to_numpy().ravel()}
    assert ids.issubset(set(range(170))), 'IDs hors de la plage documentée ; mapping supplémentaire nécessaire.'
    distances = numeric.iloc[:, 2]
    assert (distances >= 0).all()
    assert numeric.groupby(list(numeric.columns[:2]))[numeric.columns[2]].nunique().le(1).all()
    print('Arêtes :', len(edges), '| nœuds distincts :', len(ids), '| IDs min/max :', min(ids), max(ids))
    print('Distances :', distances.describe().to_dict())
    report = {
        'conclusion': ('A. Vrais fichiers PEMS08 récupérés et mapping validé.' if mapping_unique else
                       'B. Vrais fichiers PEMS08 récupérés mais mapping encore incertain pour les séries identiques.'),
        'npz_keys_shapes': shapes, 'npz_key': key, 'axes_time_sensor_feature': list(axes),
        'feature_to_npz_channel': dict(zip(features, order)), 'csv_sha256': sha256(csv_path),
        'rtol': RTOL, 'atol': ATOL, 'all_170_full_series_match': True,
        'unique_sensor_series': mapping_unique, 'unique_series_count': len(groupes),
        'duplicate_sensor_groups': doublons_capteurs,
        'uniquely_identifiable_sensor_count': sum(len(group) == 1 for group in groupes.values()),
        'max_absolute_error': float(comparisons.max_absolute_error.max()),
        'edge_columns': list(edges.columns), 'edge_count': len(edges), 'distinct_edge_nodes': len(ids),
        'id_min': min(ids), 'id_max': max(ids), 'nodes_absent_from_edges': sorted(set(range(170)) - ids),
        'distance_statistics': distances.describe().to_dict(),
    }
    report_dir = ROOT / 'results/spatial_audit'
    report_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(report_dir / 'pems08_series_comparison.csv', index=False)
    pd.DataFrame(preview).to_csv(report_dir / 'pems08_comparison_preview.csv', index=False)
    # Conserver les sources authentiques et les comparaisons même si le mapping reste ambigu.
    provenance.update({
        'source_url': f'https://github.com/{REPO}/tree/{provenance["commit"]}/data/PEMS08',
        'distance_sha256': sha256(distance_path), 'npz_sha256': sha256(npz_path),
        'validation': report
    })
    (STAGE / 'validated_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (report_dir / 'pems08_retrieval_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not mapping_unique:
        print('Fichiers vides non remplacés : preuve documentaire de l’export CSV nécessaire pour les groupes ambigus.')
        print(report['conclusion'])
        return
    # Ne remplacer que les fichiers vides autorisés, après toutes les assertions ci-dessus.
    targets = [ROOT / 'data/raw' / name for name in ['distance.csv', 'pems08.npz']]
    assert all(p.is_file() and p.stat().st_size == 0 for p in targets), 'Un fichier destination est devenu non vide ; ne pas l’écraser.'
    for target in targets:
        temporary = target.with_suffix(target.suffix + '.validated')
        temporary.write_bytes((dossier / target.name).read_bytes())
        assert sha256(temporary) == provenance['files'][target.name]['sha256']
    for target in targets:
        os.replace(target.with_suffix(target.suffix + '.validated'), target)
    provenance.update({
        'source_url': f'https://github.com/{REPO}/tree/{provenance["commit"]}/data/PEMS08',
        'distance_sha256': sha256(targets[0]), 'npz_sha256': sha256(targets[1]),
        'validation': report
    })
    (ROOT / 'data/raw/pems08_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (report_dir / 'pems08_retrieval_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report['conclusion'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--validate-install', action='store_true')
    args = parser.parse_args()
    assert args.download != args.validate_install, 'Choisir une seule étape.'
    download() if args.download else validate_install()
