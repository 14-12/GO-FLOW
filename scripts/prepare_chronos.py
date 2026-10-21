"""Vérifier Chronos dans l’environnement isolé et récupérer seulement le modèle Amazon tiny."""
from pathlib import Path
import os
import sys
import json
import hashlib
import importlib.metadata as metadata
import inspect
from packaging.requirements import Requirement

root = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(root / 'models/.huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
output = root / 'results/model_benchmark/chronos_environment.json'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

info = {'python': sys.version, 'executable': sys.executable, 'setup_success': False}
try:
    import torch
    import psutil
    info.update({'torch': torch.__version__, 'cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(), 'ram_total_bytes': psutil.virtual_memory().total,
        'ram_available_bytes': psutil.virtual_memory().available})
    versions = {}
    # Vérifier chaque contrainte active du paquet, sans laisser pip modifier les versions globales.
    for raw in metadata.requires('chronos-forecasting'):
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        version = metadata.version(requirement.name)
        versions[requirement.name] = version
        print(requirement.name, version, str(requirement.specifier), flush=True)
        assert version in requirement.specifier, 'Dépendance incompatible : ' + raw
    versions['chronos-forecasting'] = metadata.version('chronos-forecasting')
    info['package_versions'] = versions
    from chronos import ChronosBoltPipeline
    print('Import ChronosBoltPipeline : OK', inspect.signature(ChronosBoltPipeline.predict), flush=True)
    from huggingface_hub import HfApi, snapshot_download
    model_id = 'amazon/chronos-bolt-tiny'
    remote = HfApi(token=False).model_info(model_id, files_metadata=True)
    revision = remote.sha
    assert any(item.rfilename == 'model.safetensors' for item in remote.siblings)
    folder = root / 'models/chronos-bolt-tiny'
    snapshot_download(repo_id=model_id, revision=revision, token=False,
        allow_patterns=['config.json', 'model.safetensors', 'README.md'], local_dir=folder)
    weights = folder / 'model.safetensors'
    file_info = next(item for item in remote.siblings if item.rfilename == 'model.safetensors')
    expected_hash = file_info.lfs.sha256 if file_info.lfs is not None else None
    weight_hash = digest(weights)
    if expected_hash is not None:
        assert weight_hash == expected_hash, 'Empreinte des poids différente du dépôt.'
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = ChronosBoltPipeline.from_pretrained(str(folder), device_map=device,
        torch_dtype=torch.float32, trust_remote_code=False, local_files_only=True)
    pipeline.model.eval()
    for parameter in pipeline.model.parameters():
        parameter.requires_grad_(False)
    info.update({'model_id': model_id, 'model_revision': revision, 'model_folder': str(folder),
        'model_parameters': sum(p.numel() for p in pipeline.model.parameters()), 'device': device,
        'model_weights_sha256': weight_hash, 'pretrained': True, 'fine_tuned_on_pems08': False,
        'context_length_max': pipeline.model_context_length, 'quantiles': pipeline.quantiles})
    print('Modèle chargé :', info['model_parameters'], 'paramètres |', device, flush=True)
    # Tester uniquement des mesures historiques de validation, sans métrique ni cible test.
    import pandas as pd
    history = pd.read_csv(root / 'data/raw/traffic_dataset.csv', usecols=['location', 'timestep', 'occupy'])
    values = history.loc[(history.location == 0) & history.timestep.between(12453, 12500)].sort_values('timestep').occupy.to_numpy()
    assert len(values) == 48
    context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        forecasts = pipeline.predict(context, prediction_length=4)
    assert forecasts.shape == (1, len(pipeline.quantiles), 4)
    assert torch.isfinite(forecasts).all().item()
    info['setup_success'] = True
    print('Test technique sur historique de validation : shape', tuple(forecasts.shape), flush=True)
except Exception as error:
    info.update({'error_type': type(error).__name__, 'error': str(error)})
    output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    raise
output.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
