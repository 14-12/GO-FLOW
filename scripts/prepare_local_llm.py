"""Auditer puis télécharger le seul petit modèle instruct, dans le projet."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
opener = urllib.request.build_opener()
with opener.open(f"https://huggingface.co/api/models/{MODEL_ID}", timeout=60) as response:
    info = json.load(response)
revision = info["sha"]
folder = ROOT / "models/qwen2.5-0.5b-instruct"
folder.mkdir(parents=True, exist_ok=True)
allowed = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
           "vocab.json", "merges.txt", "model.safetensors", "README.md", "LICENSE"}
files = [row["rfilename"] for row in info["siblings"] if row["rfilename"] in allowed]
assert "model.safetensors" in files and "tokenizer_config.json" in files
manifest = {"model_id": MODEL_ID, "revision": revision, "files": {}}
for name in files:
    path = folder / name
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        print("Téléchargement :", name, flush=True)
        with opener.open(url, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["files"][name] = {"url": url, "bytes": path.stat().st_size, "sha256": checksum}
    print(name, path.stat().st_size, flush=True)
(folder / "download_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Modèle téléchargé :", folder, "révision :", revision, flush=True)
