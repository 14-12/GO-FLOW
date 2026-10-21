"""Récupérer le binaire CPU officiel et le GGUF Qwen officiel, sans installer de package."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[1]
diagnostics = root / "results/llm_agent/diagnostics/transformers_int8_first_attempt"
if not diagnostics.exists():
    diagnostics.mkdir(parents=True)
    for path in (root / "results/llm_agent").iterdir():
        if path.is_file():
            shutil.copy2(path, diagnostics / path.name)
opener = urllib.request.build_opener()
tag = "b11596"
tool_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-win-cpu-x64.zip"
model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
revision = "9217f5db79a29953eb74d5343926648285ec7e67"
filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
model_url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
folder = root / "models/qwen2.5-0.5b-instruct-gguf"
tool_folder = root / "tools/llama.cpp" / tag
folder.mkdir(parents=True, exist_ok=True)
tool_folder.mkdir(parents=True, exist_ok=True)
manifest = {"model_id": model_id, "revision": revision, "llama_cpp_release": tag, "files": {}}
for path, url in [(folder / filename, model_url), (tool_folder / "release.zip", tool_url)]:
    if not path.exists():
        print("Téléchargement :", path.name, flush=True)
        temporary = path.with_suffix(path.suffix + ".download")
        with opener.open(url, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["files"][str(path.relative_to(root))] = {"url": url, "bytes": path.stat().st_size, "sha256": checksum}
    print(path.name, path.stat().st_size, flush=True)
with zipfile.ZipFile(tool_folder / "release.zip") as archive:
    for member in archive.infolist():
        target = (tool_folder / member.filename).resolve()
        assert target.is_relative_to(tool_folder.resolve()), "Chemin d'archive invalide"
    archive.extractall(tool_folder)
manifest["llama_server_path"] = str(next(tool_folder.rglob("llama-server.exe")).relative_to(root))
(folder / "download_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Backend CPU local préparé.")
