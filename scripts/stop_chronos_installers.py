"""Arrêter les vérifications pip bloquées après les installations terminées dans notre venv."""
from pathlib import Path
import psutil

folder = str(Path(__file__).resolve().parents[1] / '.venv-chronos').lower()
for process in psutil.process_iter(['pid', 'cmdline']):
    try:
        args = process.info['cmdline'] or []
        command = ' '.join(args).lower()
        if '-m pip install --no-deps' not in command:
            continue
        if not any(name in command for name in ['chronos-forecasting==2.3.2', 'accelerate>=1.1']):
            continue
        parent = process.parent()
        parent_command = ' '.join(parent.cmdline()).lower() if parent else ''
        # Limiter l’arrêt aux processus appartenant à l’environnement créé pour ce travail.
        if folder in command or folder in parent_command:
            process.terminate()
            print('Vérification post-installation pip arrêtée :', process.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
