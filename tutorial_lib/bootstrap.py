"""Install the supported Python 3.12 environment before scientific imports."""
from importlib import metadata
from pathlib import Path
import subprocess
import shutil
import sys


def install(root):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            'This tutorial uses Python 3.12. In Colab choose Runtime > Change runtime type '
            '> Runtime Version > 2026.07, reconnect, then run this cell again. '
            'For local use, create a Python 3.12 environment.'
        )
    requirements = Path(root) / 'requirements.txt'
    needed = []
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            installed = metadata.version(name).split('+')[0]
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            needed.append(name)
    if needed:
        loaded = [name for name in ('numpy', 'pandas', 'scipy', 'sympy', 'torch') if name in sys.modules]
        print('Installing the tutorial environment. This can take several minutes.')
        # CPU sessions do not need the much larger CUDA wheel and its libraries.
        if 'torch' in needed and shutil.which('nvidia-smi') is None:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', 'torch==2.4.1',
                 '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True,
            )
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', '-r', str(requirements)],
            text=True, capture_output=True,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError('Installation failed; see the package error above.')
        if loaded:
            raise RuntimeError('Packages were updated after scientific imports. Restart the session, then run setup again.')
    print('Environment ready — Python 3.12; tutorial data and figures are available.')
