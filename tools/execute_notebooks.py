"""Execute the teaching notebooks and create self-contained reading previews."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('names', nargs='*', default=['00_Getting_started','01_IX_symbolic_discovery_tutorial','02_Cylinder_flow_exploration'])
args = parser.parse_args()
for name in args.names:
    path = root / (name+'.ipynb')
    notebook = nbformat.read(path, as_version=4)
    print('Executing', name, flush=True)
    NotebookClient(notebook, timeout=900, kernel_name='python3', resources={'metadata':{'path':str(root)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    exporter = HTMLExporter()
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(notebook)
    (root / (name+'.html')).write_text(body)
    print('Saved', name, flush=True)
