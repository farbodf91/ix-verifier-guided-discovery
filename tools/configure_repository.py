"""Set repository and Colab links once, before publishing or when changing revision.

Usage: python tools/configure_repository.py https://github.com/USER/REPO [--ref main]
"""
from pathlib import Path
import argparse,json,re
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('url');p.add_argument('--ref',default='main')
a=p.parse_args();url=a.url.rstrip('/').removesuffix('.git')
if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',url):
    p.error('Use a GitHub repository URL: https://github.com/USER/REPO')
if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.ref):p.error('Use a branch/tag without slashes, or a commit SHA')
config=ROOT/'repository.json'
previous=json.loads(config.read_text()) if config.exists() else {'url':'https://github.com/farbodf91/ix-verifier-guided-discovery','ref':'main'}
old,new=previous['url'],url
old_path=old.removeprefix('https://github.com/');new_path=url.removeprefix('https://github.com/')
old_ref=previous['ref']
files=list(ROOT.glob('*.ipynb'))+list(ROOT.glob('*.md'))
for path in files:
    text=path.read_text()
    text=text.replace('https://colab.research.google.com/github/'+old_path+'/blob/'+old_ref+'/',
                      'https://colab.research.google.com/github/'+new_path+'/blob/'+a.ref+'/')
    text=text.replace(old+'/blob/'+old_ref+'/',new+'/blob/'+a.ref+'/')
    text=text.replace(old,new)
    # Source cells are parsed to change exactly the setup ref, not unrelated text.
    if path.suffix=='.ipynb':
        notebook=json.loads(text)
        for cell in notebook['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='code' and 'REPOSITORY_URL =' in source:
                source=re.sub(r'^GIT_REF = .*$',f'GIT_REF = "{a.ref}"',source,flags=re.M)
                cell['source']=source.splitlines(True)
        text=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    path.write_text(text)
config.write_text(json.dumps({'url':url,'ref':a.ref},indent=2)+'\n')
# Refresh reading previews if nbconvert is installed; notebooks are configured either way.
try:
    import nbformat
    from nbconvert import HTMLExporter
    for path in ROOT.glob('*.ipynb'):
        body,_=HTMLExporter().from_notebook_node(nbformat.read(path,as_version=4))
        path.with_suffix('.html').write_text(body)
except ImportError:
    print('Notebook links configured. Install nbconvert to regenerate HTML previews.')
print('Configured:',url,'revision:',a.ref)
print('Main Colab link: https://colab.research.google.com/github/'+new_path+'/blob/'+a.ref+'/01_IX_symbolic_discovery_tutorial.ipynb')
