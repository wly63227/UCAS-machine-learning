"""Bundle readable site and editable sources, excluding original course PDFs."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
B=Path(__file__).resolve().parents[1]
output=B/'机器学习笔记初稿.zip'
folders=['docs','scripts','site','deploy','qa']
files=['README.md','进度.md','package.json','pnpm-lock.yaml','requirements.txt','content-baseline.json','math-report.json','build-report.json','check-report.json']
files += ['sources/'+name for name in ['manifest.json','other-files.json','knowledge.json','page-coverage.csv','summary.json','changes.json']]
with ZipFile(output,'w',ZIP_DEFLATED) as z:
    for folder in folders:
        for p in sorted((B/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(B))
    for file in files:z.write(B/file,file)
    z.write(B/'AGENTS.md','AGENTS.md')
with ZipFile(output) as z:
    if z.testzip() is not None:raise RuntimeError('Archive integrity failure')
    assert 'site/index.html' in z.namelist()
    print(f'{output.name}: {len(z.namelist())} entries, {output.stat().st_size} bytes, CRC verified')
