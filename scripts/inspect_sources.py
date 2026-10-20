"""Run from any directory. Raw sources are read-only; IDs are stable by filename."""
from pathlib import Path
from pypdf import PdfReader
import hashlib,json,re
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
OUT=BASE/'sources';(OUT/'text').mkdir(parents=True,exist_ok=True)
old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8')) if (OUT/'manifest.json').exists() else []
known={x['file']:x['id'] for x in old};next_id=max([int(x['id'][1:]) for x in old]+[0])+1
items=[]
for p in sorted(ROOT.rglob('*.pdf')):
    if BASE in p.parents:continue
    rel=p.relative_to(ROOT).as_posix();sid=known.get(rel)
    if sid is None:sid=f'S{next_id:02d}';next_id+=1
    role='往年课件' if '往年课件/' in rel else ('本年度课件' if 'Chapter' in p.name else '参考书')
    pages=[{'page':i,'text':q.extract_text() or ''} for i,q in enumerate(PdfReader(p).pages,1)]
    (OUT/'text'/f'{sid}.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'text'/f'{sid}.txt').write_text('\n\n'.join(f'=== PDF物理页 {q["page"]} ===\n{q["text"]}' for q in pages),encoding='utf-8')
    items.append(dict(id=sid,file=rel,role=role,pages=len(pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),low_text_pages=[q['page'] for q in pages if len(q['text'].strip())<60]))
(OUT/'manifest.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
diff=[{'id':x['id'],'file':x['file'],'change':'新增' if x['file'] not in known else '修改'} for x in items if x['file'] not in known or next(y for y in old if y['file']==x['file'])['sha256']!=x['sha256']]
missing=[x for x in old if not (ROOT/x['file']).exists()]
(OUT/'changes.json').write_text(json.dumps({'changes':diff,'missing':missing},ensure_ascii=False,indent=2),encoding='utf-8')
# All other original files, including board photo and archive, are inventoried separately.
extras=[]
for p in ROOT.rglob('*'):
    if p.is_file() and BASE not in p.parents and p.suffix.lower()!='.pdf' and '.git' not in p.parts:
        extras.append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'other-files.json').write_text(json.dumps(extras,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Indexed {len(items)} PDFs; {sum(x["pages"] for x in items)} physical pages; changes {len(diff)}; missing {len(missing)}.')
