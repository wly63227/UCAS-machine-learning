import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),site=path.join(root,'site');
function all(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?all(path.join(dir,e.name)):[path.join(dir,e.name)]);}
const pages=all(site).filter(f=>f.endsWith('.html')),errors=[];let links=0,images=0;
for(const file of pages){
 const html=fs.readFileSync(file,'utf8'),ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);
 if(new Set(ids).size!==ids.length)errors.push(`Duplicate id: ${file}`);
 if(/MATHPLACEHOLDER|katex-error|\{#ml-/i.test(html))errors.push(`Unrendered math or anchor: ${file}`);
 for(const [,kind,url] of html.matchAll(/\b(href|src)="([^"]+)"/g)){
  if(/^(https?:|mailto:|data:|javascript:)/.test(url))continue;
  const [name,anchor]=url.split('#'),target=path.resolve(path.dirname(file),decodeURIComponent(name||path.basename(file)));
  if(!target.startsWith(site+path.sep)){errors.push(`Outside site: ${url}`);continue;}
  if(!fs.existsSync(target)){errors.push(`Missing ${path.relative(site,file)} -> ${url}`);continue;}
  if(anchor&&target.endsWith('.html')){const text=fs.readFileSync(target,'utf8');if(!text.includes(`id="${decodeURIComponent(anchor)}"`))errors.push(`Missing anchor ${url}`);}
  links++;if(kind==='src'&&url.endsWith('.svg'))images++;
 }
}
const knowledge=JSON.parse(fs.readFileSync(path.join(root,'sources/knowledge.json'),'utf8'));
if(new Set(knowledge.map(x=>x.id)).size!==knowledge.length)errors.push('Duplicate knowledge ID');
for(const item of knowledge){const [name,id]=item.target.split('#'),text=fs.readFileSync(path.join(root,'docs',name),'utf8');if(!text.includes(`{#${id}}`))errors.push(`Missing knowledge target ${item.id}`);}
const baselinePath=path.join(root,'content-baseline.json');
const chapterFiles=all(path.join(root,'docs/chapters')).filter(f=>f.endsWith('.md'));
const current=Object.fromEntries(chapterFiles.map(f=>{const t=fs.readFileSync(f,'utf8');return [path.basename(f),{chars:t.length,ids:[...t.matchAll(/\{#(ml-[^}]+)\}/g)].map(x=>x[1]),displayFormulas:(t.match(/\$\$/g)||[]).length/2}]}));
if(fs.existsSync(baselinePath)){
 const baseline=JSON.parse(fs.readFileSync(baselinePath,'utf8'));
 for(const [name,old] of Object.entries(baseline)){
  const now=current[name];if(!now){errors.push(`Deleted chapter: ${name}`);continue;}
  if(now.chars<old.chars*.9)errors.push(`Chapter shrank more than 10%: ${name}`);
  if(now.displayFormulas<old.displayFormulas)errors.push(`Display formula removed: ${name}`);
  for(const id of old.ids)if(!now.ids.includes(id))errors.push(`Stable anchor removed: ${id}`);
 }
}else if(process.argv.includes('--create-baseline'))fs.writeFileSync(baselinePath,JSON.stringify(current,null,2));
else errors.push('Missing content baseline; create once with --create-baseline after review');
const result={pages:pages.length,internalResourceAndAnchorChecks:links,svgReferences:images,knowledgeTargets:knowledge.length,errors};
fs.writeFileSync(path.join(root,'check-report.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));if(errors.length)process.exitCode=1;
