import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {marked} from 'marked';
import katex from 'katex';
import {execFileSync} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const docs=path.join(root,'docs'),out=path.join(root,'site');
let sourceCommit=process.env.GITHUB_SHA??null;
if(!sourceCommit){try{sourceCommit=execFileSync('git',['-c',`safe.directory=${root}`,'rev-parse','HEAD'],{cwd:root,encoding:'utf8',stdio:['ignore','pipe','ignore']}).trim();}catch{}}
fs.mkdirSync(out,{recursive:true});
function files(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(path.join(dir,e.name)):[path.join(dir,e.name)]);}
const sources=files(docs).filter(f=>f.endsWith('.md')).sort();
const esc=s=>s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const slug=s=>s.toLowerCase().replace(/<[^>]*>/g,'').replace(/[^\p{L}\p{N}\- ]/gu,'').trim().replace(/\s+/g,'-');
let formulaCount=0;
const entries=sources.map(f=>({file:f,rel:path.relative(docs,f).replaceAll('\\','/'),text:fs.readFileSync(f,'utf8')}));
for(const e of entries){e.title=e.text.match(/^# (.+)$/m)?.[1]??e.rel;e.url=e.rel.replace(/\.md$/,'.html');}
const ordered=[...entries.filter(e=>e.rel==='index.md'),...entries.filter(e=>e.rel==='guide.md'),...entries.filter(e=>e.rel.startsWith('chapters/')),...entries.filter(e=>!['index.md','guide.md'].includes(e.rel)&&!e.rel.startsWith('chapters/'))];
const index=[];
for(let i=0;i<ordered.length;i++){
 const e=ordered[i],depth=e.rel.split('/').length-1,prefix='../'.repeat(depth),math=[];
 let source=e.text.replace(/\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g,(all,block,inline)=>{
  const html=katex.renderToString((block??inline).trim(),{displayMode:block!==undefined,throwOnError:true,output:'htmlAndMathml',strict:'warn'});formulaCount++;math.push(html);return `MATHPLACEHOLDER${math.length-1}END`;
 });
 const headings=[];
 const renderer=new marked.Renderer();
 renderer.heading=function({tokens,depth}){let text=this.parser.parseInline(tokens),id=text.match(/\s*\{#([^}]+)\}$/)?.[1];text=text.replace(/\s*\{#[^}]+\}$/,'');id??=slug(text);headings.push({depth,text,id});return `<h${depth} id="${esc(id)}">${text}<a class="anchor" href="#${esc(id)}" aria-label="本节链接">#</a></h${depth}>`;};
 let body=marked.parse(source,{renderer,gfm:true}).replace(/MATHPLACEHOLDER(\d+)END/g,(_,n)=>math[Number(n)]).replace(/href="([^"#?]*?)\.md(#[^"]*)?"/g,(_,name,anchor)=>`href="${name}.html${anchor??''}"`);
 const nav=ordered.map(x=>`<a ${x===e?'aria-current="page"':''} href="${prefix+x.url}" data-search="${esc(x.title)}">${esc(x.title)}</a>`).join('');
 const toc=headings.filter(h=>h.depth===2).map(h=>`<a href="#${esc(h.id)}">${h.text}</a>`).join('');
 const prev=ordered[i-1],next=ordered[i+1];
 const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>${esc(e.title)} · 机器学习教材</title><link rel="stylesheet" href="${prefix}assets/katex/katex.min.css"><link rel="stylesheet" href="${prefix}assets/style.css"></head><body><a class="skip" href="#content">跳到正文</a><header><a href="${prefix}index.html" class="brand">ML <span>机器学习 · 学习笔记</span></a><button id="theme" aria-label="切换深浅色">深浅色</button></header><div class="layout"><aside class="sidebar"><label for="search">搜索全书</label><input id="search" type="search" placeholder="概念、公式或关键词" autocomplete="off"><div id="results" role="status" aria-live="polite"></div><nav aria-label="章节">${nav}</nav><small>2026 主课件 · 往年延伸<br>教材初稿 / 2026-10-20</small></aside><main id="content"><div class="eyebrow">UCAS · MACHINE LEARNING</div><article>${body}</article><footer>${prev?`<a href="${prefix+prev.url}">← ${esc(prev.title)}</a>`:'<span></span>'}${next?`<a href="${prefix+next.url}">${esc(next.title)} →</a>`:''}</footer></main><aside class="toc"><strong>本页内容</strong>${toc}<a href="#content">↑ 返回顶部</a></aside></div><script src="${prefix}assets/search-data.js"></script><script src="${prefix}assets/app.js" defer></script><noscript><p class="noscript">正文、公式和目录均可阅读；全文搜索与主题按钮需要 JavaScript。</p></noscript></body></html>`;
 const target=path.join(out,e.url);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,html);
 index.push({title:e.title,url:e.url,text:e.text.replace(/\$\$[\s\S]*?\$\$/g,' ').replace(/\$[^$\n]*\$/g,' ').replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/\{#[^}]+\}/g,'').replace(/[#*`>|]/g,' ')});
}
fs.mkdirSync(path.join(out,'assets'),{recursive:true});
for(const name of ['style.css','app.js'])fs.copyFileSync(path.join(root,'scripts',name),path.join(out,'assets',name));
fs.cpSync(path.join(docs,'assets'),path.join(out,'assets'),{recursive:true});
fs.cpSync(path.join(root,'node_modules/katex/dist'),path.join(out,'assets/katex'),{recursive:true});
fs.writeFileSync(path.join(out,'assets/search-data.js'),'window.ML_SEARCH='+JSON.stringify(index)+';');
fs.writeFileSync(path.join(out,'version.json'),JSON.stringify({commit:sourceCommit,builtAt:new Date().toISOString(),contentStatus:'draft'},null,2));
fs.writeFileSync(path.join(root,'build-report.json'),JSON.stringify({pages:ordered.length,formulas:formulaCount,mathEngine:'KaTeX local HTML + MathML',date:'2026-10-20'},null,2));
console.log(`Built ${ordered.length} pages; rendered ${formulaCount} formulas locally.`);
