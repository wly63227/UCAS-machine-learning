import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {marked} from 'marked';
import katex from 'katex';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docs = path.join(root, 'docs');
const tempDir = path.join(root, 'tmp', 'pdfs');
const outputDir = path.join(root, 'output', 'pdf');
const htmlPath = path.join(tempDir, '机器学习笔记-打印版.html');
const outputPath = path.join(outputDir, 'UCAS机器学习笔记-初稿.pdf');
const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const chapters = Array.from({length: 10}, (_, i) =>
  `chapters/${String(i + 1).padStart(2, '0')}-${[
    'intro', 'bayes', 'linear', 'nonlinear', 'regression', 'clustering',
    'features', 'information', 'graphical', 'neural'
  ][i]}.md`
);
const ordered = ['guide.md', ...chapters, 'exercises.md', 'knowledge.md', 'sources.md', 'coverage.md', 'audit.md', 'maintenance.md'];

fs.mkdirSync(tempDir, {recursive: true});
fs.mkdirSync(outputDir, {recursive: true});

const esc = s => s.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const slug = s => s.toLowerCase().replace(/<[^>]*>/g, '').replace(/[^\p{L}\p{N}\- ]/gu, '').trim().replace(/\s+/g, '-');
const docId = rel => `doc-${rel.replace(/\.md$/, '').replaceAll('/', '-')}`;
const records = ordered.map(rel => {
  const text = fs.readFileSync(path.join(docs, rel), 'utf8');
  return {rel, text, title: text.match(/^# (.+)$/m)?.[1] ?? rel, id: docId(rel)};
});

function resolveMarkdownLink(currentRel, href) {
  if (/^(?:https?:|mailto:|#)/.test(href)) return href;
  const [filePart, anchor = ''] = href.split('#', 2);
  if (!filePart.endsWith('.md')) return href;
  const target = path.posix.normalize(path.posix.join(path.posix.dirname(currentRel), filePart));
  return `#${docId(target)}${anchor ? `-${anchor}` : ''}`;
}

let formulaCount = 0;
let content = '';
for (const record of records) {
  const math = [];
  const source = record.text.replace(/\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g, (all, block, inline) => {
    const rendered = katex.renderToString((block ?? inline).trim(), {
      displayMode: block !== undefined,
      throwOnError: true,
      output: 'htmlAndMathml',
      strict: 'warn'
    });
    formulaCount += 1;
    math.push(rendered);
    return `MATHPLACEHOLDER${math.length - 1}END`;
  });
  const renderer = new marked.Renderer();
  renderer.heading = function ({tokens, depth}) {
    let label = this.parser.parseInline(tokens);
    let anchor = label.match(/\s*\{#([^}]+)\}$/)?.[1];
    label = label.replace(/\s*\{#[^}]+\}$/, '');
    anchor ??= slug(label);
    const id = depth === 1 ? record.id : `${record.id}-${anchor}`;
    return `<h${depth} id="${esc(id)}">${label}</h${depth}>`;
  };
  renderer.link = function ({href, title, tokens}) {
    const label = this.parser.parseInline(tokens);
    const resolved = resolveMarkdownLink(record.rel, href);
    return `<a href="${esc(resolved)}"${title ? ` title="${esc(title)}"` : ''}>${label}</a>`;
  };
  let body = marked.parse(source, {renderer, gfm: true})
    .replace(/MATHPLACEHOLDER(\d+)END/g, (_, n) => math[Number(n)])
    .replace(/src="\.\.\/assets\//g, 'src="assets/')
    .replace(/src="assets\//g, 'src="../../docs/assets/');
  content += `<section class="document ${record.rel.startsWith('chapters/') ? 'chapter' : 'appendix'}">${body}</section>`;
}

const toc = records.map((record, i) =>
  `<li><a href="#${record.id}"><span>${i < 11 ? String(i + 1).padStart(2, '0') : 'A' + (i - 10)}</span>${esc(record.title)}</a></li>`
).join('');
const katexCss = fs.readFileSync(path.join(root, 'node_modules', 'katex', 'dist', 'katex.min.css'), 'utf8')
  .replace(/url\(fonts\//g, 'url(../../node_modules/katex/dist/fonts/');
const css = `
@page { size: A4; margin: 20mm 18mm 19mm 18mm; @bottom-center { content: counter(page); color: #64748b; font-size: 9pt; } }
@page:first { margin: 0; @bottom-center { content: none; } }
* { box-sizing: border-box; }
html { font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; color: #172033; font-size: 10.5pt; line-height: 1.72; }
body { margin: 0; }
.cover { height: 297mm; padding: 38mm 28mm 25mm; display: flex; flex-direction: column; page-break-after: always; background: linear-gradient(145deg,#f7f9ff 0%,#eef7f7 62%,#fff 100%); }
.cover .kicker { color: #246b73; letter-spacing: .18em; font-weight: 700; font-size: 10pt; }
.cover h1 { margin: 34mm 0 5mm; max-width: 130mm; font-family: Georgia,"SimSun",serif; font-size: 34pt; line-height: 1.2; letter-spacing: .04em; color: #123149; }
.cover .subtitle { font-size: 14pt; color: #48657a; }
.cover .meta { margin-top: auto; padding-top: 7mm; border-top: 1px solid #b7ccd2; color: #536977; }
.cover .status { display: inline-block; margin-top: 4mm; padding: 1.5mm 4mm; border: 1px solid #d69f35; border-radius: 999px; color: #8b5a05; background: #fff9e9; font-size: 9pt; }
.toc-page { page-break-after: always; }
.toc-page h1 { font-size: 24pt; }
.book-toc { list-style: none; padding: 0; margin: 10mm 0 0; column-count: 2; column-gap: 12mm; }
.book-toc li { break-inside: avoid; margin: 0 0 3.5mm; }
.book-toc a { display: flex; gap: 3mm; color: #25394d; text-decoration: none; border-bottom: 1px dotted #bac7d1; }
.book-toc span { color: #2e7a80; font-variant-numeric: tabular-nums; font-weight: 700; }
.document { page-break-before: always; }
.document > h1:first-child { margin-top: 0; padding-bottom: 4mm; border-bottom: 2px solid #2e7a80; font-size: 24pt; color: #123149; }
h2 { margin: 9mm 0 3mm; padding-left: 3mm; border-left: 3px solid #d69f35; font-size: 16pt; line-height: 1.35; color: #173e58; break-after: avoid; }
h3 { margin: 6mm 0 2mm; font-size: 12.5pt; color: #24556d; break-after: avoid; }
h4 { break-after: avoid; }
p { margin: 0 0 3.2mm; text-align: justify; orphans: 3; widows: 3; }
a { color: #176b78; text-decoration: none; }
ul, ol { padding-left: 6mm; margin: 2mm 0 4mm; }
li { margin-bottom: 1.2mm; }
blockquote { margin: 4mm 0; padding: 3mm 5mm; border-left: 3px solid #6aa2a7; background: #f1f7f7; color: #314d58; break-inside: avoid; }
pre { padding: 4mm; border-radius: 2mm; background: #f3f5f7; white-space: pre-wrap; break-inside: avoid; font-size: 8.5pt; line-height: 1.5; }
code { font-family: Consolas,monospace; background: #f2f4f6; padding: .2mm .7mm; border-radius: 1mm; font-size: .9em; }
pre code { background: none; padding: 0; }
table { width: 100%; border-collapse: collapse; margin: 4mm 0 6mm; font-size: 9pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #cbd6dc; padding: 2mm 2.5mm; vertical-align: top; }
th { background: #eaf2f3; color: #173e58; text-align: left; }
img { display: block; max-width: 92%; max-height: 165mm; margin: 5mm auto; break-inside: avoid; }
.katex-display { margin: 4mm 0; overflow: visible; break-inside: avoid; font-size: .98em; }
.katex { color: #111827; }
hr { border: 0; border-top: 1px solid #cbd6dc; margin: 8mm 0; }
.print-note { margin-top: 10mm; padding: 4mm; background: #fff8e6; border: 1px solid #e5c77c; color: #715311; }
`;
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>UCAS机器学习笔记（初稿）</title><style>${katexCss}\n${css}</style></head><body>
<section class="cover"><div class="kicker">UCAS · MACHINE LEARNING</div><h1>机器学习课程笔记</h1><div class="subtitle">从概率基础到神经网络的十章自学教材</div><div class="meta">整理日期：2026-10-30<br>在线版：https://wly63227.github.io/UCAS-machine-learning/<br><span class="status">教材初稿 · 全资料逐页视觉验收尚未完成</span></div></section>
<section class="toc-page"><h1>目录</h1><ol class="book-toc">${toc}</ol><p class="print-note">正文以 2026 主课件为优先来源，后续章节包含往年资料延伸。来源边界与验收状态见附录。</p></section>
${content}</body></html>`;
fs.writeFileSync(htmlPath, html);
if (!fs.existsSync(edge)) throw new Error(`Microsoft Edge not found: ${edge}`);
execFileSync(edge, [
  '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
  `--print-to-pdf=${outputPath}`, `file:///${htmlPath.replaceAll('\\', '/')}`
], {stdio: 'inherit', timeout: 180000});
if (!fs.existsSync(outputPath)) throw new Error('PDF was not created.');
console.log(`Created ${outputPath} with ${records.length} sections and ${formulaCount} formulas.`);
