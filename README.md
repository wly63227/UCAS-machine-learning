# 机器学习课程教材初稿

从 [学习入口](docs/index.md) 开始；可直接打开生成的 `site/index.html`。当前正文覆盖十章主线：前三章用 2026 主课件，后续用往年资料。**全资料逐页视觉验收尚未完成**，详见 [建设与验收记录](docs/audit.md)。

## 环境与命令

Python 3.10+，Node.js 20+，pnpm。以下命令在本目录执行：

```powershell
python -m pip install -r requirements.txt
pnpm install --frozen-lockfile
python scripts/figures.py
python scripts/indexes.py
python scripts/verify_math.py
node scripts/build.mjs
node scripts/check.mjs
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

浏览器访问 `http://127.0.0.1:8765/`。也可直接用浏览器打开 `site/index.html`，正文、公式和搜索无需远端请求。结束预览按 Ctrl+C。

资料变更时先运行 `python scripts/inspect_sources.py`，该步骤需要原始课程 PDF。普通网站构建只需仓库内 Markdown 和索引数据，不需要发布原始课件或参考书。

## 文件结构

- `docs/`：学习指南、十章正文、索引、练习、验收与维护说明。
- `docs/assets/`：6 张确定性教学 SVG。
- `scripts/`：盘点、索引、图表、数学检查、静态构建与链接／基线检查。
- `sources/`：资料清单、逐页状态 CSV、核心知识 JSON；提取文本与审查截图不纳入 Git。
- `site/`：生成网站，可静态托管，不手工编辑。
- `deploy/`：GitHub Pages 工作流模板。
- `进度.md`：下一次恢复工作的位置。

遵循仓库根目录的 `AGENTS.md`。项目仓库为 [wly63227/UCAS-machine-learning](https://github.com/wly63227/UCAS-machine-learning)，按用户确认保持公开。上传内容包括教材源码与索引，不含原始 PDF、参考书、依赖、缓存和截图。GitHub Actions 只运行检查并保存离线网站构建产物，不启用 Pages。

网站不用 VitePress：采用规模较小的静态生成器，实现同一 Markdown 来源、本地数学、全文搜索、前后导航、深浅色与移动阅读，减少构建依赖。后续如需迁移，稳定锚点和 Markdown 可直接保留。
