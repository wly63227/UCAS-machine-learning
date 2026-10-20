"""Generate inventories and honest page mappings, never infer reviewed status."""
from pathlib import Path
import json,re,csv,hashlib
B=Path(__file__).resolve().parents[1];D=B/'docs';S=B/'sources'
m=json.loads((S/'manifest.json').read_text(encoding='utf-8'))
chapters={p.name[:2]:p for p in sorted((D/'chapters').glob('*.md'))}
# page-start, page-end, chapter, anchor, treatment
ranges={
'S01':[(1,7,'01','ml-01-01','课程信息；考试日期冲突'),(8,32,'01','ml-01-01','背景；产品列表外部事实待核实'),(33,41,'01','ml-01-01','学习定义与背景'),(42,54,'01','ml-01-02','任务与建模流程'),(55,64,'01','ml-01-03','课程大纲与参考资料')],
'S02':[(1,3,'02','ml-02-01','封面与目录'),(4,7,'02','ml-02-01','天气例'),(8,10,'02','ml-02-01','概率基础'),(11,14,'02','ml-02-02','判别与参考文献')],
'S03':[(1,5,'02','ml-02-02','回顾；半朴素仅提及'),(6,17,'02','ml-02-03','朴素贝叶斯；西瓜表待逐格复核'),(18,18,'02','ml-02-03','休息页'),(19,30,'02','ml-02-04','高斯判别与二次例'),(31,32,'02','ml-02-04','休息与目录'),(33,41,'02','ml-02-06','参数后验与Beta共轭'),(42,47,'02','ml-02-05','ML与高斯估计'),(48,51,'02','ml-02-06','MAP'),(52,53,'02','ml-02-06','小结与参考文献')],
'S05':[(1,5,'03','ml-03-01','封面与章节导言'),(6,28,'03','ml-03-01','向量空间与超平面'),(29,37,'03','ml-03-02','距离、散度、数值例'),(38,41,'03','ml-03-03','评价；检验仅提及'),(42,49,'03','ml-03-07','线性可分、多类策略与参考文献')],
'S06':[(1,9,'03','ml-03-04','导言、增广和梯度下降'),(10,27,'03','ml-03-04','感知机准则、算法与例子'),(28,44,'03','ml-03-05','Fisher判别与休息页'),(45,60,'03','ml-03-06','logistic与softmax'),(61,62,'03','ml-03-07','小结与参考文献')],
'S23':[(1,7,'04','ml-04-01','导言与非线性方法'),(8,42,'04','ml-04-01','树、划分、剪枝；原图例待复核'),(43,59,'04','ml-04-02','近邻、快速检索、压缩与拒绝'),(60,60,'04','ml-04-02','参考文献')],
'S24':[(1,13,'04','ml-04-03','组合、Bagging与随机森林'),(14,26,'04','ml-04-04','AdaBoost；原多轮表待复核'),(27,41,'04','ml-04-05','最大间隔、对偶、KKT'),(42,49,'04','ml-04-06','核与软间隔'),(50,51,'04','ml-04-06','小结与参考文献')],
'S10':[(1,17,'05','ml-05-01','问题与评价'),(18,20,'05','ml-05-02','最小二乘'),(21,32,'05','ml-05-03','高斯ML与梯度例'),(33,38,'05','ml-05-04','MAP与正则'),(39,44,'05','ml-05-05','基函数与多项式'),(45,50,'05','ml-05-05','偏差方差与参考文献')],
'S25':[(1,18,'06','ml-06-01','定义、相似性与评价'),(19,30,'06','ml-06-02','序贯与增强'),(31,48,'06','ml-06-02','层次与聚类数'),(49,66,'06','ml-06-03','K均值迭代图'),(67,71,'06','ml-06-04','单点转移'),(72,76,'06','ml-06-05','原数据例待复核；参考文献')],
'S11':[(1,12,'07','ml-07-01','动机与定义'),(13,25,'07','ml-07-02','子集生成'),(26,37,'07','ml-07-02','可分性；部分概率距离未完整转录'),(38,49,'07','ml-07-02','三种框架与Relief；原表待复核')],
'S12':[(1,12,'07','ml-07-01','线性变换、投影与分解'),(13,38,'07','ml-07-03','PCA重构与方差推导'),(39,43,'07','ml-07-04','SVD、LDA'),(44,56,'07','ml-07-05','KPCA、LLE、NMF与小结')],
'S13':[(1,7,'08','ml-08-01','概述'),(8,13,'08','ml-08-01','熵与微分熵'),(14,18,'08','ml-08-03','最大熵与高斯'),(19,27,'08','ml-08-02','条件熵、互信息、KL'),(28,34,'08','ml-08-04','条件最大熵'),(35,40,'08','ml-08-03','五状态例与文献')],
'S14':[(1,13,'09','ml-09-01','DAG与草坪例；原概率表待核对'),(14,27,'09','ml-09-02','d分离与马尔可夫毯'),(28,40,'09','ml-09-03','MRF、RBM与图像去噪'),(41,59,'09','ml-09-04','学习、推断与变量消去'),(60,69,'09','ml-09-04','信念传播与参考文献')],
'S15':[(1,13,'09','ml-09-04','前向采样'),(14,28,'09','ml-09-04','Gibbs采样'),(29,47,'09','ml-09-05','HMM定义与三问题'),(48,60,'09','ml-09-05','前向后向与期望'),(61,70,'09','ml-09-05','Viterbi、学习、信封例待核对'),(71,80,'09','ml-09-06','CRF与小结')],
'S16':[(1,23,'10','ml-10-01','历史与应用；事实未逐项核实'),(24,34,'10','ml-10-01','神经元、损失与优化'),(35,58,'10','ml-10-01','网络族概述；扩展模型未完整推导'),(59,60,'10','ml-10-01','重点与参考文献')],
'S17':[(1,10,'10','ml-10-01','多层网络'),(11,42,'10','ml-10-02','BP与梯度推导'),(43,48,'10','ml-10-01','激活、XOR、逼近；定理细节待核对'),(49,72,'10','ml-10-03','CNN、池化与参考文献')],
'S18':[(1,9,'10','ml-10-04','递归结构与能力；定理细节待核对'),(10,20,'10','ml-10-04','RNN与BPTT'),(21,21,'10','ml-10-04','明确课堂作业：截断深度h；待视觉转录'),(22,28,'10','ml-10-04','连续训练、任务结构与文献')],
'S19':[(1,20,'10','ml-10-02','梯度、ReLU与视觉应用图'),(21,41,'10','ml-10-05','语言表示、注意力、预训练'),(42,44,'10','ml-10-05','历史平台列表'),(45,51,'10','ml-10-05','语言模型与RLHF；参数列表未核实'),(52,62,'10','ml-10-06','GAN与应用'),(63,69,'10','ml-10-06','扩散模型'),(70,74,'10','ml-10-06','模型安全与参考文献')]
}
duplicates={'S08':'S01','S09':'S02','S20':'S03','S21':'S05','S22':'S06'}
visual={'S01':{5,42,43,44,45,48,53,54,57},'S03':set(range(33,52))|{20,21,24,27,30},'S06':{16,18,19,23,24,27,42,43,46,54,55,58,59,60}}
lines=['# 资料清单与版本说明','','## 版本取舍','','本年度 S01、S02、S03、S05、S06 是前三章主来源。往年同章文件作为比较版本，第四至十章使用往年资料，不能替代尚未收到的本年度课件。页码一律为 PDF 物理页码。哈希完整值见下表与本地 sources/manifest.json。','','教材与仓库的发布状态见项目 README；不根据构建成功推断线上发布成功。参考项目的两个链接曾尝试访问，但读取失败，因此未声称检查过其源码。工程依据任务文件中的明确要求独立建立。','','## PDF 清单','','| ID | 文件 | 角色 | 页数 | SHA-256 |','| --- | --- | --- | --- | --- |']
for x in m:lines.append(f'| {x["id"]} | {x["file"]} | {x["role"]} | {x["pages"]} | `{x["sha256"]}` |')
lines+=['','## 非 PDF 资料','','已有贝叶斯 Markdown 笔记保留原样，仅作比对，不作为未经核实的外部引用依据；原笔记的 Obsidian 返回链接不复制到本站。课堂板书照片已单独查看，其共同协方差高斯边界推导对应第二章 2.4，并补充了可逆性前提。ZIP 是往年课件压缩包，解压目录已存在，未重复解压或删除。三个中文参考书文本层不完整，不能据其页码声称完成书本核对；Bishop 有文本层，但本版未完整研读全书。','','参考书仅为可用资源，不作为本版所有编者解释的逐条外部出处。原课件与参考书不复制进网站发布目录。']
(D/'sources.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
cover=['# 来源覆盖与视觉核对记录','','**有映射不等于已经逐项验收。** 下表为主题页段到正文的初步对应；逐物理页状态另存 `sources/page-coverage.csv`。文字已批量提取，但不能据此认定所有图片公式、原始案例或扫描页均已阅读完整。','','“已视觉抽查”只用于实际打开查看的原页。其他页是待视觉复核；休息、目录与参考页不要求编写独立知识点。四本参考书均不是逐页教学覆盖对象。','','| 来源 | PDF 页段 | 正文对应 | 处理说明 |','| --- | --- | --- | --- |']
rows=[]
for x in m:
    sid=x['id']
    if sid in ranges:
        for lo,hi,ch,anchor,desc in ranges[sid]:
            target=f'chapters/{chapters[ch].name}#{anchor}';cover.append(f'| {sid} | {lo}–{hi} | [{anchor}]({target}) | {desc}；内容验收待完成 |')
        for page in range(1,x['pages']+1):
            matches=[r for r in ranges[sid] if r[0]<=page<=r[1]]
            if len(matches)!=1:raise ValueError((sid,page,matches))
            lo,hi,ch,anchor,desc=matches[0]
            rows.append({'source':sid,'page':page,'target':f'chapters/{chapters[ch].name}#{anchor}','text':'extracted','visual':'spot-checked' if page in visual.get(sid,set()) else 'pending','content':'draft-mapped-not-accepted','note':desc})
    elif sid in duplicates:
        cover.append(f'| {sid} | 1–{x["pages"]} | 同章主来源 {duplicates[sid]} | 往年比对版；不宣称逐页完全一致 |')
        for page in range(1,x['pages']+1):rows.append({'source':sid,'page':page,'target':'','text':'extracted','visual':'pending','content':'comparison-version','note':f'对应主版本 {duplicates[sid]}；需差异复核'})
    else:
        for page in range(1,x['pages']+1):rows.append({'source':sid,'page':page,'target':'','text':'extracted','visual':'pending','content':'reference-only','note':'参考书，不计课件正文覆盖'})
(D/'coverage.md').write_text('\n'.join(cover)+'\n',encoding='utf-8')
with (S/'page-coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
knowledge=['# 核心知识点索引','','编号是维护用的稳定锚点；本表收录当前正文核心节，不代表全部课件细节已穷尽。重要程度是编者的学习组织判断，不是教师考试重点。来源“对应”仍需结合覆盖表的验收状态。','','| 编号 | 知识点 | 章节 | 重要程度 | 本版深度 | 来源页段 |','| --- | --- | --- | --- | --- | --- |']
k=[]
for ch,p in chapters.items():
    for title,anchor in re.findall(r'^## (.*?) \{#([^}]+)\}$',p.read_text(encoding='utf-8'),re.M):
        refs=[f'{sid} {lo}–{hi}' for sid,rr in ranges.items() for lo,hi,c,a,desc in rr if a==anchor]
        depth='推导与例子' if any(x in title for x in ['似然','MAP','Fisher','Logistic','PCA','K-means','BP','SVM','AdaBoost','最大熵','HMM']) else '概念与方法'
        knowledge.append(f'| [{anchor}](chapters/{p.name}#{anchor}) | {title} | {ch} | 核心／支撑 | {depth} | {"；".join(refs)} |')
        k.append({'id':anchor,'title':title,'target':f'chapters/{p.name}#{anchor}','sources':refs,'depth':depth})
(D/'knowledge.md').write_text('\n'.join(knowledge)+'\n',encoding='utf-8')
(S/'knowledge.json').write_text(json.dumps(k,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'pdf_files':len(m),'all_pdf_pages':sum(x['pages'] for x in m),'current_slides':sum(x['role']=='本年度课件' for x in m),'current_pages':sum(x['pages'] for x in m if x['role']=='本年度课件'),'historical_slides':sum(x['role']=='往年课件' for x in m),'lecture_pages':sum(x['pages'] for x in m if x['role']!='参考书'),'core_knowledge_sections':len(k),'visually_spot_checked_pages':sum(r['visual']=='spot-checked' for r in rows),'accepted_complete_pages':0}
(S/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
