"""Deterministic scientific SVGs; no random sampling or external data."""
from pathlib import Path
import math
OUT=Path(__file__).resolve().parents[1]/'docs/assets'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,title,content):
    (OUT/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" role="img" aria-label="{title}"><rect width="760" height="360" fill="#fff"/><style>text{{font:14px sans-serif;fill:#23383b}}.axis{{stroke:#6b7779;stroke-width:1;fill:none}}.main{{stroke:#087f8c;fill:none;stroke-width:3}}.alt{{stroke:#a34b32;fill:none;stroke-width:3;stroke-dasharray:8 5}}</style><text x="40" y="30" font-weight="bold">{title}</text>{content}</svg>',encoding='utf-8')
def plot(name,title,functions,xmin,xmax,ymax,labels,lines=()):
    def xx(x):return 65+(x-xmin)/(xmax-xmin)*640
    def yy(y):return 295-y/ymax*225
    s='<path class="axis" d="M65 65V295H705"/>'
    for i in range(5):
        x=xmin+(xmax-xmin)*i/4;s+=f'<text x="{xx(x)-12}" y="318">{x:g}</text>'
    for y in (0,ymax/2,ymax):s+=f'<text x="14" y="{yy(y)+5}">{y:.2g}</text>'
    for j,fun in enumerate(functions):
        points=' '.join(f'{xx(x):.2f},{yy(fun(x)):.2f}' for x in (xmin+(xmax-xmin)*i/320 for i in range(321)))
        s+=f'<polyline class="{"main" if j==0 else "alt"}" points="{points}"/><text x="{100+j*300}" y="350">{"实线" if j==0 else "虚线"}：{labels[j]}</text>'
    for x,label in lines:s+=f'<path d="M{xx(x)} 70V295" stroke="#555" stroke-dasharray="3 4"/><text x="{xx(x)+5}" y="85">{label}</text>'
    save(name,title,s)
normal=lambda mu:lambda x:math.exp(-.5*(x-mu)**2)/math.sqrt(2*math.pi)
plot('gaussian.svg','共同方差高斯：横轴 x，纵轴为条件概率密度',[normal(0),normal(2)],-3,5,.5,['类 1：均值 0','类 2：均值 2'],[(1,'等先验'),(1+math.log(4)/2,'先验 0.8 / 0.2')])
def beta(a,b):
    c=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b))
    return lambda x:c*x**(a-1)*(1-x)**(b-1)
plot('beta.svg','Beta 共轭更新：横轴 θ，纵轴为参数密度',[beta(5,5),beta(20,15)],0,1,5.5,['先验 Beta(5,5)','后验 Beta(20,15)'])
plot('sigmoid.svg','Sigmoid：横轴线性得分 z，纵轴正类概率',[lambda z:1/(1+math.exp(-z))],-6,6,1,['σ(z)'],[(0,'p = 0.5')])
s='<path class="axis" d="M65 245H700"/>'
for x in [0,2,8,10]:
    px=80+x*58;s+=f'<text x="{px-5}" y="275">{x}</text>'
    if x<5:s+=f'<circle cx="{px}" cy="245" r="7" fill="#087f8c"/>'
    else:s+=f'<rect x="{px-6}" y="239" width="12" height="12" fill="#a34b32"/>'
for old,new in [(0,1),(8,9)]:
    a,b=80+old*58,80+new*58;s+=f'<path d="M{a} 130H{b}" stroke="#111" stroke-width="2"/><path d="M{b-7} 125L{b} 130L{b-7} 135" fill="none" stroke="#111"/><text x="{a-10}" y="110">中心 {old} → {new}</text><path d="M{b} 145V230" stroke="#555" stroke-dasharray="4 4"/>'
s+='<text x="80" y="325">分配不变时，均值更新让平方误差从 8 降到 4。</text>'
save('kmeans.svg','一维 K-means：中心向各簇均值移动',s)
s='<path class="axis" d="M65 180H700M380 320V60"/>'
for radius in [1,2]:s+=f'<ellipse cx="380" cy="180" rx="{110*radius}" ry="{30*radius}" fill="none" stroke="#8da8ab" transform="rotate(-25 380 180)"/>'
s+='<path class="main" d="M165 280L595 80"/><path class="alt" d="M340 95L420 265"/><text x="535" y="65">第一主方向（方差 4）</text><text x="430" y="280">第二主方向（方差 0.25）</text><text x="80" y="345">示意模型：协方差特征值为 4 与 0.25；椭圆轴比为标准差比 4:1。</text>'
# Exact ellipse ratio must be 4:1, matching sqrt(4/.25).
s=s.replace('ry="30"','ry="27.5"').replace('ry="60"','ry="55.0"')
save('pca.svg','协方差椭圆的主方向（两轴使用相同显示尺度）',s)
s=''
for row,label in enumerate(['链：A → B → C','分叉：A ← B → C','碰撞：A → B ← C']):
    y=90+row*95;s+=f'<text x="30" y="{y-25}">{label}</text>'
    for j,t in enumerate('ABC'):
        x=320+j*150;s+=f'<circle cx="{x}" cy="{y}" r="22" fill="#e6f2f0" stroke="#087f8c"/><text x="{x-5}" y="{y+5}">{t}</text>'
    for j in range(2):
        a,b=342+j*150,448+j*150;right=(row==0 or (row==1 and j==1) or (row==2 and j==0));tip=b if right else a
        s+=f'<path d="M{a} {y}H{b}" stroke="#333"/><path d="M{tip+(-7 if right else 7)} {y-5}L{tip} {y}L{tip+(-7 if right else 7)} {y+5}" fill="none" stroke="#333"/>'
s+='<text x="30" y="340">前两种：给定 B 阻断路径；碰撞：给定 B 或其后代会打开路径。</text>'
save('d-separation.svg','三种局部结构：箭头相遇方式决定阻断规则',s)

# Course-aligned overview figures. These redraw the mathematical structure of
# the slides instead of publishing slide screenshots.
s='''
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs>
<g text-anchor="middle">
<rect x="35" y="105" width="145" height="70" rx="10" fill="#e6f2f0" stroke="#087f8c"/><text x="108" y="135" font-weight="bold">原始观测与标签</text><text x="108" y="158">训练数据 (x, y)</text>
<rect x="225" y="105" width="145" height="70" rx="10" fill="#f4efe3" stroke="#a88442"/><text x="298" y="135" font-weight="bold">特征表达</text><text x="298" y="158">x ∈ R^d</text>
<rect x="415" y="105" width="145" height="70" rx="10" fill="#e6f2f0" stroke="#087f8c"/><text x="488" y="135" font-weight="bold">学习算法</text><text x="488" y="158">拟合模型参数</text>
<rect x="605" y="105" width="120" height="70" rx="10" fill="#f4efe3" stroke="#a88442"/><text x="665" y="135" font-weight="bold">训练模型</text><text x="665" y="158">f(x)</text>
</g>
<g stroke="#087f8c" stroke-width="2" marker-end="url(#arrow)"><path d="M180 140H220"/><path d="M370 140H410"/><path d="M560 140H600"/></g>
<path d="M488 190V250H665" fill="none" stroke="#087f8c" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="365" y="230" width="123" height="48" rx="8" fill="#fff" stroke="#6b7779"/><text x="427" y="260" text-anchor="middle">新样本 x*</text>
<rect x="605" y="230" width="120" height="48" rx="8" fill="#fff" stroke="#6b7779"/><text x="665" y="260" text-anchor="middle">预测 y*</text>
<text x="40" y="320">训练阶段用已知标签学习参数；应用阶段只输入新样本，标签未知。</text>'''
save('learning-process.svg','课件主线：从训练数据到新样本预测',s)

s='''
<g transform="translate(20 55)"><rect width="225" height="250" rx="10" fill="#fbfbf8" stroke="#d6ddd8"/><text x="112" y="28" text-anchor="middle" font-weight="bold">分类：输出离散类别</text><path d="M35 210V55M35 210H205" class="axis"/><path d="M65 195L175 70" class="main"/>
<g fill="#087f8c">''' + ''.join(f'<circle cx="{x}" cy="{y}" r="6"/>' for x,y in [(62,173),(78,160),(88,184),(95,150)]) + '''</g><g fill="#a34b32">''' + ''.join(f'<circle cx="{x}" cy="{y}" r="6"/>' for x,y in [(145,105),(160,92),(174,116),(184,82)]) + '''</g><text x="112" y="235" text-anchor="middle">例如：年轻 / 老年</text></g>
<g transform="translate(268 55)"><rect width="225" height="250" rx="10" fill="#fbfbf8" stroke="#d6ddd8"/><text x="112" y="28" text-anchor="middle" font-weight="bold">回归：输出连续数值</text><path d="M35 210V55M35 210H205" class="axis"/><path d="M48 192L195 70" class="main"/><g fill="#a34b32">''' + ''.join(f'<circle cx="{x}" cy="{y}" r="5"/>' for x,y in [(52,187),(74,174),(93,158),(118,139),(146,112),(180,83)]) + '''</g><text x="112" y="235" text-anchor="middle">例如：面积 → 房价</text></g>
<g transform="translate(516 55)"><rect width="225" height="250" rx="10" fill="#fbfbf8" stroke="#d6ddd8"/><text x="112" y="28" text-anchor="middle" font-weight="bold">聚类：没有给定标签</text><g fill="#087f8c">''' + ''.join(f'<circle cx="{x}" cy="{y}" r="7"/>' for x,y in [(65,90),(82,73),(92,103)]) + '''</g><g fill="#a34b32">''' + ''.join(f'<rect x="{x-7}" y="{y-7}" width="14" height="14"/>' for x,y in [(155,92),(177,75),(178,108)]) + '''</g><ellipse cx="80" cy="88" rx="42" ry="38" fill="none" stroke="#087f8c"/><ellipse cx="170" cy="92" rx="42" ry="38" fill="none" stroke="#a34b32"/><text x="112" y="178" text-anchor="middle">分组取决于特征与准则</text><text x="112" y="235" text-anchor="middle">颜色、形状、大小</text></g>'''
save('intro-tasks.svg','分类、回归与聚类的输出差异',s)

def bar(x,y,w,h,color,label,value):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#edf0e9"/><rect x="{x}" y="{y}" width="{w*value:.1f}" height="{h}" fill="{color}"/><text x="{x}" y="{y-8}">{label} = {value:.3f}</text>'
s='<defs><marker id="arrb" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs>'
s+='<text x="95" y="72" font-weight="bold">看到晚霞之前：先验</text>'+bar(55,105,250,35,'#087f8c','P(晴)',265/365)+bar(55,190,250,35,'#a34b32','P(非晴)',100/365)
s+='<path d="M330 165H430" stroke="#087f8c" stroke-width="3" marker-end="url(#arrb)"/><text x="380" y="135" text-anchor="middle">证据：晚霞</text><text x="380" y="190" text-anchor="middle">似然 0.7 / 0.1</text>'
s+='<text x="500" y="72" font-weight="bold">看到晚霞之后：后验</text>'+bar(455,105,250,35,'#087f8c','P(晴 | 晚霞)',.948849)+bar(455,190,250,35,'#a34b32','P(非晴 | 晚霞)',.051151)
s+='<text x="55" y="295">似然比 0.7 / 0.1 = 7，使证据明显偏向“晴”；后验仍同时受先验影响。</text>'
save('bayes-update.svg','天气例：证据如何把先验更新为后验',s)

s='''<g transform="translate(30 60)"><rect width="330" height="245" rx="10" fill="#fbfbf8" stroke="#d6ddd8"/><text x="165" y="25" text-anchor="middle" font-weight="bold">线性可分：存在一条直线分开两类</text><path d="M35 215V45M35 215H300" class="axis"/><path d="M135 210L220 48" class="main"/>'''+''.join(f'<circle cx="{x}" cy="{y}" r="7" fill="#087f8c"/>' for x,y in [(70,170),(90,150),(102,185),(120,145)])+''.join(f'<rect x="{x-7}" y="{y-7}" width="14" height="14" fill="#a34b32"/>' for x,y in [(205,130),(230,105),(245,145),(267,90)])+'''</g>
<g transform="translate(400 60)"><rect width="330" height="245" rx="10" fill="#fbfbf8" stroke="#d6ddd8"/><text x="165" y="25" text-anchor="middle" font-weight="bold">线性不可分：XOR</text><path d="M35 215V45M35 215H300" class="axis"/><circle cx="90" cy="165" r="8" fill="#087f8c"/><circle cx="250" cy="75" r="8" fill="#087f8c"/><rect x="82" y="67" width="16" height="16" fill="#a34b32"/><rect x="242" y="157" width="16" height="16" fill="#a34b32"/><text x="165" y="235" text-anchor="middle">任意直线都会让至少一个点落错侧</text></g>'''
save('linear-separability.svg','线性可分与线性不可分的几何区别',s)

s='''<defs><marker id="arrm" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs>
<text x="55" y="70" font-weight="bold">训练阶段：4 类的一对一分类器</text><text x="55" y="100">A/B　A/C　A/D　B/C　B/D　C/D</text><text x="55" y="130">共 4×3/2 = 6 个模型</text><path d="M310 115H390" stroke="#087f8c" stroke-width="2" marker-end="url(#arrm)"/>
<text x="430" y="55" font-weight="bold">测试阶段：一个样本走淘汰树</text>
<g text-anchor="middle"><rect x="420" y="85" width="110" height="42" rx="7" fill="#e6f2f0" stroke="#087f8c"/><text x="475" y="111">A vs B</text><rect x="610" y="85" width="110" height="42" rx="7" fill="#e6f2f0" stroke="#087f8c"/><text x="665" y="111">C vs D</text><rect x="515" y="195" width="110" height="48" rx="7" fill="#f4efe3" stroke="#a88442"/><text x="570" y="216">胜者 1</text><text x="570" y="234">vs 胜者 2</text><rect x="525" y="285" width="90" height="40" rx="7" fill="#fff" stroke="#6b7779"/><text x="570" y="310">最终类别</text></g>
<g stroke="#087f8c" stroke-width="2" marker-end="url(#arrm)"><path d="M475 127L545 190"/><path d="M665 127L595 190"/><path d="M570 243V280"/></g><text x="55" y="225">一条测试路径只做 3 次比较 = K-1。</text><text x="55" y="260">“训练 6 个模型”与“测试比较 3 次”不是同一个计数。</text>'''
save('multiclass-tree.svg','四分类示例：两两训练与二叉淘汰测试',s)

s='''<path d="M65 285H700M90 315V55" class="axis"/><path d="M100 70L500 300" class="alt"/><path d="M210 55L385 305" class="main"/>
<g fill="#087f8c">'''+''.join(f'<circle cx="{x}" cy="{y}" r="6"/>' for x,y in [(135,110),(155,95),(170,130),(190,112),(205,145)])+'''</g><g fill="#6b7779">'''+''.join(f'<circle cx="{x}" cy="{y}" r="6"/>' for x,y in [(400,220),(425,205),(448,235),(465,215),(490,245)])+'''</g>
<circle cx="342" cy="268" r="9" fill="#a34b32"/><circle cx="210" cy="105" r="9" fill="#a34b32"/><text x="520" y="95">虚线：初始边界</text><text x="520" y="120">x₁ + x₂ - 0.5 = 0</text><text x="520" y="165">实线：批量修正后</text><text x="520" y="190">1.42x₁ + 0.51x₂ - 0.5 = 0</text><text x="520" y="235">红点：本轮两个错分样本</text><text x="95" y="340">课件例 1：η = 0.7；更新后全部训练点被正确分类。</text>'''
save('perceptron-update.svg','感知机课堂例 1：错分点推动决策边界旋转',s)

s='''<defs><marker id="arrf" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs><path d="M70 275H690M90 305V55" class="axis"/>
<g fill="#087f8c">'''+''.join(f'<circle cx="{x}" cy="{y}" r="7"/>' for x,y in [(175,210),(190,175),(210,230),(230,150),(245,195)])+'''</g><g fill="#a34b32">'''+''.join(f'<rect x="{x-7}" y="{y-7}" width="14" height="14"/>' for x,y in [(430,165),(455,115),(475,205),(500,145),(525,185)])+'''</g><path d="M120 255L600 95" stroke="#087f8c" stroke-width="3" marker-end="url(#arrf)"/><text x="530" y="80">投影方向 w</text><path d="M145 310H585" stroke="#6b7779"/><g stroke="#6b7779">'''+''.join(f'<path d="M{x} 302V318"/>' for x in [220,240,260,440,460,480])+'''</g><text x="350" y="340" text-anchor="middle">投影后：同类紧凑、两类均值相距远</text>'''
save('fisher-projection.svg','Fisher 线性鉴别：寻找类间远、类内紧的投影',s)

print('Generated 13 deterministic SVG figures.')
