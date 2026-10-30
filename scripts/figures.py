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

# Restored examples from historical slides (S10--S25).  The figures are
# structural redraws and computed plots, not screenshots of the slides.
s='''<defs><marker id="arrdt" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#6b7779"/></marker></defs>
<g text-anchor="middle"><rect x="315" y="52" width="130" height="38" rx="7" fill="#e6f2f0" stroke="#087f8c"/><text x="380" y="77">纹理？</text>
<rect x="125" y="135" width="110" height="38" rx="7" fill="#fff" stroke="#087f8c"/><text x="180" y="160">根蒂？</text>
<rect x="325" y="135" width="110" height="38" rx="7" fill="#fff" stroke="#087f8c"/><text x="380" y="160">触感？</text>
<ellipse cx="595" cy="154" rx="48" ry="24" fill="#f4e6e1" stroke="#a34b32"/><text x="595" y="159">坏瓜</text>
<ellipse cx="75" cy="245" rx="46" ry="24" fill="#e6f2f0" stroke="#087f8c"/><text x="75" y="250">好瓜</text>
<rect x="150" y="226" width="105" height="38" rx="7" fill="#fff" stroke="#087f8c"/><text x="202" y="251">色泽？</text>
<ellipse cx="285" cy="245" rx="46" ry="24" fill="#f4e6e1" stroke="#a34b32"/><text x="285" y="250">坏瓜</text>
<ellipse cx="355" cy="245" rx="46" ry="24" fill="#e6f2f0" stroke="#087f8c"/><text x="355" y="250">好瓜</text>
<ellipse cx="455" cy="245" rx="46" ry="24" fill="#f4e6e1" stroke="#a34b32"/><text x="455" y="250">坏瓜</text></g>
<g stroke="#6b7779" fill="none"><path d="M335 90L205 135"/><path d="M380 90V135"/><path d="M425 90L570 135"/><path d="M150 173L92 222"/><path d="M180 173L195 226"/><path d="M218 173L276 222"/><path d="M355 173V221"/><path d="M405 173L445 221"/></g>
<g font-size="12"><text x="255" y="112">清晰</text><text x="390" y="115">稍糊</text><text x="515" y="112">模糊</text><text x="102" y="205">蜷缩</text><text x="185" y="202">稍蜷</text><text x="245" y="202">硬挺</text><text x="335" y="205">硬滑</text><text x="420" y="205">软粘</text><text x="95" y="300">“稍蜷”分支继续按色泽与触感划分；图中省略三级叶的重复标签。</text></g>'''
save('decision-tree-watermelon.svg','西瓜决策树：根节点按纹理划分',s)

s='''<defs><marker id="arra" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs>
<g text-anchor="middle"><rect x="40" y="75" width="160" height="70" rx="8" fill="#e6f2f0" stroke="#087f8c"/><text x="120" y="101" font-weight="bold">第 1 轮</text><text x="120" y="125">p₁=0.300, α₁=0.424</text>
<rect x="300" y="75" width="160" height="70" rx="8" fill="#f4efe3" stroke="#a88442"/><text x="380" y="101" font-weight="bold">第 2 轮</text><text x="380" y="125">p₂=0.214, α₂=0.649</text>
<rect x="560" y="75" width="160" height="70" rx="8" fill="#f4e6e1" stroke="#a34b32"/><text x="640" y="101" font-weight="bold">第 3 轮</text><text x="640" y="125">p₃=0.1365, α₃=0.922</text></g>
<g stroke="#087f8c" stroke-width="2" marker-end="url(#arra)"><path d="M200 110H292"/><path d="M460 110H552"/></g>
<text x="65" y="205">错分样本权重：0.1 → 0.2333 → 0.3664</text><path d="M65 230H690" stroke="#6b7779"/><rect x="65" y="222" width="125" height="16" fill="#087f8c"/><rect x="190" y="222" width="167" height="16" fill="#a88442"/><rect x="357" y="222" width="233" height="16" fill="#a34b32"/>
<text x="65" y="285">最终：H(x)=sign(0.42h₁(x)+0.65h₂(x)+0.92h₃(x))</text><text x="65" y="320">轮次越晚并非必然权重越大；这里因为加权错误率持续下降，α 才持续增大。</text>'''
save('adaboost-rounds.svg','AdaBoost 原例：三轮错误率、分类器权重与样本权重',s)

house_x=[35,40,45,65,74,80,120,140,230,300,400,500,600]
house_y=[150,170,190,200,224,245,320,400,640,780,900,1100,1300]
def hx(x): return 70+x/600*610
def hy(y): return 300-y/1400*230
s='<path class="axis" d="M70 60V300H700"/><text x="650" y="325">面积 x (m²)</text><text x="10" y="68">房价 y</text>'
s+=''.join(f'<circle cx="{hx(x):.1f}" cy="{hy(y):.1f}" r="5" fill="#087f8c"/>' for x,y in zip(house_x,house_y))
s+=f'<path d="M{hx(0):.1f} {hy(93.362):.1f}L{hx(600):.1f} {hy(2.056026*600+93.362):.1f}" stroke="#a34b32" stroke-width="3"/>'
s+=f'<path d="M{hx(90):.1f} 300V{hy(278.404):.1f}H70" fill="none" stroke="#a88442" stroke-dasharray="5 4"/><circle cx="{hx(90):.1f}" cy="{hy(278.404):.1f}" r="7" fill="#a88442"/>'
s+='<text x="175" y="85">ŷ = 2.056x + 93.362</text><text x="175" y="110">x=90 → ŷ≈278.40 万元</text>'
save('regression-house.svg','房价课件原例：13 个样本的最小二乘直线',s)

melon=[(.697,.460),(.774,.376),(.634,.264),(.608,.318),(.556,.215),(.403,.237),(.481,.149),(.437,.211),(.666,.091),(.243,.267),(.245,.057),(.343,.099),(.639,.161),(.657,.198),(.360,.370),(.593,.042),(.719,.103),(.359,.188),(.339,.241),(.282,.257),(.748,.232),(.714,.346),(.483,.312),(.478,.437),(.525,.369),(.751,.489),(.532,.472),(.473,.376),(.725,.445),(.446,.459)]
c1={5,6,7,8,9,10,13,14,15,17,18,19,20,23}; c2={11,12,16}
def mx(x): return 70+(x-.2)/.6*610
def my(y): return 300-(y-.02)/.49*230
s='<path class="axis" d="M70 60V300H700"/><text x="650" y="325">密度</text><text x="10" y="68">含糖率</text>'
for i,(x,y) in enumerate(melon,1):
    color='#087f8c' if i in c1 else ('#a88442' if i in c2 else '#a34b32')
    s+=f'<circle cx="{mx(x):.1f}" cy="{my(y):.1f}" r="5" fill="{color}"/><text x="{mx(x)+6:.1f}" y="{my(y)-5:.1f}" font-size="9">{i}</text>'
for (x,y,label,color) in [(.403,.237,'μ₁','#087f8c'),(.343,.099,'μ₂','#a88442'),(.532,.472,'μ₃','#a34b32')]:
    s+=f'<path d="M{mx(x)-8:.1f} {my(y):.1f}H{mx(x)+8:.1f}M{mx(x):.1f} {my(y)-8:.1f}V{my(y)+8:.1f}" stroke="{color}" stroke-width="3"/><text x="{mx(x)+10:.1f}" y="{my(y)+16:.1f}">{label}</text>'
s+='<text x="70" y="342">颜色表示第一轮分配；十字为课件指定的三个初始中心。</text>'
save('kmeans-watermelon.svg','西瓜数据 K-means：第一轮分配与初始中心',s)

scores=[('全部 3 维',53.2917),('(x₂,x₃)',43.1667),('(x₁,x₂)',26.2821),('(x₂)',3.1250),('(x₁,x₃)',.7292),('(x₃)',.2917),('(x₁)',.1042)]
s='<path class="axis" d="M160 55V315H710"/>'
for i,(label,value) in enumerate(scores):
    y=72+i*35; w=value/55*520
    s+=f'<text x="35" y="{y+14}">{label}</text><rect x="160" y="{y}" width="{w:.1f}" height="20" fill="{("#087f8c" if i<2 else "#a88442")}"/><text x="{170+w:.1f}" y="{y+15}">{value:.4f}</text>'
save('filter-subsets.svg','Filter 原例：不同特征子集的 Fisher 型评分',s)

s='''<defs><marker id="arrg" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs>
<g text-anchor="middle"><ellipse cx="380" cy="75" rx="52" ry="27" fill="#e6f2f0" stroke="#087f8c"/><text x="380" y="80">下雨 R</text><ellipse cx="210" cy="165" rx="62" ry="27" fill="#f4efe3" stroke="#a88442"/><text x="210" y="170">洒水 S</text><ellipse cx="380" cy="270" rx="70" ry="27" fill="#f4e6e1" stroke="#a34b32"/><text x="380" y="275">草坪湿 G</text></g>
<g stroke="#087f8c" stroke-width="2" marker-end="url(#arrg)"><path d="M345 95L258 148"/><path d="M380 103V237"/><path d="M245 183L330 247"/></g>
<text x="455" y="80">P(R=1)=0.2</text><text x="55" y="215">P(S=1|R=1)=0.01</text><text x="55" y="238">P(S=1|R=0)=0.40</text><text x="470" y="230">P(G=1|1,1)=0.99</text><text x="470" y="253">P(G=1|1,0)=0.90</text><text x="470" y="276">P(G=1|0,1)=0.80</text><text x="470" y="299">P(G=1|0,0)=0</text>'''
save('grass-bn.svg','草坪贝叶斯网络：下雨、洒水与草坪湿',s)

s='''<defs><marker id="arrh" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#087f8c"/></marker></defs><g text-anchor="middle">'''
for t in range(5):
    x=90+t*140
    s+=f'<circle cx="{x}" cy="105" r="28" fill="#e6f2f0" stroke="#087f8c"/><text x="{x}" y="110">E{t+1}</text><circle cx="{x}" cy="235" r="25" fill="{("#f4e6e1" if t in (0,4) else "#d9dde1")}" stroke="#6b7779"/><text x="{x}" y="240">{("红" if t in (0,4) else "黑")}</text>'
    if t<4:s+=f'<path d="M{x+30} 105H{x+107}" stroke="#087f8c" stroke-width="2" marker-end="url(#arrh)"/>'
    s+=f'<path d="M{x} 135V205" stroke="#6b7779" marker-end="url(#arrh)"/>'
s+='</g><text x="55" y="310">π=(0.5,0.5)，A=[[0,1],[0.5,0.5]]，B=[[0,1],[0.5,0.5]]</text><text x="55" y="335">红球排除状态 0；黑球在状态 0 必然出现，在状态 1 概率为 0.5。</text>'
save('hmm-envelope.svg','信封 HMM：隐藏信封序列与红黑观测',s)

s='<g stroke="#c8d2d0" fill="none">'
for i in range(8):
    s+=f'<path d="M70 {70+i*30}H280M{70+i*30} 70V280"/>'
for i in range(6):
    s+=f'<path d="M470 {100+i*30}H620M{470+i*30} 100V250"/>'
s+='</g><rect x="70" y="70" width="90" height="90" fill="#a8d7d1" fill-opacity=".75" stroke="#087f8c"/><path d="M160 115H445" stroke="#087f8c" stroke-width="3"/><path d="M435 108L450 115L435 122" fill="none" stroke="#087f8c" stroke-width="3"/><rect x="470" y="100" width="30" height="30" fill="#087f8c"/>'
s+='<text x="75" y="315">7×7 输入；3×3 核从左上角开始</text><text x="470" y="285">5×5 输出；每个格对应一个局部窗口</text><text x="315" y="95">同一组 3×3 权重在 25 个位置共享</text>'
save('cnn-convolution.svg','七乘七输入上的三乘三卷积：局部连接与权重共享',s)

print('Generated 21 deterministic SVG figures.')
