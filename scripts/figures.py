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
print('Generated 6 deterministic SVG figures.')
