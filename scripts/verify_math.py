"""Independent checks for examples, identities and gradients. Requires numpy."""
from pathlib import Path
import math,itertools,json
import numpy as np
checks=[]
def check(name,condition):
    if not bool(condition):raise AssertionError(name)
    checks.append(name)
def close(a,b):return np.allclose(a,b,rtol=1e-8,atol=1e-9)
p=(265*.7)/(265*.7+100*.1)
check('weather posterior',close(p,185.5/195.5))
check('weather without sunset',close(265*.3/(265*.3+100*.9),79.5/169.5))
check('Beta MAP and predictive differ',close(19/33,.5757575757575758) and close(20/35,.5714285714285714))
boundary=1+math.log(4)/2
check('Gaussian prior-shift boundary',abs(-2*boundary+2+math.log(4))<1e-12)
mu=[np.array([3.,6.]),np.array([3.,-2.])];cov=[np.diag([.5,2]),np.diag([2.,2.])]
def g(x,c):
    v=x-mu[c];return -.5*v@np.linalg.solve(cov[c],v)-.5*np.linalg.slogdet(cov[c])[1]+math.log(.5)
for x1 in [-3.,0.,3.,7.]:
    x2=.1875*(x1-3)**2+2-math.log(2)/4
    check(f'Gaussian quadratic boundary x1={x1}',abs(g(np.array([x1,x2]),0)-g(np.array([x1,x2]),1))<1e-10)
a=np.array([3,3]);b=np.array([1,2])
check('distance example',close(np.linalg.norm(a-b),math.sqrt(5)) and close(abs(a-b).sum(),3) and close(abs(a-b).max(),2) and close(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),9/math.sqrt(90)))
check('tree information gain',close(1+2*(.75*math.log2(.75)+.25*math.log2(.25))/2,.18872187554086717))
check('three-classifier majority',close(3*.7**2*.3+.7**3,.784))
check('AdaBoost multiplier',close(math.exp(2*.5*math.log(.7/.3)),7/3))
X=np.array([[0.,1.],[1.,1.],[2.,1.]]);y=np.array([1.,2.,2.]);coef=np.linalg.lstsq(X,y,rcond=None)[0]
check('least squares coefficients',close(coef,[.5,7/6]))
check('least squares normal residual',close(X.T@(y-X@coef),[0,0]))
check('least squares MSE',close(np.mean((y-X@coef)**2),1/18))
data=np.array([0.,2.,8.,10.]);centers=np.array([0.,8.]);labels=abs(data[:,None]-centers).argmin(axis=1);before=sum((data-centers[labels])**2);new=np.array([data[labels==k].mean() for k in range(2)])
check('K-means monotone update',close(before,8) and close(sum((data-new[labels])**2),4))
aa=np.array([0.,1.,5.]);bb=np.array([8.,9.]);point=5.;na=len(aa);nb=len(bb)
delta=nb/(nb+1)*(point-bb.mean())**2-na/(na-1)*(point-aa.mean())**2
sse=lambda x:sum((x-x.mean())**2)
check('single transfer identity',close(delta,sse(aa[:-1])+sse(np.append(bb,point))-sse(aa)-sse(bb)))
P=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
check('PCA covariance eigenvalues',close(np.linalg.eigvalsh(P.T@P/4),[0.,2.5]))
check('maximum entropy example',close(.15+.15+3*7/30,1))
# HMM forward and Viterbi independently compared with complete path enumeration.
A=np.array([[.7,.3],[.4,.6]]);B=np.array([[.9,.1],[.2,.8]]);pi=np.array([.6,.4]);obs=[0,1,0]
alpha=pi*B[:,obs[0]];v=alpha.copy()
for o in obs[1:]:alpha=(alpha@A)*B[:,o];v=(v[:,None]*A).max(axis=0)*B[:,o]
probs=[]
for z in itertools.product(range(2),repeat=3):
    q=pi[z[0]]*B[z[0],obs[0]]
    for t in range(1,3):q*=A[z[t-1],z[t]]*B[z[t],obs[t]]
    probs.append(q)
check('HMM forward versus enumeration',close(alpha.sum(),sum(probs)))
check('Viterbi versus enumeration',close(v.max(),max(probs)))
hmm_best=float(v.max())
# Logistic gradient central finite differences, including bias.
XX=np.array([[1.,2.,1.],[-1.,.5,1.],[.3,-2.,1.]]);yy=np.array([1.,0.,1.]);theta=np.array([.2,-.4,.1])
loss=lambda t:np.mean(np.logaddexp(0,XX@t)-yy*(XX@t))
analytic=XX.T@(1/(1+np.exp(-XX@theta))-yy)/len(yy)
eps=1e-6;numeric=np.array([(loss(theta+eps*np.eye(3)[j])-loss(theta-eps*np.eye(3)[j]))/(2*eps) for j in range(3)])
check('logistic gradient finite difference',close(analytic,numeric))
# Two-layer tanh network BP against finite differences for all parameters.
x=np.array([.4,-.7]);target=.3;t=np.array([.2,-.1,.3,.5,.1,-.2,.4,-.6,.05])
def unpack(t):return t[:4].reshape(2,2),t[4:6],t[6:8],t[8]
def network(t):
    W,b,v,c=unpack(t);h=np.tanh(W@x+b);return .5*(v@h+c-target)**2
W,b,v,c=unpack(t);h=np.tanh(W@x+b);e=v@h+c-target;delta=e*v*(1-h*h);grad=np.r_[(delta[:,None]*x).ravel(),delta,e*h,e]
fd=np.array([(network(t+eps*np.eye(9)[j])-network(t-eps*np.eye(9)[j]))/(2*eps) for j in range(9)])
check('multilayer BP all-parameter finite difference',close(grad,fd))
check('CNN output and parameter count',(32-5)//1+1==28 and 6*(5*5*3+1)==456)
for x1,x2 in itertools.product([0.,1.],repeat=2):check(f'XOR {x1},{x2}',close(max(0,x1+x2)-2*max(0,x1+x2-1),int(x1!=x2)))
out={'passed':len(checks),'checks':checks,'weather_posterior':p,'hmm_likelihood':float(alpha.sum()),'hmm_best_path_probability':hmm_best,'randomness':'none; deterministic inputs'}
(Path(__file__).resolve().parents[1]/'math-report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
