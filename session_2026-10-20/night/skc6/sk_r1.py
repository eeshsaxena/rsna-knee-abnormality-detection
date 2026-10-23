# skeptic C6': R1 arithmetic (MODEL). delta ~ N(mu, tau); public = delta + e_pub (s_pub); private = delta + e_priv (s_priv); move iff public > 0.
import numpy as np
rng=np.random.default_rng(3); N=2_000_000
for lab,mu,tau in [('probe as proposed',.4,.6),('skeptic central',-.1,.65),('skeptic RSNA-only',-.25,.6),('skeptic + Route O 20%',0.,.7)]:
    d=rng.normal(mu,tau,N); pub=d+rng.normal(0,1.0,N); pri=d+rng.normal(0,.65,N); m=pub>0
    print('%-24s mu %+.2f | P(mu>0) %.2f | P(move) %.2f | E[priv|move] %+.2f | R1 value E[priv*1(move)] %+.2f | P(priv>0|move) %.2f | P(priv<-0.5|move) %.2f'%(
        lab,mu,(d>0).mean(),m.mean(),pri[m].mean(),(pri*m).mean(),(pri[m]>0).mean(),(pri[m]<-.5).mean()))
