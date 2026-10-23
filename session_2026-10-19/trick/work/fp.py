import numpy as np, glob, os, sys, time
from multiprocessing import Pool
S="C:/Users/eeshs/Downloads/rsna-knee/_crop_shards"
def one(path):
    z=np.load(path,allow_pickle=False)
    img=z['img']; meta=z['meta']
    out={}
    small=img.reshape(len(img),32,12,32,12).mean(axis=(2,4)).astype(np.float32)  # 32x32
    for si in np.unique(meta[:,0]):
        for pl in range(3):
            for fb in (0,1):
                m=(meta[:,0]==si)&(meta[:,1]==pl)&(meta[:,2]==fb)
                if m.sum()==0: continue
                a=small[m]
                out[(int(si),pl,fb)]=(a.mean(0), a)   # mean image and per-slice smalls
    return out
if __name__=="__main__":
    t=time.time()
    files=sorted(glob.glob(S+"/shard_*.npz"))
    res={}
    with Pool(6) as p:
        for d in p.imap_unordered(one,files):
            res.update(d)
    print("keys",len(res),"time",round(time.time()-t))
    import pickle
    pickle.dump(res,open("fp32.pkl","wb"),protocol=4)
