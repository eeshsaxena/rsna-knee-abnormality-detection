#!/bin/bash
# surface_mc.py variants (sequential, CPU, local). Each writes surface_<tag>.log / .json.
cd "$(dirname "$0")"
run() { py -3.12 -I -B surface_mc.py N=2000000 "$@" > /dev/null 2>&1 || echo "FAIL $*"; }
run tag=unif klo=0 khi=0
run tag=noswr swr=off
run tag=nolink link=off
run tag=lskep lomm=skeptic
run tag=mstack mcl=stack
run tag=mzero mcl=zero
run tag=mneg mcl=neg
run tag=kaphi kapl=.5 kapl_sd=.2 kapm=1.4
run tag=kaplo kapl=.15 kapl_sd=.08 kapm=.5
run tag=linear kapl=0 kapl_sd=0 kapm=0
run tag=tau6 tau=.6
run tag=eoaflat cmed=1.2 vmu=-1.0
run tag=eoasteep cmed=3.5 vmu=-3.0
run tag=wsd2 wsd=2.0
run tag=wsd4 wsd=4.0
run tag=rsd8 rsd=.8
run tag=lsd13 lsd=1.3
run tag=moaregime moa=regime
run tag=moaoptim moa=optim
run tag=moapess moa=pess
run tag=moaflip moa=flip
run tag=rho0 rho0=on
run tag=seed2 seed=77
echo done
