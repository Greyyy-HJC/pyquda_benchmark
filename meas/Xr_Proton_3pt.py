import sys
from itertools import permutations
import numpy as np
import cupy as cp
from opt_einsum import contract

#sys.path.insert(1, "/home/jiangxy/PyQUDA/")
from pyquda import init, core, LatticeInfo
from pyquda.utils import io, gamma, source

cfg = sys.argv[1]

init([1, 1, 1, 2], resource_path=".cache")

latt_info = LatticeInfo([24, 24, 24, 72], -1, 1.0)
dirac = core.getDirac(latt_info, -0.2400, 1e-12, 1000, 1.0, 1.160920226, 1.160920226, [[6, 6, 6, 4], [4, 4, 4, 9]])
gauge = io.readChromaQIOGauge("/public/home/jiangxy/summer_school/coulomb_fixed_configuration/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_"+str(cfg)+"_hyp0_gfixed3.scidac")
gauge.stoutSmear(1, 0.125, 4)
dirac.loadGauge(gauge)


C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
G_V = gamma.gamma(8)
G_A = gamma.gamma(11)

P_2pt = cp.zeros((72, 4, 4), "<c16")
P_2pt[:36] = (G0 + G4) / 2
P_2pt[36:] = (G0 - G4) / 2
P_3pt = 0.5*(gamma.gamma(0)+gamma.gamma(8))@(gamma.gamma(0)+gamma.gamma(11))
t_src_list = list(range(0, 72, 18))

epsilon= cp.zeros((3,3,3))
for a in range (3):
    b= (a+1)%3
    c= (a+2)%3
    epsilon[a,b,c]=1
    epsilon[a,c,b]=-1

# 2pt ###
proton_2pt = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
for t_idx, t_src in enumerate(t_src_list):
    propag = io.readNPYPropagator(f"/public/home/jiangxy/summer_school/coulomb_wall_propagator/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3.strange.tsrc_{t_src:02d}.npy")
    propag.toDevice()
    P2_ = cp.roll(P_2pt, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
    proton_2pt[t_idx] = contract(
            "abc, def, ij, kl, tmn, wtzyxikad, wtzyxjlbe, wtzyxmncf->t",
            epsilon,    epsilon,    C @ G5,    C @ G5,    P2_,
            propag.data,  propag.data,  propag.data,)
    - contract(
            "abc, def, ij, kl, tmn, wtzyxikad, wtzyxjnbf, wtzyxmlce->t",
            epsilon,    epsilon,    C @ G5,    C @ G5,    P2_,
            propag.data,  propag.data,  propag.data,
        )
tmp = core.gatherLattice(proton_2pt.real.get(), [1, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
    np.save("./result/proton_2pt_"+str(cfg)+".npy", tmp)
    print(tmp)

### 3pt ###
for tseq in [6,7,8,9]:
    proton_V = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    proton_A = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    for t_idx, t_src in enumerate(t_src_list):
        propag = io.readNPYPropagator(f"/public/home/jiangxy/summer_school/coulomb_wall_propagator/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3.strange.tsrc_{t_src:02d}.npy")
        propag.toDevice()
        src_seq = core.LatticePropagator(propag.latt_info)
        seq_p = contract(
            "abc, def, ij, kl, mn, wtzyxikad, wtzyxjlbe -> wtzyxmncf",
            epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
            propag.data,  propag.data
            ) 
        + contract(
            "abc, def, ij, kl, mn, wtzyxmkad, wtzyxjlbe -> wtzyxincf",
            epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
            propag.data,  propag.data
            ) 
        + contract(
            "abc, def, ij, kl, mn, wtzyxikad, wtzyxjnbe -> wtzyxmlcf ",
            epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
            propag.data,  propag.data
            ) 
        + contract(
            "abc, def, ij, kl, mn, wtzyxmkad, wtzyxjnbe  -> wtzyxilcf",
            epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
            propag.data,  propag.data
            )
        src_seq.data = contract("ij,wtzyxjkab,kl->wtzyxilab", G5, seq_p.conj(), G5)

        src_seq = source.sequential12(src_seq, (t_src + tseq) % (latt_info.global_size[3]))
        propag_seq = core.invertPropagator(dirac, src_seq)

        io.writeNPYPropagator(
            "./sequential_propagator_plus/"
            f"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3_SS_seq{tseq}_tsrc_{t_src:02d}.npy",
            propag_seq,
        )

        proton_V[t_idx] += contract(
            "ni, wtzyxjicf, jk,  km,  wtzyxmncf -> t",
             G5,   propag_seq.data.conj(),   G5,  G_V,  propag.data     
        )
        proton_A[t_idx] += contract(
            "ni, wtzyxjicf, jk,  km,  wtzyxmncf -> t",
             G5,   propag_seq.data.conj(),   G5,  G_A,  propag.data  
        )

    tmp = core.gatherLattice(proton_V.real.get(), [1, -1, -1, -1])
    if latt_info.mpi_rank == 0:
        for t_idx, t_src in enumerate(t_src_list):
            tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
        np.save("./result/protonV_seq"+str(tseq)+"_"+str(cfg)+".npy", tmp)
        print(tmp)

    tmp = core.gatherLattice(proton_A.real.get(), [1, -1, -1, -1])
    if latt_info.mpi_rank == 0:
        for t_idx, t_src in enumerate(t_src_list):
            tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
        np.save("./result/protonA_seq"+str(tseq)+"_"+str(cfg)+".npy", tmp)
        print(tmp)
dirac.destroy()