# %%
import os
import gvar as gv
import cupy as cp
from tqdm import tqdm
from pyquda import init
from pyquda_utils import core, io, source
from opt_einsum import contract
from itertools import permutations


from pyquda_utils import core, io, gamma, source

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")

init([1, 1, 1, 1], resource_path=".cache")

xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

N_conf = 1


# Lattice and solver setup
latt_info = core.LatticeInfo([8, 8, 8, 32], -1, xi_0 / nu)
Ls = latt_info.global_size[0]
Lt = latt_info.global_size[3]
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# * gamma.gamma(n) is the same as QLUA setting
C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
G5Z = gamma.gamma(11)

GA = G5Z
GV = G4

# 质子投影算符
P = cp.zeros((Lt, 4, 4), "<c16")
P[:int(Lt/2)] = (G0 + G4) / 2  # 正宇称投影
P[int(Lt/2):] = (G0 - G4) / 2   # 负宇称投影
P_3pt = 0.5 * (G0 + G4) @ (G0 + G5Z)
T = cp.ones((2 * Lt), "<f8")
T[:] = -1
T[int(Lt/2) : int(Lt/2) + Lt] = 1

# Source/sink separations and ensemble size
t_src_list = [0] # list(range(0, Lt, int(Lt/4)))
t_sep_list = [6, 8, 10]

epsilon= cp.zeros((3,3,3))
for a in range (3):
    b = (a+1) % 3
    c = (a+2) % 3
    epsilon[a,b,c] = 1
    epsilon[a,c,b] = -1

# Storage for correlators
pt2_proton = []
pt3_proton_gA = []
pt3_proton_gV = []

# Loop over gauge configurations
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    pt2_conf = cp.zeros((len(t_src_list), Lt), dtype="<c16")
    pt3_conf_gA = cp.zeros((len(t_src_list), len(t_sep_list), Lt), dtype="<c16")
    pt3_conf_gV = cp.zeros((len(t_src_list), len(t_sep_list), Lt), dtype="<c16")

    gauge = io.readNERSCGauge(f"../conf/S8T32/wilson_b6.{cfg}")
    dirac.loadGauge(gauge)

    for t_idx, t_src in enumerate(t_src_list):
        # create point source and compute propagator
        point_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        point_propag = core.invertPropagator(dirac, point_source)
                
        
        #! 2pt: proton two-point function
        P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        T_ = T[Lt - t_src : 2 * Lt - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        
        
        pt2_conf[t_idx] += contract(
            "abc, def, ij, kl, tmn, wtzyxikad, wtzyxjlbe, wtzyxmncf->t",
            epsilon,    epsilon,    C @ G5,    C @ G5,    P_,
            point_propag.data,  point_propag.data,  point_propag.data,
        )
        - contract(
            "abc, def, ij, kl, tmn, wtzyxikad, wtzyxjnbf, wtzyxmlce->t",
            epsilon,    epsilon,    C @ G5,    C @ G5,    P_,
            point_propag.data,  point_propag.data,  point_propag.data,
        )

        #! 3pt: proton 3-point function with sequential source technique
        for t_sep_idx, t_sep in enumerate(t_sep_list):
            t_snk = (t_src + t_sep) % Lt
            
            # Create sequential source at sink position
            seq_source = core.LatticePropagator(latt_info)
            
            seq_temp = contract(
                "abc, def, ij, kl, mn, wtzyxikad, wtzyxjlbe -> wtzyxmncf",
                epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
                point_propag.data,  point_propag.data
            )
            + contract(
                "abc, def, ij, kl, mn, wtzyxmkad, wtzyxjlbe -> wtzyxincf",
                epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
                point_propag.data,  point_propag.data
            )
            + contract(
                "abc, def, ij, kl, mn, wtzyxikad, wtzyxjnbe -> wtzyxmlcf ",
                epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
                point_propag.data,  point_propag.data
            ) 
            + contract(
                "abc, def, ij, kl, mn, wtzyxmkad, wtzyxjnbe  -> wtzyxilcf",
                epsilon,    epsilon,    C@G5,    C@G5,    P_3pt,
                point_propag.data,  point_propag.data
            )
            
            seq_source.data = contract("ij,wtzyxjkab,kl->wtzyxilab", G5, seq_temp.conj(), G5)
            seq_source = source.sequential12(seq_source, t_snk)
            seq_propag = core.invertPropagator(dirac, seq_source)
            
            pt3_conf_gA[t_idx, t_sep_idx] += contract(
                "ni, wtzyxjicf, jk, km, wtzyxmncf -> t",
                G5, seq_propag.data.conj(), G5, GA, point_propag.data  
            )
            pt3_conf_gV[t_idx, t_sep_idx] += contract(
                "ni, wtzyxjicf, jk, km, wtzyxmncf -> t",
                G5, seq_propag.data.conj(), G5, GV, point_propag.data  
            )

    # Gather and average over spatial lattice
    pt2_tmp = core.gatherLattice(pt2_conf.real.get(), [1, -1, -1, -1])
    pt3_tmp_gA = core.gatherLattice(pt3_conf_gA.real.get(), [2, -1, -1, -1]) # * since the 3pt has one more dimension of tsep
    pt3_tmp_gV = core.gatherLattice(pt3_conf_gV.real.get(), [2, -1, -1, -1])

    if latt_info.mpi_rank == 0:
        for t_idx, t_src in enumerate(t_src_list):
            pt2_tmp[t_idx] = np.roll(pt2_tmp[t_idx], -t_src, 0)
            pt3_tmp_gA[t_idx] = np.roll(pt3_tmp_gA[t_idx], -t_src, 0)
            pt3_tmp_gV[t_idx] = np.roll(pt3_tmp_gV[t_idx], -t_src, 0)
        
        pt2_proton.append(pt2_tmp.mean(0))
        pt3_proton_gA.append(pt3_tmp_gA.mean(0))
        pt3_proton_gV.append(pt3_tmp_gV.mean(0))

dirac.destroy()

# %%
# Jackknife and average

print("shape of pt2_proton: ", np.shape(pt2_proton))
print("shape of pt3_proton_gA: ", np.shape(pt3_proton_gA))
print("shape of pt3_proton_gV: ", np.shape(pt3_proton_gV))

pt2_p_jk = jackknife(pt2_proton)
pt3_p_jk_gA = jackknife(pt3_proton_gA)
pt3_p_jk_gV = jackknife(pt3_proton_gV)

pt2_p_avg = jk_ls_avg(pt2_p_jk)
pt3_p_avg_gA = jk_ls_avg(pt3_p_jk_gA)
pt3_p_avg_gV = jk_ls_avg(pt3_p_jk_gV)

print("shape of pt2_p_avg: ", np.shape(pt2_p_avg))
print("shape of pt3_p_avg_gA: ", np.shape(pt3_p_avg_gA))
print("shape of pt3_p_avg_gV: ", np.shape(pt3_p_avg_gV))

# Scalar charge ratio g_S = pt3 / pt2
tau_cut = 1
ratio_gA_ls = []
ratio_gV_ls = []
for tsep_idx, tsep in enumerate(t_sep_list):
    ratio_gA_ls.append(pt3_p_avg_gA[tsep_idx] / pt2_p_avg[tsep])
    ratio_gV_ls.append(pt3_p_avg_gV[tsep_idx] / pt2_p_avg[tsep])
    
# Plot the meff
from lametlat.preprocess.read_raw import pt2_to_meff

pt2_p_meff = pt2_to_meff(pt2_p_avg, boundary="periodic")

fig, ax = default_plot()
ax.errorbar(np.arange(len(pt2_p_meff)), gv.mean(pt2_p_meff), yerr=gv.sdev(pt2_p_meff), **errorb)
ax.set_xlabel("t", **fs_p)
ax.set_ylabel("meff", **fs_p)
plt.tight_layout()
plt.show()


# Plot the ratio
fig, ax = default_plot()
for i, (ratio_gA, ratio_gV) in enumerate(zip(ratio_gA_ls, ratio_gV_ls)):
    tsep = t_sep_list[i]
    tau_vals = np.arange(tau_cut, tsep+1-tau_cut) - tsep/2
    ratio_gv = -( np.array(ratio_gA)[tau_cut:tsep+1-tau_cut] / np.array(ratio_gV)[tau_cut:tsep+1-tau_cut] )
    ax.errorbar(
        tau_vals,
        gv.mean(ratio_gv),
        yerr=gv.sdev(ratio_gv),
        label=f"tsep={tsep}",
        **errorb
    )
ax.set_xlabel("τ (current insertion time)", **fs_p)
ax.set_ylabel("Ratio", **fs_p)
ax.legend(ncol=3, **fs_small_p)
# ax.set_ylim(auto_ylim([gv.mean(ratio_gv)], [gv.sdev(ratio_gv)], 2))
ax.set_ylim(-0.5, 2)
plt.tight_layout()
plt.savefig("../output/plots/proton_3pt_ratio.pdf", transparent=True)
plt.show()


# %%
