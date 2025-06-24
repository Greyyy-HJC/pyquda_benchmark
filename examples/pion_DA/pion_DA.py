# %%
import os
import time
import numpy as np
import gvar as gv
import cupy as cp
from tqdm import tqdm
from pyquda import init
from pyquda_utils import core, io, source
from opt_einsum import contract
from pyquda.field import LatticeGauge
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

import gpt as g
from pyquda_utils.gpt import LatticePropagatorGPT


N_conf = g.default.get_int("--N_conf", 50)
mpi_geometry = g.default.get_single("--mpi_geometry", "1.1.1.4").split(".")
mpi_geometry = [int(x) for x in mpi_geometry]

if not os.path.exists(".cache"):
    os.makedirs(".cache")
    print("Created .cache directory for PyQUDA resources")


Ls = 48
Lt = 64
init(mpi_geometry, [Ls, Ls, Ls, Lt], resource_path=".cache", enable_mps=True)

conf_path = "/lustre1/pion3d/ensemble/l4864f21b7373m00125m0250a.nersc.cg_high_prec/fixed_GLU"

# Lattice parameters
xi_0, nu = 1.0, 1.0
qmass = -0.0191  # -0.038888 for 300 MeV pion; -0.0191 for 670 MeV pion #todo
inv_precision = 1e-10  # todo
csw_r = 1.0336
csw_t = 1.0336
last_mg = int(64 / mpi_geometry[-1] / 4)
multigrid = [[3, 3, 3, 2], [4, 4, 4, 2], [4, 4, 4, last_mg]]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(
    latt_info, qmass, inv_precision, 1000, xi_0, csw_r, csw_t, multigrid
)

# * gamma.gamma(n) is the same as QLUA setting
C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
G4G5 = gamma.gamma(7)
G5Z = gamma.gamma(11)

t_src_list = [0]
z_list = list(range(8))

# Loop over gauge configurations
conf_num_ls = np.arange(1008, 1008 + N_conf * 6, 6)
pion_DA = []

for cfg in conf_num_ls:
    # Print current time and configuration number
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nProcessing configuration {cfg} at {current_time}")
    
    gauge = io.readNERSCGauge(
        f"{conf_path}/l4864f21b7373m00125m0250a.{cfg}.coulomb.1e-14"
    )
    gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)  # TODO: Note that HYP here can be different from the HYP in GPT
    dirac.loadGauge(gauge)
    
    #! GPT
    gpt_read = g.convert(g.load(f"{conf_path}/l4864f21b7373m00125m0250a.{cfg}.coulomb.1e-14"), g.double) # load configuration
    grid = gpt_read[0].grid
    GEN_SIMD_WIDTH = 64
    #!
    

    # pion_DA_tmp = cp.zeros((len(t_src_list), len(z_list), latt_info.Lt), "<c16")
    gpt_DA_tmp = []

    for t_idx, t_src in enumerate(t_src_list):
        gpt_DA_tmp.append([])
        
        # create point source and compute propagator
        point_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        point_propag = core.invertPropagator(dirac, point_source)
        point_propag_shift = point_propag.copy()
        
        point_propag_backward_data = contract(
            "li,wtzyxjiba,jk->wtzyxklba",
            G5 @ G5,
            point_propag.data.conj(),
            G5 @ G4G5
        )
        
        #! GPT
        gpt_propag = g.mspincolor(grid)
        LatticePropagatorGPT(gpt_propag, GEN_SIMD_WIDTH, point_propag)
        #!
        
        for z in z_list:
            # Time the contraction
            cp.cuda.runtime.deviceSynchronize()
            start_contract = time.time()
            
            
            # #! contract
            # pion_DA_tmp[t_idx, z] = contract(
            #     "wtzyxklba,wtzyxklba->t",
            #     point_propag_backward_data,
            #     point_propag_shift.data,
            # )
            # #!
            
            #! GPT slice_trDA
            gpt_temp = np.real( np.array( g.slice(g.trace(
                g.gamma[5] * g.gamma[5]
                * g.adj( gpt_propag )
                * g.gamma["T"]
                * g.eval( g.cshift( gpt_propag, 2, z ) )
            ), 3) ) )
            gpt_DA_tmp[t_idx].append(gpt_temp)
            
            cp.cuda.runtime.deviceSynchronize()
            end_contract = time.time()
            print(f">>> Contraction time: {end_contract - start_contract:.3f} seconds")
            
            # # Time the shift operation
            # cp.cuda.runtime.deviceSynchronize()
            # start_shift = time.time()
            
            #! use gauge.pure_gauge.covDev to shift each fermion's data
            # unit = LatticeGauge(latt_info)
            # unit.gauge_dirac.loadGauge(unit)
            # for spin in range(4):
            #     for color in range(3):
            #         #! if CG, no Wilson link
            #         fermion = point_propag_shift.getFermion(spin, color)
            #         fermion_unit = unit.pure_gauge.covDev(fermion, 2) # x, y, z, t
            #         point_propag_shift.setFermion(fermion_unit, spin, color)
            
            # unit.gauge_dirac.loadGauge(gauge)
                    
            #! shift each fermion's data
            # point_propag_shift = point_propag_shift.shift(-1, Z)
            #!
            
            # cp.cuda.runtime.deviceSynchronize()
            # end_shift = time.time()
            # print(f">>> Shift time: {end_shift - start_shift:.3f} seconds")
                    
    # Gather and average over spatial lattice
    # pion_DA_tmp = core.gatherLattice(pion_DA_tmp.real.get(), [2, -1, -1, -1]) # 2 means the t-axis is the 2-nd axis
    
    pion_DA_tmp = np.array(gpt_DA_tmp)

    if latt_info.mpi_rank == 0:
        print("latt_info.Lt: ", latt_info.Lt)
        print("pion_DA_tmp shape: ", np.shape(pion_DA_tmp))
        
        for t_idx, t_src in enumerate(t_src_list):
            pion_DA_tmp[t_idx] = np.roll(pion_DA_tmp[t_idx], -t_src, 0)
        
        pion_DA.append(pion_DA_tmp.mean(0))

dirac.destroy()

if latt_info.mpi_rank == 0:
    dump_dic = {}
    dump_dic["pion_DA"] = np.array(pion_DA)
    
    np.savez(f"dump/pion_DA_Nconf{N_conf}_gpt.npz", **dump_dic)
    print(">>> Mean value of pion_DA: \n", np.mean(pion_DA, axis=0)[:, :20])

# %%
