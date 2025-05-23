# %%
import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt

from pyquda_utils import core, io, gamma, phase

core.init([1, 1, 1, 1], resource_path=".cache")
latt_info = core.LatticeInfo([8, 8, 8, 32], -1, 1.0)

dirac = core.getDirac(latt_info, -0.2770, 1e-12, 1000, 1.0, 1.160920226, 1.160920226)
gauge = io.readNERSCGauge(f"../conf/S8T32/wilson_b6.0")
# gauge.stoutSmear(1, 0.125, 4)
dirac.loadGauge(gauge)

G5 = gamma.gamma(15)
momentum_list = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 2], [0, 1, 2], [1, 1, 2], [0, 2, 2], [1, 2, 2]]
momentum_phase = phase.MomentumPhase(latt_info).getPhases(momentum_list)
t_src_list = list(range(0, 32, 32))
pion = cp.zeros((len(t_src_list), len(momentum_list), latt_info.Lt), "<c16")

for t_idx, t_src in enumerate(t_src_list):
    propag = core.invert(dirac, "point", [0, 0, 0, t_src])

    pion[t_idx] += contract(
        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
        momentum_phase,
        propag.data.conj(),
        G5 @ G5,
        propag.data,
        G5 @ G5,
    )

dirac.destroy()

tmp = core.gatherLattice(pion.real.get(), [2, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 1)
    twopt = tmp.mean(0)
    mass = np.arccosh((twopt[:, 2:] + twopt[:, :-2]) / (2 * twopt[:, 1:-1]))
    for p_idx, mom in enumerate(momentum_list):
        plt.plot(np.arange(1, 32 - 1), mass[p_idx], ",-", label=f"{mom}")
    plt.legend()
    plt.savefig("../output/plots/pion_dispersion.svg")
    plt.clf()
# %%
