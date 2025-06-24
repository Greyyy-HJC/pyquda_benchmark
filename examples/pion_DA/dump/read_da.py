# %%
import numpy as np
import gvar as gv

import matplotlib.pyplot as plt
import os

from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff
from lametlat.utils.plot_settings import *
from lametlat.utils.constants import *


N_conf = 10  # Adjust based on your data
z_list = np.arange(8)

# Load the data
shift_path = f"pion_DA_Nconf{N_conf}_shift.npz"
covdev_path = f"pion_DA_Nconf{N_conf}_covDev.npz"


shift_data = np.load(shift_path, allow_pickle=True)    
shift_DA = shift_data["pion_DA"]
print(shift_DA.shape)

covdev_data = np.load(covdev_path, allow_pickle=True)
covdev_DA = covdev_data["pion_DA"]
print(covdev_DA.shape)
    
# %%
shift_da_jk = jackknife(shift_DA)
covdev_da_jk = jackknife(covdev_DA)

shift_da_jk_avg = jk_ls_avg(shift_da_jk)
covdev_da_jk_avg = jk_ls_avg(covdev_da_jk)

fig, ax = default_plot()
for idx, z in enumerate(z_list[:2]):
    shift_meff = pt2_to_meff(shift_da_jk_avg[idx], boundary="periodic")
    covdev_meff = pt2_to_meff(covdev_da_jk_avg[idx], boundary="periodic")
    ax.errorbar(np.arange(len(shift_meff)), gv.mean(shift_meff), yerr=gv.sdev(shift_meff), label=f"shift, z={z}", **errorb)
    ax.errorbar(np.arange(len(covdev_meff)), gv.mean(covdev_meff), yerr=gv.sdev(covdev_meff), label=f"covdev, z={z}", **errorb)

ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
# plt.savefig("../output/plots/pion_meff_DA.pdf", transparent=True)
plt.show()

# %%
fix_t = 15
bare_shift_da = []
for z in z_list:
    bare_shift_da.append(shift_da_jk_avg[z][fix_t])
bare_shift_da = np.array(bare_shift_da)
bare_shift_da = bare_shift_da / bare_shift_da[0]

bare_covdev_da = []
for z in z_list:
    bare_covdev_da.append(covdev_da_jk_avg[z][fix_t])
bare_covdev_da = np.array(bare_covdev_da)
bare_covdev_da = bare_covdev_da / bare_covdev_da[0]

fig, ax = default_plot()
ax.errorbar(z_list, gv.mean(bare_shift_da), yerr=gv.sdev(bare_shift_da), label="shift", **errorb)
ax.errorbar(z_list+0.1, gv.mean(bare_covdev_da), yerr=gv.sdev(bare_covdev_da), label="covdev", **errorb)

ax.legend(**fs_small_p)
ax.set_xlabel(r"$z$", **fs_p)
ax.set_ylabel(r"$h^0(z)$", **fs_p)
plt.tight_layout()
# plt.savefig("../output/plots/pion_DA_bare.pdf", transparent=True)
plt.show()
# %%
