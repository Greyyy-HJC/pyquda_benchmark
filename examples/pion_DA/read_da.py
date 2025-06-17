# %%
import numpy as np
import gvar as gv

import matplotlib.pyplot as plt

from lametlat.utils.resampling import *
from lametlat.preprocess.read_raw import pt2_to_meff
from lametlat.utils.plot_settings import *
from lametlat.utils.constants import *


N_conf = 10  # Adjust based on your data
z_list = np.arange(30)

# Load the data
da_path = f"pion_DA_Nconf{N_conf}.npz" # this is the output file from test on Swing


da_data = np.load(da_path, allow_pickle=True)    
da_DA = da_data["pion_DA"]
print(da_DA.shape)
    
# %%
da_jk = jackknife(da_DA)

da_jk_avg = jk_ls_avg(da_jk)

fig, ax = default_plot()
for idx, z in enumerate(z_list[:2]):
    da_meff = pt2_to_meff(da_jk_avg[idx], boundary="periodic")
    ax.errorbar(np.arange(len(da_meff)), gv.mean(da_meff), yerr=gv.sdev(da_meff), label=f"DA, z={z}", **errorb)

ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
ax.set_ylabel(r"$m_{\mathrm{eff}}$", **fs_p)
plt.tight_layout()
plt.show()

# %%
fix_t = 15
bare_da = []
for z in z_list:
    bare_da.append(da_jk_avg[z][fix_t])
bare_da = np.array(bare_da)
bare_da = bare_da / bare_da[0]

fig, ax = default_plot()
ax.errorbar(z_list, gv.mean(bare_da), yerr=gv.sdev(bare_da), label="DA", **errorb)

ax.legend(**fs_small_p)
ax.set_xlabel(r"$z$", **fs_p)
ax.set_ylabel(r"$h^0(z)$", **fs_p)
plt.tight_layout()
plt.show()
# %%
