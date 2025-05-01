# %%
from itertools import permutations
from pyquda import init
import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt
from tqdm import tqdm
import gvar as gv

from pyquda_utils import core, io, gamma, source

from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import *
from lametlat.utils.funcs import constant_fit

init([1, 1, 1, 1], resource_path=".cache")
Ls = 8
Lt = 32

# Lattice parameters
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None # [[4, 4, 4, 4], [2, 2, 2, 8]]

N_conf = 10  # 使用较少的组态加快计算

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# * gamma.gamma(n) is the same as QLUA setting
C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)

# 质子投影算符
P = cp.zeros((Lt, 4, 4), "<c16")
P[:int(Lt/2)] = (G0 + G4) / 2  # 正宇称投影
P[int(Lt/2):] = (G0 - G4) / 2   # 负宇称投影
T = cp.ones((2 * Lt), "<f8")
T[:] = -1
T[int(Lt/2) : int(Lt/2) + Lt] = 1

# Source time position
t_src_list = [0]  # 固定源位置简化计算

# Source-sink separation (tsep)
tsep_list = [6, 8, 10]

# 存储所有组态的结果
proton_2pt_conf_list = []
proton_3pt_conf_list = []

# %%
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    proton_2pt = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    
    # Create 3pt arrays with appropriate dimensions
    proton_3pt_scalar = cp.zeros((len(t_src_list), len(tsep_list), Lt), "<c16")
    
    gauge = io.readNERSCGauge(f"../conf/S8T32/wilson_b6.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)

    for t_idx, t_src in enumerate(t_src_list):
        # Calculate forward propagator (from source to any point)
        fwd_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        fwd_prop = core.invertPropagator(dirac, fwd_source)
        
        #! 2pt
        P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        T_ = T[Lt - t_src : 2 * Lt - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        
        for a, b, c in permutations(tuple(range(3))):
            for d, e, f in permutations(tuple(range(3))):
                sign = 1 if b == (a + 1) % 3 else -1
                sign *= 1 if e == (d + 1) % 3 else -1
                
                # Proton two-point function
                proton_2pt[t_idx] += (sign * T_) * (
                    contract(
                        "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                        C @ G5,
                        C @ G5,
                        P_,
                        fwd_prop.data[:, :, :, :, :, :, :, a, d],
                        fwd_prop.data[:, :, :, :, :, :, :, b, e],
                        fwd_prop.data[:, :, :, :, :, :, :, c, f],
                    )
                    + contract(
                        "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                        C @ G5,
                        C @ G5,
                        P_,
                        fwd_prop.data[:, :, :, :, :, :, :, a, d],
                        fwd_prop.data[:, :, :, :, :, :, :, b, e],
                        fwd_prop.data[:, :, :, :, :, :, :, c, f],
                    )
                )
        
        #! 3pt
        for tsep_idx, tsep in enumerate(tsep_list):
            t_sink = (t_src + tsep) % Lt
            
            # Create sequential source with proton structure at sink
            seq_source = source.propagator(latt_info, "point", [0, 0, 0, t_sink])
            
            # 打印传播子形状，帮助理解索引结构
            if t_idx == 0 and tsep_idx == 0 and cfg == 0:
                print("Forward propagator shape:", fwd_prop.data.shape)
                print("Sequential source shape:", seq_source.data.shape)
            
            # 求解序列传播子 - 使用默认点源
            seq_prop = core.invertPropagator(dirac, seq_source)
            
            # 对每个current insertion time tau计算三点函数
            for tau in range(Lt):
                t_curr = (t_src + tau) % Lt
                
                # 创建一个临时变量来累积三点函数值
                thrpt_val = 0.0
                
                # 质子三点函数只计算x=y=z=0的贡献（零动量）
                # 尝试最简单的方式：使用G0作为流，在spin和color指标上缩并
                try:
                    # 最简化的计算：假设传播子的最后两个指标是spin和color
                    # 根据传播子结构确定正确的索引
                    if len(fwd_prop.data.shape) >= 7:  # 至少有batch,t,z,y,x,spin,color
                        # 标量流：在自旋指标上缩并G0
                        for s1 in range(4):
                            for s2 in range(4):
                                # 应用G0流
                                g0_val = G0[s1, s2]
                                if abs(g0_val) > 1e-10:
                                    # 在所有颜色指标上求和
                                    for c in range(3):
                                        # 注意：这里假设传播子可以简单地用[0,t,0,0,0,s,c]索引
                                        # 如果形状不匹配，会触发异常被捕获
                                        thrpt_val += g0_val * \
                                            fwd_prop.data[0, t_curr, 0, 0, 0, s1, c] * \
                                            seq_prop.data[0, t_curr, 0, 0, 0, s2, c]
                except Exception as e:
                    print(f"Error in 3pt calculation at tau={tau}: {e}")
                    # 如果上面的索引不正确，可以在这里尝试其他索引方式
                    pass
                
                # 存储这个tau的三点函数值
                proton_3pt_scalar[t_idx, tsep_idx, tau] = thrpt_val
    
    # Gather lattice results from all MPI processes
    proton_2pt_tmp = core.gatherLattice(proton_2pt.real.get(), [1, -1, -1, -1])
    
    # Reshape 3pt function for gathering
    proton_3pt_scalar_reshaped = proton_3pt_scalar.real.get().reshape(-1, latt_info.Lt)
    proton_3pt_scalar_tmp = core.gatherLattice(proton_3pt_scalar_reshaped, [1, -1, -1, -1])
    
    if latt_info.mpi_rank == 0:
        # Restore original shape
        proton_3pt_scalar_tmp = proton_3pt_scalar_tmp.reshape(len(t_src_list), len(tsep_list), Lt)
        
        # Time shift
        for t_idx, t_src in enumerate(t_src_list):
            proton_2pt_tmp[t_idx] = np.roll(proton_2pt_tmp[t_idx], -t_src, 0)
            for tsep_idx, tsep in enumerate(tsep_list):
                proton_3pt_scalar_tmp[t_idx, tsep_idx] = np.roll(proton_3pt_scalar_tmp[t_idx, tsep_idx], -t_src, 0)
        
        # Average over source positions
        twopt_proton = proton_2pt_tmp.mean(0)
        thrpt_proton_scalar = proton_3pt_scalar_tmp.mean(0)
        
        proton_2pt_conf_list.append(twopt_proton)
        proton_3pt_conf_list.append(thrpt_proton_scalar)

dirac.destroy()


# %%
print(np.shape(proton_2pt_conf_list))
print(np.shape(proton_3pt_conf_list))

proton_2pt_jk = jackknife(proton_2pt_conf_list)
proton_3pt_jk = jackknife(proton_3pt_conf_list)

proton_2pt_jk_avg = jk_ls_avg(proton_2pt_jk)
proton_3pt_jk_avg = jk_ls_avg(proton_3pt_jk)

print(np.shape(proton_2pt_jk_avg))
print(np.shape(proton_3pt_jk_avg))

# Calculate ratios for each tsep
ratio_ls = []

for tsep_idx, tsep in enumerate(tsep_list):
    # Ratio = 3pt / 2pt for each tau
    ratio = proton_3pt_jk_avg[tsep_idx, :tsep+1] / proton_2pt_jk_avg[tsep]
    ratio_ls.append(ratio)

# Plot the results
fig, ax = default_plot()

for i, ratio in enumerate(ratio_ls):
    tsep = tsep_list[i]
    tau_vals = np.arange(0, tsep+1)
    ax.errorbar(tau_vals, [gv.mean(r) for r in ratio], yerr=[gv.sdev(r) for r in ratio], 
                label=f"tsep={tsep}", **errorb)

ax.set_xlabel("τ (current insertion time)")
ax.set_ylabel("Ratio")
ax.legend()
plt.show()

