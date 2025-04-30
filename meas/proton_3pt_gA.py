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

N_conf = 20  # 使用较少的组态加快计算

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)

# * gamma.gamma(n) is the same as QLUA setting
C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)

# 轴矢流算符需要的 gamma_5 * gamma_mu 
# gamma_mu 包括： gamma_x (1), gamma_y (2), gamma_z (4)
GAx = gamma.gamma(1) @ G5  # γ_5 * γ_x
GAy = gamma.gamma(2) @ G5  # γ_5 * γ_y
GAz = gamma.gamma(4) @ G5  # γ_5 * γ_z

# 质子投影算符
P = cp.zeros((Lt, 4, 4), "<c16")
P[:int(Lt/2)] = (G0 + G4) / 2  # 正宇称投影
P[int(Lt/2):] = (G0 - G4) / 2   # 负宇称投影
T = cp.ones((2 * Lt), "<f8")
T[:] = -1
T[int(Lt/2) : int(Lt/2) + Lt] = 1

# 源时间位置
t_src_list = [0]  # 固定源位置简化计算

# 定义流插入的间隔距离
tau_list = list(range(1, 12))  # 流与源之间的时间间隔

# 存储所有组态的结果
proton_2pt_conf_list = []
proton_3pt_x_conf_list = []
proton_3pt_y_conf_list = []
proton_3pt_z_conf_list = []

# %%
for cfg in tqdm(range(N_conf), desc="Processing configurations"):
    
    proton_2pt = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
    proton_3pt_x = cp.zeros((len(t_src_list), len(tau_list), latt_info.Lt), "<c16")
    proton_3pt_y = cp.zeros((len(t_src_list), len(tau_list), latt_info.Lt), "<c16")
    proton_3pt_z = cp.zeros((len(t_src_list), len(tau_list), latt_info.Lt), "<c16")
    
    gauge = io.readNERSCGauge(f"../conf/S8T32/wilson_b6.{cfg}")
    # gauge.stoutSmear(1, 0.125, 4)
    dirac.loadGauge(gauge)

    for t_idx, t_src in enumerate(t_src_list):
        # 计算向前传播子（从源到任意点）
        fwd_source = source.propagator(latt_info, "point", [0, 0, 0, t_src])
        fwd_prop = core.invertPropagator(dirac, fwd_source)
        
        # 计算质子二点函数
        P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        T_ = T[Lt - t_src : 2 * Lt - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
        
        for a, b, c in permutations(tuple(range(3))):
            for d, e, f in permutations(tuple(range(3))):
                sign = 1 if b == (a + 1) % 3 else -1
                sign *= 1 if e == (d + 1) % 3 else -1
                
                # 质子二点函数
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
        
        # 对每个流插入时间计算三点函数
        for tau_idx, tau in enumerate(tau_list):
            t_curr = (t_src + tau) % Lt
            
            # 创建序列源（从汇到流）
            # 先创建一个位于汇的点源（作为序列传播子的起点）
            sink_source = source.propagator(latt_info, "point", [0, 0, 0, t_curr])
            
            # 为每个方向计算序列传播子
            # 在 x 方向上插入 γ5γx
            seq_prop_x = core.invertPropagator(dirac, sink_source)
            # 手动应用 GAx 算符：传播子的每个自旋分量与轴矢流 GAx 做内积
            seq_prop_x_with_current = contract("wtzyxjiba,jk->wtzyxkiba", seq_prop_x.data, GAx)
            
            # 在 y 方向上插入 γ5γy
            seq_prop_y = core.invertPropagator(dirac, sink_source)
            # 手动应用 GAy 算符
            seq_prop_y_with_current = contract("wtzyxjiba,jk->wtzyxkiba", seq_prop_y.data, GAy)
            
            # 在 z 方向上插入 γ5γz
            seq_prop_z = core.invertPropagator(dirac, sink_source)
            # 手动应用 GAz 算符
            seq_prop_z_with_current = contract("wtzyxjiba,jk->wtzyxkiba", seq_prop_z.data, GAz)
            
            # 计算三点函数（针对每个轴矢流方向）
            for a, b, c in permutations(tuple(range(3))):
                for d, e, f in permutations(tuple(range(3))):
                    sign = 1 if b == (a + 1) % 3 else -1
                    sign *= 1 if e == (d + 1) % 3 else -1
                    
                    # x 方向轴矢流的三点函数
                    proton_3pt_x[t_idx, tau_idx] += (sign * T_) * (
                        contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_x_with_current[:, :, :, :, :, :, :, c, f],
                        )
                        + contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_x_with_current[:, :, :, :, :, :, :, c, f],
                        )
                    )
                    
                    # y 方向轴矢流的三点函数
                    proton_3pt_y[t_idx, tau_idx] += (sign * T_) * (
                        contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_y_with_current[:, :, :, :, :, :, :, c, f],
                        )
                        + contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_y_with_current[:, :, :, :, :, :, :, c, f],
                        )
                    )
                    
                    # z 方向轴矢流的三点函数
                    proton_3pt_z[t_idx, tau_idx] += (sign * T_) * (
                        contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_z_with_current[:, :, :, :, :, :, :, c, f],
                        )
                        + contract(
                            "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                            C @ G5,
                            C @ G5,
                            P_,
                            fwd_prop.data[:, :, :, :, :, :, :, a, d],
                            fwd_prop.data[:, :, :, :, :, :, :, b, e],
                            seq_prop_z_with_current[:, :, :, :, :, :, :, c, f],
                        )
                    )
    
    # 收集每个组态的结果
    proton_2pt_tmp = core.gatherLattice(proton_2pt.real.get(), [1, -1, -1, -1])
    
    # 为3pt函数重塑数组，使其可以与gatherLattice兼容
    # 先将t_src和tau维度合并处理
    proton_3pt_x_reshaped = proton_3pt_x.real.get().reshape(-1, latt_info.Lt)
    proton_3pt_y_reshaped = proton_3pt_y.real.get().reshape(-1, latt_info.Lt)
    proton_3pt_z_reshaped = proton_3pt_z.real.get().reshape(-1, latt_info.Lt)
    
    # 使用标准的4维轴参数
    proton_3pt_x_tmp = core.gatherLattice(proton_3pt_x_reshaped, [1, -1, -1, -1])
    proton_3pt_y_tmp = core.gatherLattice(proton_3pt_y_reshaped, [1, -1, -1, -1])
    proton_3pt_z_tmp = core.gatherLattice(proton_3pt_z_reshaped, [1, -1, -1, -1])
    
    if latt_info.mpi_rank == 0:
        # 恢复原始形状
        proton_3pt_x_tmp = proton_3pt_x_tmp.reshape(len(t_src_list), len(tau_list), latt_info.Lt)
        proton_3pt_y_tmp = proton_3pt_y_tmp.reshape(len(t_src_list), len(tau_list), latt_info.Lt)
        proton_3pt_z_tmp = proton_3pt_z_tmp.reshape(len(t_src_list), len(tau_list), latt_info.Lt)
        
        # 时间平移
        for t_idx, t_src in enumerate(t_src_list):
            proton_2pt_tmp[t_idx] = np.roll(proton_2pt_tmp[t_idx], -t_src, 0)
            for tau_idx, tau in enumerate(tau_list):
                proton_3pt_x_tmp[t_idx, tau_idx] = np.roll(proton_3pt_x_tmp[t_idx, tau_idx], -t_src, 0)
                proton_3pt_y_tmp[t_idx, tau_idx] = np.roll(proton_3pt_y_tmp[t_idx, tau_idx], -t_src, 0)
                proton_3pt_z_tmp[t_idx, tau_idx] = np.roll(proton_3pt_z_tmp[t_idx, tau_idx], -t_src, 0)
        
        # 在不同源位置上取平均
        twopt_proton = proton_2pt_tmp.mean(0)
        thrpt_proton_x = proton_3pt_x_tmp.mean(0)
        thrpt_proton_y = proton_3pt_y_tmp.mean(0)
        thrpt_proton_z = proton_3pt_z_tmp.mean(0)
        
        proton_2pt_conf_list.append(twopt_proton)
        proton_3pt_x_conf_list.append(thrpt_proton_x)
        proton_3pt_y_conf_list.append(thrpt_proton_y)
        proton_3pt_z_conf_list.append(thrpt_proton_z)

dirac.destroy()


# %%
print(np.shape(proton_2pt_conf_list))
print(np.shape(proton_3pt_x_conf_list))

proton_2pt_jk = jackknife(proton_2pt_conf_list)
proton_3pt_x_jk = jackknife(proton_3pt_x_conf_list)

proton_2pt_jk_avg = jk_ls_avg(proton_2pt_jk)
proton_3pt_x_jk_avg = jk_ls_avg(proton_3pt_x_jk)

print(np.shape(proton_2pt_jk_avg))
print(np.shape(proton_3pt_x_jk_avg))


tsep_ls = [6, 8, 10]

ratio_ls = []

for tsep in tsep_ls:
    ratio = proton_3pt_x_jk_avg[:,tsep] / proton_2pt_jk_avg[tsep]
    
    ratio = ratio[1:tsep]
    ratio_ls.append(ratio)
    
fig, ax = default_plot()

for i, ratio in enumerate(ratio_ls):
    ax.errorbar(np.arange(1, tsep_ls[i]) - tsep_ls[i] / 2, [gv.mean(r) for r in ratio], yerr=[gv.sdev(r) for r in ratio], **errorb)
plt.show()


# %%
# Jackknife 分析
proton_2pt_jk = jackknife(proton_2pt_conf_list)
proton_3pt_x_jk = jackknife(proton_3pt_x_conf_list)
proton_3pt_y_jk = jackknife(proton_3pt_y_conf_list)
proton_3pt_z_jk = jackknife(proton_3pt_z_conf_list)

proton_2pt_jk_avg = jk_ls_avg(proton_2pt_jk)
proton_3pt_x_jk_avg = jk_ls_avg(proton_3pt_x_jk)
proton_3pt_y_jk_avg = jk_ls_avg(proton_3pt_y_jk)
proton_3pt_z_jk_avg = jk_ls_avg(proton_3pt_z_jk)

# 计算比例 R = C3pt(t,tau) / C2pt(t)
# 只保留相关的时间位置
t_range = range(3, 12)  # 选择一个合适的时间范围
gA_x = {}
gA_y = {}
gA_z = {}

for tau_idx, tau in enumerate(tau_list):
    gA_x[tau] = []
    gA_y[tau] = []
    gA_z[tau] = []
    
    for t in t_range:
        if t > tau:  # 确保 t > tau
            ratio_x = proton_3pt_x_jk_avg[tau_idx][t] / proton_2pt_jk_avg[t]
            ratio_y = proton_3pt_y_jk_avg[tau_idx][t] / proton_2pt_jk_avg[t]
            ratio_z = proton_3pt_z_jk_avg[tau_idx][t] / proton_2pt_jk_avg[t]
            
            gA_x[tau].append(ratio_x)
            gA_y[tau].append(ratio_y)
            gA_z[tau].append(ratio_z)

# %%
# 绘制 gA 随 tau 的变化
fig, ax = default_plot()

gA_x_avg = []
gA_y_avg = []
gA_z_avg = []
gA_avg = []  # 空间平均

tau_plot = []

for tau in tau_list:
    if len(gA_x[tau]) > 0:  # 确保有数据
        # 对每个 tau 值，计算在不同 t 上的平均值
        gA_x_val = constant_fit(gv.gvar(np.mean([gv.mean(r) for r in gA_x[tau]]), 
                                         np.std([gv.mean(r) for r in gA_x[tau]])))
        gA_y_val = constant_fit(gv.gvar(np.mean([gv.mean(r) for r in gA_y[tau]]), 
                                         np.std([gv.mean(r) for r in gA_y[tau]])))
        gA_z_val = constant_fit(gv.gvar(np.mean([gv.mean(r) for r in gA_z[tau]]), 
                                         np.std([gv.mean(r) for r in gA_z[tau]])))
        
        # 三个方向的平均
        gA_val = (gA_x_val + gA_y_val + gA_z_val) / 3
        
        gA_x_avg.append(gA_x_val)
        gA_y_avg.append(gA_y_val)
        gA_z_avg.append(gA_z_val)
        gA_avg.append(gA_val)
        tau_plot.append(tau)

# 绘制每个方向和平均值
ax.errorbar(tau_plot, [gv.mean(val) for val in gA_x_avg], yerr=[gv.sdev(val) for val in gA_x_avg], 
           label=r"$g_A^x$", marker='o', **errorb)
ax.errorbar(tau_plot, [gv.mean(val) for val in gA_y_avg], yerr=[gv.sdev(val) for val in gA_y_avg], 
           label=r"$g_A^y$", marker='s', **errorb)
ax.errorbar(tau_plot, [gv.mean(val) for val in gA_z_avg], yerr=[gv.sdev(val) for val in gA_z_avg], 
           label=r"$g_A^z$", marker='^', **errorb)
ax.errorbar(tau_plot, [gv.mean(val) for val in gA_avg], yerr=[gv.sdev(val) for val in gA_avg], 
           label=r"$g_A$ (avg)", marker='*', **errorb)

# 拟合一个常数值作为最终的 gA
final_gA = constant_fit(gv.gvar([gv.mean(val) for val in gA_avg], [gv.sdev(val) for val in gA_avg]))
ax.axhline(y=gv.mean(final_gA), color='k', linestyle='-', alpha=0.5)
ax.axhspan(gv.mean(final_gA) - gv.sdev(final_gA), gv.mean(final_gA) + gv.sdev(final_gA), 
          alpha=0.2, color='gray')

ax.legend(ncol=2, **fs_small_p)
ax.set_xlabel(r"$\tau$", **fs_p)
ax.set_ylabel(r"$g_A$", **fs_p)
plt.tight_layout()
plt.savefig("../output/plots/proton_gA.pdf", transparent=True)
plt.show()

print(f"Final gA value: {final_gA}")

# %% 