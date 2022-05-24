import matplotlib.pyplot as plt
import numpy as np
import pickle

with open(f'LTSvsDOSLB.pickle', 'rb') as file2:
     obj_regret_DOLB, obj_regret_LTS, cst_regret_DOLB, cst_regret_LTS, BIS0, BIS1 = pickle.load(file2)


T = 10000 # time horizon
ts = 1 + np.arange(T)
epsilon = 0.01
d = 2
alpha_safety = 1.1
theta_star = np.array([-0.1, -1.0])
zeta_star = np.array([0, 1])
S = np.linalg.norm(theta_star)
R = 0.1
L = 4
Lambda = 1
delta = 0.01
Xi = 0.14 # adjust Xi according to alpha_safety

# plot the 2 figures of LTS vs DOSLB
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax= plt.subplots()
ax.plot(ts, np.cumsum(obj_regret_DOLB.mean(axis=0)), 'g-', label = 'DOSLB Efficacy Regret')
ax.fill_between(ts, np.cumsum(obj_regret_DOLB.mean(axis=0)) - np.cumsum(obj_regret_DOLB.std(axis=0)), np.cumsum(obj_regret_DOLB.mean(axis=0)) + np.cumsum(obj_regret_DOLB.std(axis=0)), color='green', alpha=0.2)
ax.plot(ts, np.cumsum(obj_regret_LTS.mean(axis=0)), 'b--', label = 'LTS Efficacy Regret')
ax.fill_between(ts, np.cumsum(obj_regret_LTS.mean(axis=0)) - np.cumsum(obj_regret_LTS.std(axis=0)), np.cumsum(obj_regret_LTS.mean(axis=0)) + np.cumsum(obj_regret_LTS.std(axis=0)), color='blue', alpha=0.2)
plt.legend(loc='best')
plt.savefig("Figs/DOvsLTSeff.pdf", bbox_inches='tight')

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax= plt.subplots()
ax.plot(ts, np.cumsum(cst_regret_DOLB.mean(axis=0)), 'g-', label = 'DOSLB Safety Regret')
ax.fill_between(ts, np.cumsum(cst_regret_DOLB.mean(axis=0)) - np.cumsum(cst_regret_DOLB.std(axis=0)), np.cumsum(cst_regret_DOLB.mean(axis=0)) + np.cumsum(cst_regret_DOLB.std(axis=0)), color='green', alpha=0.2)
ax.plot(ts, np.cumsum(cst_regret_LTS.mean(axis=0)), 'b--', label = 'LTS Safety Regret')
ax.fill_between(ts, np.cumsum(cst_regret_LTS.mean(axis=0)) - np.cumsum(cst_regret_LTS.std(axis=0)), np.cumsum(cst_regret_LTS.mean(axis=0)) + np.cumsum(cst_regret_LTS.std(axis=0)), color='blue', alpha=0.2)
plt.legend(loc='best')
plt.savefig("Figs/DOvsLTSsaf.pdf", bbox_inches='tight')

# plot the regret
regret = np.maximum(obj_regret_DOLB, cst_regret_DOLB)
slacked_regret = np.maximum(obj_regret_DOLB, cst_regret_DOLB - epsilon)
upper_bound = 4 * R * d * np.sqrt(ts * d * np.log(1+ts/Lambda/d)) * (np.sqrt(Lambda) * S + np.sqrt(0.5 * np.log(2/delta) + d/4 * np.log(1+ ts/Lambda/d)))
log_bound = 8 * d * np.log(1 + ts / d / Lambda) * (np.sqrt(Lambda) * S + R * np.sqrt(0.5 * np.log(2 / delta) + d / 4 * np.log(1 + ts / d / Lambda))) ** 2 * (1 / Xi + 1 / epsilon)

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax = plt.subplots()
ax.plot(np.cumsum(regret.mean(axis=0)), 'k-', label='Regret')
ax.fill_between(ts, np.cumsum(regret.mean(axis=0)) - np.cumsum((regret.std(axis=0))), np.cumsum(regret.mean(axis=0)) + np.cumsum((regret.std(axis=0))), color='black', alpha=0.2)
ax.plot(np.cumsum(slacked_regret.mean(axis=0)), 'b:', label='Slacked Regret')
ax.fill_between(ts, np.cumsum(slacked_regret.mean(axis=0)) - np.cumsum((slacked_regret.std(axis=0))), np.cumsum(slacked_regret.mean(axis=0)) + np.cumsum((slacked_regret.std(axis=0))), color='blue', alpha=0.2)
ax.plot(upper_bound, 'r--', label='Upper Bound: Thm. 4.1')
#ax.plot(log_bound, 'y-.', label='Upper Bound: Thm. 5.14')
plt.legend(loc='best')
plt.savefig("Figs/3regret.pdf", bbox_inches='tight')

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax = plt.subplots()
ax.plot(np.cumsum(regret.mean(axis=0)), 'k-', label='Regret')
ax.fill_between(ts, np.cumsum(regret.mean(axis=0)) - np.cumsum((regret.std(axis=0))), np.cumsum(regret.mean(axis=0)) + np.cumsum((regret.std(axis=0))), color='black', alpha=0.2)
ax.plot(np.cumsum(slacked_regret.mean(axis=0)), 'b:', label='Slacked Regret')
ax.fill_between(ts, np.cumsum(slacked_regret.mean(axis=0)) - np.cumsum((slacked_regret.std(axis=0))), np.cumsum(slacked_regret.mean(axis=0)) + np.cumsum((slacked_regret.std(axis=0))), color='blue', alpha=0.2)
ax.plot(upper_bound, 'r--', label='Upper Bound: Thm. 4.1')
ax.plot(log_bound, 'y-.', label='Upper Bound: Thm. 5.14')
plt.legend(loc='best')
plt.savefig("Figs/4regret.pdf", bbox_inches='tight')

# plot the BIS behavior
ind_plt = np.logical_or(BIS0, BIS1)
ind_int = 1 - ind_plt.astype(int) # number of sub optimal bis played

BIS_bound = 8 * d * np.log(1 + ts /Lambda /d) * (np.sqrt(Lambda) * S + R * np.sqrt(0.5 * np.log(2/delta)+ d/4 * np.log(1+ ts /Lambda /d))) ** 2 /Xi/Xi
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax = plt.subplots()
ax.plot(np.cumsum(ind_int.mean(axis=0)), 'k-', label = 'Number of Non-optimal BIS Played')
ax.fill_between(ts, np.cumsum(ind_int.mean(axis=0)) - np.cumsum((ind_int.std(axis=0))), np.cumsum(ind_int.mean(axis=0)) + np.cumsum((ind_int.std(axis=0))), color='black', alpha=0.2)
ax.plot(BIS_bound, 'r--', label = 'Upper Bound: Thm 5.12')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig("Figs/BIS_ylog.pdf", bbox_inches='tight')

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax = plt.subplots()
ax.plot(np.cumsum(ind_int.mean(axis=0)), 'k-', label = 'Number of Non-optimal BIS Played')
ax.fill_between(ts, np.cumsum(ind_int.mean(axis=0)) - np.cumsum((ind_int.std(axis=0))), np.cumsum(ind_int.mean(axis=0)) + np.cumsum((ind_int.std(axis=0))), color='black', alpha=0.2)
ax.plot(BIS_bound, 'r--', label = 'Upper Bound')
plt.legend(loc='best')
plt.savefig("Figs/BIS.pdf", bbox_inches='tight')

# plot L1 vs L infty
with open(f'LinftyvsL1.pickle', 'rb') as file2:
     obj_regret_Linfty, obj_regret_L1, cst_regret_Linfty, cst_regret_L1, BIS0, BIS1, BIS0_i, BIS1_i = pickle.load(file2)

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax= plt.subplots()
ax.plot(ts, np.cumsum(obj_regret_Linfty.mean(axis=0)), 'g-', label = 'Linfty Efficacy Regret')
ax.fill_between(ts, np.cumsum(obj_regret_Linfty.mean(axis=0)) - np.cumsum(obj_regret_Linfty.std(axis=0)), np.cumsum(obj_regret_Linfty.mean(axis=0)) + np.cumsum(obj_regret_Linfty.std(axis=0)), color='green', alpha=0.2)
ax.plot(ts, np.cumsum(obj_regret_L1.mean(axis=0)), 'b--', label = 'L1 Efficacy Regret')
ax.fill_between(ts, np.cumsum(obj_regret_L1.mean(axis=0)) - np.cumsum(obj_regret_L1.std(axis=0)), np.cumsum(obj_regret_L1.mean(axis=0)) + np.cumsum(obj_regret_L1.std(axis=0)), color='blue', alpha=0.2)
plt.legend(loc='best')
plt.savefig("Figs/LinftyvsL1eff.pdf", bbox_inches='tight')

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 3
fig, ax= plt.subplots()
ax.plot(ts, np.cumsum(cst_regret_Linfty.mean(axis=0)), 'g-', label = 'Linfty Safety Regret')
ax.fill_between(ts, np.cumsum(cst_regret_Linfty.mean(axis=0)) - np.cumsum(cst_regret_Linfty.std(axis=0)), np.cumsum(cst_regret_Linfty.mean(axis=0)) + np.cumsum(cst_regret_Linfty.std(axis=0)), color='green', alpha=0.2)
ax.plot(ts, np.cumsum(cst_regret_L1.mean(axis=0)), 'b--', label = 'L1 Safety Regret')
ax.fill_between(ts, np.cumsum(cst_regret_L1.mean(axis=0)) - np.cumsum(cst_regret_L1.std(axis=0)), np.cumsum(cst_regret_L1.mean(axis=0)) + np.cumsum(cst_regret_L1.std(axis=0)), color='blue', alpha=0.2)
plt.legend(loc='best')
plt.savefig("Figs/LinftyvsL1saf.pdf", bbox_inches='tight')
