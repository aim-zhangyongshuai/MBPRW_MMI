
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
scripts_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(scripts_dir)
print(f"Working directory: {os.getcwd()}")

# ------------------------------------------------------------
# 2. Font settings
# ------------------------------------------------------------
simhei_path = r"C:\Windows\Fonts\simhei.ttf"

if os.path.exists(simhei_path):
    font_manager.fontManager.addfont(simhei_path)
    simhei_name = font_manager.FontProperties(fname=simhei_path).get_name()
    mpl.rcParams["font.family"] = ["Times New Roman", simhei_name]
else:
    mpl.rcParams["font.family"] = ["Times New Roman"]

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 19
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 14

# ------------------------------------------------------------
# 3. Read data (matched to your project directory)
# ------------------------------------------------------------

counts_dir = os.path.join(project_root, "output", "Number of drug targets")
file1 = os.path.join(counts_dir, "Drug_protein_in_graph_counts.csv")
file2 = os.path.join(counts_dir, "CTD_protein_in_graph_counts.csv")
file3 = os.path.join(counts_dir, "TCM_protein_in_graph_counts.csv")

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

target_num1 = df1["gene_num_in_graph"].dropna().astype(float)
target_num2 = df2["gene_num_in_graph"].dropna().astype(float)
target_num3 = df3["gene_num_in_graph"].dropna().astype(float)

# Keep positive values for log scale
target_num1 = target_num1[target_num1 > 0]
target_num2 = target_num2[target_num2 > 0]
target_num3 = target_num3[target_num3 > 0]

log1 = np.log10(target_num1)
log2 = np.log10(target_num2)
log3 = np.log10(target_num3)

mean1, med1 = np.mean(target_num1), np.median(target_num1)
mean2, med2 = np.mean(target_num2), np.median(target_num2)
mean3, med3 = np.mean(target_num3), np.median(target_num3)
log_mean1, log_mean2, log_mean3 = np.log10(mean1), np.log10(mean2), np.log10(mean3)

# ------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,5))
positions = [1,2,3]
labels = ["drug", "chemical", "TCM herb"]
colors = ["skyblue", "orange", "seagreen"]

violin = ax.violinplot([log1, log2, log3], positions=positions, widths=0.85,
                       showmeans=False, showmedians=False, showextrema=False)
for i, body in enumerate(violin["bodies"]):
    body.set_facecolor(colors[i])
    body.set_edgecolor("black")
    body.set_alpha(0.35)
    body.set_linewidth(1.0)

ax.boxplot([log1, log2, log3], positions=positions, widths=0.22, patch_artist=True, showfliers=False,
           boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
           medianprops=dict(color="black", linewidth=1.8),
           whiskerprops=dict(color="black", linewidth=1.2),
           capprops=dict(color="black", linewidth=1.2))

ax.scatter(positions, [log_mean1, log_mean2, log_mean3], marker="D", s=45, color="black", zorder=4, label="Mean")

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_title("Distribution of target numbers")

# Y axis: show original counts on log10 positions
y_ticks = [0, 1, 2, 3]
y_labels = ["1", "10", "100", "1000"]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_ylabel("Number of targets")

all_log = np.concatenate([log1, log2, log3])
y_max = np.ceil(np.max(all_log))
ax.set_ylim(bottom=-0.05, top=y_max+0.2)

plt.tight_layout()

out_path = os.path.join(project_root, "output", "graph", "target_number_distribution.png")
plt.savefig(out_path, dpi=600)
plt.show()

# ------------------------------------------------------------
# 5. Print statistics
# ------------------------------------------------------------
def sig_label(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else "ns"

u12, p12 = mannwhitneyu(target_num1, target_num2, alternative="two-sided")
u13, p13 = mannwhitneyu(target_num1, target_num3, alternative="two-sided")
u23, p23 = mannwhitneyu(target_num2, target_num3, alternative="two-sided")
ks12, pks12 = ks_2samp(target_num1, target_num2)
ks13, pks13 = ks_2samp(target_num1, target_num3)
ks23, pks23 = ks_2samp(target_num2, target_num3)

print("\nMann–Whitney U Test Results:")
print(f"drug vs chemical: U={u12:.4f}, p={p12:.4e} {sig_label(p12)}")
print(f"drug vs TCM:      U={u13:.4f}, p={p13:.4e} {sig_label(p13)}")
print(f"chemical vs TCM:  U={u23:.4f}, p={p23:.4e} {sig_label(p23)}")
print("\nKolmogorov–Smirnov Test Results:")
print(f"drug vs chemical: D={ks12:.4f}, p={pks12:.4e} {sig_label(pks12)}")
print(f"drug vs TCM:      D={ks13:.4f}, p={pks13:.4e} {sig_label(pks13)}")
print(f"chemical vs TCM:  D={ks23:.4f}, p={pks23:.4e} {sig_label(pks23)}")