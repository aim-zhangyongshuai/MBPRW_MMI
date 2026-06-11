import csv
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
SCRIPT_ABS_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_ABS_PATH)
SCRIPT_DIR2 = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR2)
# ============================================================================================
# ------------------------------
# 1. Font settings (optional)
# ------------------------------
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

# ------------------------------
# 2. Helper functions
# ------------------------------
def parse_drug_target1(file_path):
    """Parse drug-target file (format: node_1, node_2_name)"""
    drug_targets = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            drug = row['node_1']
            gene = row['node_2_name']
            drug_targets.setdefault(drug, []).append(gene)
    print(f'> Done parsing drug targets: {len(drug_targets)} drugs')
    return drug_targets

def parse_drug_target2(file_path):
    """Parse chemical-target file (format: ChemicalID, GeneSymbol)"""
    drug_targets = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            drug = row['ChemicalID']
            gene = row['GeneSymbol']
            drug_targets.setdefault(drug, []).append(gene)
    print(f'> Done parsing chemical targets: {len(drug_targets)} chemicals')
    return drug_targets

def parse_drug_target3(file_path):
    """Parse TCM herb-target file (format: tcm_id, Gene Symbol)"""
    drug_targets = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            drug = row['tcm_id']
            gene = row['Gene Symbol']
            drug_targets.setdefault(drug, []).append(gene)
    print(f'> Done parsing herb targets: {len(drug_targets)} herbs')
    return drug_targets

def parse_disease_genes1(file_path):
    """Parse disease-gene file (format: node_1, node_2_name)"""
    disease_genes = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['node_1']
            gene = row['node_2_name']
            disease_genes.setdefault(disease, []).append(gene)
    print(f'> Done parsing disease genes: {len(disease_genes)} diseases')
    return disease_genes

def parse_disease_genes2(file_path):
    """Parse chemical-disease gene file (format: DiseaseID, GeneSymbol)"""
    disease_genes = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['DiseaseID']
            gene = row['GeneSymbol']
            disease_genes.setdefault(disease, []).append(gene)
    print(f'> Done parsing chemical-disease genes: {len(disease_genes)} diseases')
    return disease_genes

def parse_disease_genes3(file_path):
    """Parse TCM symptom-gene file (format: Symptom, Symbol)"""
    disease_genes = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['Symptom']
            gene = row['Symbol']
            disease_genes.setdefault(disease, []).append(gene)
    print(f'> Done parsing symptom genes: {len(disease_genes)} symptoms')
    return disease_genes

def plot_violin_box_distribution(data1, data2, data3, title, ylabel, output_path, custom_labels=None):
    """Generate violin + box plot with mean points."""
    # Clean data
    data1 = np.array(data1, dtype=float)
    data2 = np.array(data2, dtype=float)
    data3 = np.array(data3, dtype=float)
    data1 = data1[~np.isnan(data1) & (data1 > 0)]
    data2 = data2[~np.isnan(data2) & (data2 > 0)]
    data3 = data3[~np.isnan(data3) & (data3 > 0)]

    log_data1 = np.log10(data1)
    log_data2 = np.log10(data2)
    log_data3 = np.log10(data3)

    mean1, median1 = np.mean(data1), np.median(data1)
    mean2, median2 = np.mean(data2), np.median(data2)
    mean3, median3 = np.mean(data3), np.median(data3)
    log_mean1, log_mean2, log_mean3 = np.log10(mean1), np.log10(mean2), np.log10(mean3)

    fig, ax = plt.subplots(figsize=(8, 5))
    data_list = [log_data1, log_data2, log_data3]
    positions = [1, 2, 3]
    labels = custom_labels if custom_labels else ["Drug", "Chemical", "TCM herb"]
    colors = ["skyblue", "orange", "seagreen"]

    # Violin plot
    violin = ax.violinplot(data_list, positions=positions, widths=0.85,
                           showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(violin["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        body.set_linewidth(1.0)

    # Box plot overlay
    ax.boxplot(data_list, positions=positions, widths=0.22, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
               medianprops=dict(color="black", linewidth=1.8),
               whiskerprops=dict(color="black", linewidth=1.2),
               capprops=dict(color="black", linewidth=1.2))

    # Mean points
    ax.scatter(positions, [log_mean1, log_mean2, log_mean3], marker="D", s=45,
               color="black", zorder=4, label="Mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Set y-axis ticks to original scale (1,10,100,...) on log10 axis
    all_vals = np.concatenate(data_list)
    y_min = np.floor(np.min(all_vals) * 10) / 10
    y_max = np.ceil(np.max(all_vals) * 10) / 10
    candidate_ticks = np.array([0, 1, 2, 3, 4])
    candidate_labels = ["1", "10", "100", "1000", "10000"]
    valid_idx = (candidate_ticks >= y_min - 1e-8) & (candidate_ticks <= y_max + 1e-8)
    ax.set_yticks(candidate_ticks[valid_idx])
    ax.set_yticklabels(np.array(candidate_labels)[valid_idx])
    ax.set_ylim(bottom=y_min - 0.05, top=y_max + 0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

    # Print statistics
    print(f"\n{title}")
    print(f"Drug group: mean={mean1:.1f}, median={median1:.1f}")
    print(f"Chemical group: mean={mean2:.1f}, median={median2:.1f}")
    print(f"TCM group: mean={mean3:.1f}, median={median3:.1f}")

    # Statistical tests
    u12, p12 = mannwhitneyu(data1, data2, alternative="two-sided")
    u13, p13 = mannwhitneyu(data1, data3, alternative="two-sided")
    u23, p23 = mannwhitneyu(data2, data3, alternative="two-sided")
    ks12, pks12 = ks_2samp(data1, data2)
    ks13, pks13 = ks_2samp(data1, data3)
    ks23, pks23 = ks_2samp(data2, data3)

    def sig(p):
        return "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print("\nMann-Whitney U test:")
    print(f"drug vs chemical: U={u12:.2f}, p={p12:.2e} {sig(p12)}")
    print(f"drug vs TCM:      U={u13:.2f}, p={p13:.2e} {sig(p13)}")
    print(f"chemical vs TCM:  U={u23:.2f}, p={p23:.2e} {sig(p23)}")
    print("\nKolmogorov-Smirnov test:")
    print(f"drug vs chemical: D={ks12:.3f}, p={pks12:.2e} {sig(pks12)}")
    print(f"drug vs TCM:      D={ks13:.3f}, p={pks13:.2e} {sig(pks13)}")
    print(f"chemical vs TCM:  D={ks23:.3f}, p={pks23:.2e} {sig(pks23)}")

# ------------------------------
# 3. Main （所有路径已替换为自适应相对路径）
# ------------------------------
if __name__ == "__main__":
    # Load MMI graph (largest connected component)
    graph_path = os.path.join(PROJECT_ROOT, "output", "graph_MSI.pkl")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    # Keep largest connected component
    lcc = max(nx.connected_components(G), key=len)
    G1 = G.subgraph(lcc).copy()

    # ---------- Target degrees (Figure 1E) ----------
    drug_target_file = os.path.join(PROJECT_ROOT, "data", "drug_disease", "drug_disease", "1_drug_to_protein_update.csv")
    chem_target_file = os.path.join(PROJECT_ROOT, "data", "drug_disease", "Chemical-Disease", "CTD_chem_gene_ixns_filtered_renew.csv")
    herb_target_file = os.path.join(PROJECT_ROOT, "data", "drug_disease", "Herb-Symptom", "S4-1. HIT_herb_target_data_0412dropna_filter_ncRNA.csv")

    drug_targets = parse_drug_target1(drug_target_file)
    chem_targets = parse_drug_target2(chem_target_file)
    herb_targets = parse_drug_target3(herb_target_file)

    drug_genes = set().union(*drug_targets.values())
    chem_genes = set().union(*chem_targets.values())
    herb_genes = set().union(*herb_targets.values())

    drug_deg = [G1.degree(g) for g in drug_genes if g in G1]
    chem_deg = [G1.degree(g) for g in chem_genes if g in G1]
    herb_deg = [G1.degree(g) for g in herb_genes if g in G1]

    plot_violin_box_distribution(
        drug_deg, chem_deg, herb_deg,
        title="Distribution of target degrees",
        ylabel="Degree",
        output_path=os.path.join(PROJECT_ROOT, "output", "graph", "target_degree_distribution.png"),
        custom_labels=["drug", "chemical", "TCM herb"]
    )

    # ---------- Disease-associated protein degrees (Figure 1C) ----------
    disease_file1 = os.path.join(PROJECT_ROOT, "data", "drug_disease", "drug_disease", "2_disease_to_protein_association_genes_20.csv")
    disease_file2 = os.path.join(PROJECT_ROOT, "data", "drug_disease", "Chemical-Disease", "CTD_genes_diseases_association_genes_20.csv")
    disease_file3 = os.path.join(PROJECT_ROOT, "data", "drug_disease", "Herb-Symptom", "S1. TCM_symptom_genes_association_genes_20.csv")

    disease_genes1 = parse_disease_genes1(disease_file1)
    disease_genes2 = parse_disease_genes2(disease_file2)
    disease_genes3 = parse_disease_genes3(disease_file3)

    disease_prots1 = set().union(*disease_genes1.values())
    disease_prots2 = set().union(*disease_genes2.values())
    disease_prots3 = set().union(*disease_genes3.values())

    disease_deg1 = [G1.degree(g) for g in disease_prots1 if g in G1]
    disease_deg2 = [G1.degree(g) for g in disease_prots2 if g in G1]
    disease_deg3 = [G1.degree(g) for g in disease_prots3 if g in G1]

    plot_violin_box_distribution(
        disease_deg1, disease_deg2, disease_deg3,
        title="Distribution of disease-associated protein degrees",
        ylabel="Degree",
        output_path=os.path.join(PROJECT_ROOT, "output", "graph", "disease_protein_degree_distribution.png"),
        custom_labels=["DD", "CD", "HS"]
    )