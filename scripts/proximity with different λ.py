import pandas as pd
import networkx as nx
import os
import pickle
import time
from MMI_utils.MMI_remodify import MMI
from joblib import Parallel, delayed
import csv
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_drug_target(file_path):
    """
    Parse chemical-drug target CSV file into a dictionary: {chemical_id: [gene1, gene2, ...]}
    """
    drug_targets = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            drug = row['ChemicalID']
            gene = row['GeneSymbol']
            if drug in drug_targets:
                drug_targets[drug].append(gene)
            else:
                drug_targets[drug] = [gene]
    print(f'\n> Done parsing Chemical: read {len(drug_targets)} total Chemical')
    return drug_targets


def parse_disease_genes(file_path):
    """
    Parse disease-gene CSV file into a dictionary: {disease_id: [gene1, gene2, ...]}
    """
    disease_genes = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            disease = row['DiseaseID']
            gene = row['GeneSymbol']
            # disease = row['node_1']
            # gene = row['node_2_name']

            # disease = row['Symptom']
            # gene = row['Symbol']
            if disease in disease_genes:
                disease_genes[disease].append(gene)
            else:
                disease_genes[disease] = [gene]
    print(f'\n> Done parsing disease genes: read {len(disease_genes)} total diseases')
    return disease_genes


def single_proximity(sample, adj_matrix, node_to_index):
    """
    Calculate proximity z-score for one drug-disease pair
    sample: (drug_key, disease_key, drug_targets, disease_genes, network, nsims)
    """
    drug_key, disease_key, drug_targets, disease_genes, network, nsims = sample

    gene = set(str(g) for g in disease_genes[disease_key])
    nodes_from = set(drug_targets[drug_key]) & set(network.nodes())
    nodes_to = gene & set(network.nodes())

    if len(nodes_from) > 0 and len(nodes_to) > 0:
        # MMI object is inside loop now, or passed in
        ppi = MMI()
        d, z, (mean, sd) = ppi.calculate_proximity_with_weights(
            adj_matrix, node_to_index, network, nodes_from, nodes_to,
            n_random=nsims, min_bin_size=100, seed=None
        )
        return {
            'DrugID': drug_key,
            'DiseaseID': disease_key,
            'd': d,
            'z': z,
            'mean': mean,
            'std': sd
        }
    return None


def batch_writer(results, results_file):
    """Write batch results to CSV (append mode)"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)


def batch_single_proximity(samples, adj_matrix, node_to_index):
    """Process a batch of samples and return list of results"""
    batch_results = []
    for sample in samples:
        res = single_proximity(sample, adj_matrix, node_to_index)
        if res:
            batch_results.append(res)
    return batch_results


def process_and_write_batch(batch, adj_matrix, node_to_index, results_file):
    """Wrapper for parallel: process batch and write to file"""
    batch_results = batch_single_proximity(batch, adj_matrix, node_to_index)
    batch_writer(batch_results, results_file)


if __name__ == "__main__":
    # ---------------------- Input Paths ----------------------
    disease_gene_file = r'data/drug_disease/Chemical-Disease/CTD_genes_diseases_association_genes_20.csv'
    # disease_gene_file = r'data/drug_disease/drug_disease/2_disease_to_protein_association_genes_20.csv'
    # disease_gene_file = r'S1. TCM_symptom_genes_association_genes_20.csv'
    disease_genes = parse_disease_genes(disease_gene_file)

    drug_targets_file = r'data/drug_disease/Chemical-Disease/CTD_chem_gene_ixns_filtered_renew.csv'
    # drug_targets_file = r'data/drug_disease/drug_disease/1_drug_to_protein_update.csv'
    # drug_targets_file = r'S4-1. HIT_herb_target_data_0412dropna_filter_ncRNA.csv'
    drug_targets = parse_drug_target(drug_targets_file)

    # Load PPI network (no GO for drug-disease binning)
    with open("output/graph_PPI.pkl", "rb") as f:
        G = pickle.load(f)
    lcclist = sorted(nx.connected_components(G), key=len, reverse=True)
    G1 = G.subgraph(lcclist[0]).copy()

    nsims = 100
    for n in [0.1, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.7, 5]:
        # Generate drug-disease pairs
        samples = [(drug, dis, drug_targets, disease_genes, G1, nsims)
                   for drug in drug_targets for dis in disease_genes]
        print("Number of samples:", len(samples))

        # Load precomputed shortest-path matrix for current lambda
        with open(f"output/matrices/lambda_{n}/adj_matrix_1_{n}_{n}.pkl", "rb") as f:
            adj_matrix = pickle.load(f)
        with open(f"output/matrices/lambda_{n}/node_to_index_1_{n}_{n}.pkl", "rb") as f:
            node_to_index = pickle.load(f)

        # Output path
        results_file = f"output/proximity of different λ/Chemical-Disease/CTD_λ_{n}.csv"
        # results_file = f"output/proximity of different λ/drug_disease/drug_λ_{n}.csv"
        # results_file = f"output/proximity of different λ/Herb-Symptom/TCM_λ_{n}.csv"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        batch_size = 7000
        batch_samples = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

        s = time.time()
        # Parallel process batches
        Parallel(n_jobs=16)(
            delayed(process_and_write_batch)(batch, adj_matrix, node_to_index, results_file)
            for batch in batch_samples
        )
        e = time.time() - s
        print('Finished in %f hours' % (e / 3600))