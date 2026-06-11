import csv
import os
import pickle
import numpy as np
from MMI_utils.MMI_remodify import MMI
import networkx as nx
import pandas as pd
import random
from multiprocessing import Pool
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def parse_drug_target(file_path):
    drug_targets = {}


    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            drug = row['ChemicalID']
            gene = row['GeneSymbol']

            # drug = row['node_1']
            # gene = row['node_2_name']

            # drug = row['tcm_id']
            # gene = row['Gene Symbol']


            if drug in drug_targets:
                drug_targets[drug].append(gene)
            else:

                drug_targets[drug] = [gene]

    print(f'\n> Done parsing Chemical: read {len(drug_targets)} total Chemical')

    return drug_targets

def parse_disease_genes(file_path):
    disease_genes = {}

    # 打开并读取CSV文件
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

def explain_subgraph(herb_id, umls_id, drug_targets, disease_genes, network,file_path, num_workers=32):
    nodes_from = set(drug_targets.get(herb_id, [])) & set(network.nodes())
    nodes_to = set(disease_genes.get(umls_id, [])) & set(network.nodes())

    if not nodes_from or not nodes_to:
        print(f'No valid nodes found for herb_id: {herb_id} or umls_id: {umls_id}')
        return None

    all_paths = {}  # Store shortest paths for each node_from

    # Prepare tasks for parallel processing
    tasks = [(node_from, node_to, network) for node_from in nodes_from for node_to in nodes_to]

    # Parallel computation of shortest paths
    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_shortest_path, tasks)

    # Process results
    for node_from in nodes_from:
        node_results = [res for res in results if res[0] == node_from]
        if not node_results:
            continue

        min_length = min(res[1] for res in node_results if res[2] is not None)
        shortest_paths = [res[2] for res in node_results if res[1] == min_length]

        if shortest_paths:
            all_paths[node_from] = {
                'shortest_paths': shortest_paths,
                'path_length': min_length
            }
    umls_id = umls_id.replace('MESH:', '')
    # Write results to file
    output_file = f"{file_path}/{herb_id}_{umls_id}_{n}.txt"
    with open(output_file, 'w') as f:
        for node_from, data in all_paths.items():
            for path in data['shortest_paths']:
                result_str = (f"{herb_id}\t{umls_id}\t"
                              f"{' -> '.join(path)}\t{data['path_length']}\n")
                # print(result_str.strip())  # Output to console
                f.write(result_str)  # Write to file

    return all_paths

def calculate_shortest_path(args):
    node_from, node_to, network = args
    try:
        length, path = nx.single_source_dijkstra(network, source=node_from, target=node_to, weight='weight')
        return (node_from, length, path)
    except nx.NetworkXNoPath:
        return (node_from, float('inf'), None)


if __name__ == "__main__":
    disease_gene_file = r'data/Therapeutic Association Data/Chemical-Disease/CTD_genes_diseases_association_genes_20.csv'
    #disease_gene_file = r'data/Therapeutic Association Data/drug_disease/2_disease_to_protein_association_genes_20.csv'
    #disease_gene_file = r'data/Therapeutic Association Data/Herb-Symptom/S1. TCM_symptom_genes_association_genes_20.csv'
    disease_genes = parse_disease_genes(disease_gene_file)

    # drug_targets_file = r'data/ncPPI/1_drug_to_protein.csv'
    drug_targets_file = r'data/Therapeutic Association Data/Chemical-Disease/CTD_chem_gene_ixns_filtered_renew.csv'
    #drug_targets_file = r'data/Therapeutic Association Data/drug_disease/1_drug_to_protein_update.csv'
    #drug_targets_file = r'data/Therapeutic Association Data/Herb-Symptom/S4-1. HIT_herb_target_data_0412dropna_filter_ncRNA.csv'
    drug_targets = parse_drug_target(drug_targets_file)

    pairs = [(herb_id, umls_id) for herb_id in drug_targets for umls_id in disease_genes]
    print(f"[pairs] total = {len(pairs)}")


    for n in [0.3, 0.4, 0.6, 0.7, 5]:

        weights = {
            "protein_protein": 1,
            "protein_biological_function": n,
            "biological_function_biological_function": n
        }

        mmi = MMI(weights=weights)
        mmi.load()
        save_load_file_path = 'output/Path_analysis'
        mmi.save_graph(save_load_file_path)
        with open('output/Path_analysis/graph.pkl', 'rb') as file:
            G = pickle.load(file)
        components = nx.connected_components(G)
        lcclist = sorted(list(components), key=len, reverse=True)
        G1 = G.subgraph(lcclist[0]).copy()

        file_path = 'output/Path_analysis'
        file_path1 = file_path + '/CD_update'
        #file_path1 = file_path + '/DD_update'
        #file_path1 = file_path + '/HS_update'

        if not os.path.exists(file_path1):
            os.makedirs(file_path1)

        for pair in pairs:
            drug_id = pair[0]
            disease_id = pair[1]

            disease_id_clean = disease_id.replace('MESH:', '')
            output_file = os.path.join(file_path1, f"{drug_id}_{disease_id_clean}_{n}.txt")


            if os.path.exists(output_file):
                print(f"Skip existing file: {output_file}")
                continue

            explain_subgraph(
                drug_id,
                disease_id,
                drug_targets,
                disease_genes,
                G1,
                file_path1,
                num_workers=32
            )

