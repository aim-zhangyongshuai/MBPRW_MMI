import os
import pickle
import numpy as np
import csv
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_pickle(file_path):
    """Load data from a pickle file"""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def get_protein_indices_from_graph(graph, node_to_index):
    """
    Extract nodes with type='protein' from the graph,
    and map them to corresponding matrix indices from node_to_index.
    """
    protein_nodes = [
        node for node, attr in graph.nodes(data=True)
        if attr.get('type') == 'protein'
    ]

    protein_indices = []
    missing_nodes = []

    for node in protein_nodes:
        if node in node_to_index:
            protein_indices.append(node_to_index[node])
        else:
            missing_nodes.append(node)

    print(f'Number of protein nodes in graph: {len(protein_nodes)}')
    print(f'Number of protein nodes matched in matrix: {len(protein_indices)}')
    print(f'Number of unmatched nodes: {len(missing_nodes)}')
    if missing_nodes:
        print('First 10 unmatched nodes:', missing_nodes[:10])

    return protein_indices


def extract_submatrix(matrix, indices):
    """Extract submatrix corresponding to given indices"""
    return matrix[np.ix_(indices, indices)]


def compare_matrices_upper(matrix1, matrix2, exclude_diagonal=True):
    """
    Compare only upper triangle to avoid duplicate checks for symmetric matrices.
    Round to 1 decimal place and multiply by 10 to eliminate floating-point errors.
    """
    matrix1_processed = np.round(matrix1, 1)
    matrix2_processed = np.round(matrix2, 1)

    # Convert to integers to avoid floating-point errors (e.g., 1.3 vs 1.2999999)
    matrix1_int = np.rint(matrix1_processed * 10).astype(np.int32)
    matrix2_int = np.rint(matrix2_processed * 10).astype(np.int32)

    k = 1 if exclude_diagonal else 0
    triu_idx = np.triu_indices(matrix1.shape[0], k=k)

    same_count = np.sum(matrix1_int[triu_idx] == matrix2_int[triu_idx])
    different_count = np.sum(matrix1_int[triu_idx] != matrix2_int[triu_idx])
    r = different_count / same_count if same_count != 0 else float('inf')

    return same_count, different_count, r


# =======================
# 1. Load graph and node-to-index mapping
# =======================
graph_path = r'output/graph_PPI.pkl'
# ------------------------------------------------------------
# Load a node-to-index mapping from any lambda (here lambda=0.4)
# The mapping is identical for all lambdas because protein nodes
# are sorted alphabetically when building each matrix.
# So we can safely use this one as a reference to reorder rows/cols
# consistently across all matrices.
node_to_index_path = r'output/matrices/lambda_0.4/node_to_index_1_0.4_0.4.pkl'

graph = load_pickle(graph_path)
node_to_index = load_pickle(node_to_index_path)

protein_indices = get_protein_indices_from_graph(graph, node_to_index)

# =======================
# 2. Parameter settings
# =======================
results = []
n_values = [0.1, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.7, 5]

# Baseline matrix (lambda = 5)
base_file_path = r'output/matrices/lambda_5/adj_matrix_1_5_5.pkl'
base_matrix = load_pickle(base_file_path)

# Extract protein-only submatrix
base_protein_matrix = extract_submatrix(base_matrix, protein_indices)

# =======================
# 3. Compare matrix changes across different lambda values
# =======================
for n in n_values:
    file_path1 = f'output/matrices/lambda_{n}/adj_matrix_1_{n}_{n}.pkl'
    matrix1 = load_pickle(file_path1)

    protein_matrix1 = extract_submatrix(matrix1, protein_indices)

    same_count, different_count, r = compare_matrices_upper(
        protein_matrix1,
        base_protein_matrix,
        exclude_diagonal=True
    )

    results.append([n, r, same_count, different_count])

    print(f'Lambda={n}: same={same_count}, diff={different_count}, r={r}')

# =======================
# 4. Save comparison results
# =======================
output_file = r'output/compare_matrix_result.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['n', 'r', 'same_count', 'diff_count'])
    writer.writerows(results)

print(f'Results saved to: {output_file}')