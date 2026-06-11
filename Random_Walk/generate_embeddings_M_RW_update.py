import argparse
import os
import math
import random
import pickle
import networkx as nx
import numpy as np
from scipy import stats
import parmap
from collections import defaultdict, Counter
import multiprocessing
from .HeterogeneousSG import HeterogeneousSG
from .utils import read_graph, set_seed

from typing import List, Dict, Optional
from functools import partial


def read_node_types(nodetype_file: str) -> Dict[str, str]:
    node_types = {}
    with open(nodetype_file, 'r', encoding='utf-8') as f:
        header = next(f, None)  # Skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            node, ntype = parts[0], parts[1]
            node_types[node] = ntype
    return node_types


def multi_metapath_walk(
    G: nx.Graph,
    node_types: Dict[str, str],
    start_node: str,
    metapaths: List[List[str]],
    walk_length: int,
    metapath_weights: Optional[List[float]] = None,  # NEW: Weight/probability for each metapath
    max_retry_per_step: int = 50
) -> List[str]:
    """
    Generate a walk from start_node in a metapath-guided manner.
    - metapaths: e.g., [['drug','gene','gene','disease'], ...]
    - metapath_weights: Weights corresponding to metapaths (relative probability, no normalization required)
    """
    if metapath_weights is None:
        metapath_weights = [1.0] * len(metapaths)
    if len(metapath_weights) != len(metapaths):
        raise ValueError(
            f"Length of metapath_weights ({len(metapath_weights)}) must equal length of metapaths ({len(metapaths)})"
        )

    walk = [start_node]
    current_node = start_node
    current_type = node_types.get(current_node)

    while len(walk) < walk_length:
        # Find metapaths starting with current type and sync weights
        candidate_paths = []
        candidate_weights = []
        for mp, w in zip(metapaths, metapath_weights):
            if mp and mp[0] == current_type and w > 0:
                candidate_paths.append(mp)
                candidate_weights.append(float(w))

        if not candidate_paths:
            break  # No available metapath for current type

        retry_count = 0
        success = False

        while retry_count < max_retry_per_step:
            # NEW: Select metapath by weight (not equal probability random.choice)
            metapath = random.choices(candidate_paths, weights=candidate_weights, k=1)[0]

            node = current_node
            path_nodes = [node]
            metapath_success = True

            for next_type in metapath[1:]:
                neighbors = list(G.neighbors(node))
                valid_neighbors = [n for n in neighbors if node_types.get(n) == next_type]
                if not valid_neighbors:
                    metapath_success = False
                    break
                node = random.choice(valid_neighbors)
                path_nodes.append(node)

            if metapath_success:
                walk.extend(path_nodes[1:])          # Remove duplicate start node
                walk = walk[:walk_length]            # Truncate to walk_length
                current_node = path_nodes[-1]
                current_type = node_types.get(current_node)
                success = True
                break

            retry_count += 1

        if not success:
            break

    return walk


def _generate_single_walk_worker(args, G, node_types, metapaths, metapath_weights, walk_length):
    """Worker for processing a single walk task"""
    node, task_seed = args
    random.seed(task_seed)

    walk = multi_metapath_walk(
        G=G,
        node_types=node_types,
        start_node=node,
        metapaths=metapaths,
        metapath_weights=metapath_weights,  # NEW
        walk_length=walk_length,
        max_retry_per_step=50
    )
    return walk


def generate_metapath_walks(
    G: nx.Graph,
    node_types: Dict[str, str],
    metapaths: List[List[str]],
    walk_length: int,
    num_walks: int,
    metapath_weights: Optional[List[float]] = None,  #  NEW
    seed: int = 43,
    workers: int = os.cpu_count()
) -> List[List[str]]:
    """
    Generate metapath-guided walks in parallel
    """
    if metapath_weights is None:
        metapath_weights = [1.0] * len(metapaths)
    if len(metapath_weights) != len(metapaths):
        raise ValueError(
            f"Length of metapath_weights ({len(metapath_weights)}) must equal length of metapaths ({len(metapaths)})"
        )

    random.seed(seed)

    # Only select drug or disease nodes as start points (original logic)
    start_nodes = [n for n in G.nodes() if node_types.get(n) in ['drug', 'disease']]

    # Task list: (node, task_seed)
    tasks = []
    for node_idx, node in enumerate(start_nodes):
        for walk_idx in range(num_walks):
            task_seed = seed + node_idx * num_walks + walk_idx
            tasks.append((node, task_seed))

    worker_func = partial(
        _generate_single_walk_worker,
        G=G,
        node_types=node_types,
        metapaths=metapaths,
        metapath_weights=metapath_weights,  #  NEW
        walk_length=walk_length
    )

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_func, tasks)

    walks = [walk for walk in results if len(walk) > 1]
    return walks


def save_embedding_files(
    netf: str,
    outputf: str,
    nodetypef: str = None,
    seed: int = 43,
    directed: bool = False,
    weighted: bool = True,
    num_walks: int = 100,
    walk_length: int = 5,
    workers: int = os.cpu_count(),
    dimension: int = 128,
    window_size: int = 4,
    p: float = 1,
    q: float = 1,
    net_delimiter: str = '\t',
    metapaths: Optional[List[List[str]]] = None,         #  NEW: Allow passing metapaths externally
    metapath_weights: Optional[List[float]] = None       # NEW: Allow passing weights externally
):
    set_seed(seed)

    print('Reading network files...')
    G = read_graph(netf, weighted=weighted, directed=directed, delimiter=net_delimiter)

    node_types = read_node_types(nodetypef)

    # =========================
    # Default metapaths (original PPI long chains)
    # You can pass metapaths/metapath_weights externally to switch between Western medicine/Compound/TCM schemes
    # =========================
    if metapaths is None:
        metapaths = [
            # # Western Medicine-Disease
            # # D–P–S（0.11）
            # ['disease', 'gene', 'drug'],
            # ['drug', 'gene', 'disease'],
            #
            # # D–P–P–S（0.47）
            # ['disease', 'gene', 'gene', 'drug'],
            # ['drug', 'gene', 'gene', 'disease'],
            #
            # # D–P–P–P–S（0.28）
            # ['disease', 'gene', 'gene', 'gene', 'drug'],
            # ['drug', 'gene', 'gene', 'gene', 'disease'],
            #
            # # D–P–G–P–S（0.14）
            # ['disease', 'gene', 'GO', 'gene', 'drug'],
            # ['drug', 'gene', 'GO', 'gene', 'disease']


            # # Compound-Disease
            # # D–P–S（0.11）
            # ['disease', 'gene', 'drug'],
            # ['drug', 'gene', 'disease'],
            # # D–P–P–S（0.05）
            # ['disease', 'gene', 'gene', 'drug'],
            # ['drug', 'gene', 'gene', 'disease'],
            # # D–P–P–G–P–S（0.05）
            # ['disease', 'gene', 'gene', 'GO', 'gene', 'drug'],
            # ['drug', 'gene', 'GO', 'gene', 'gene', 'disease'],
            # # D–P–G–P–S（0.65）
            # ['disease', 'gene', 'GO', 'gene', 'drug'],
            # ['drug', 'gene', 'GO', 'gene', 'disease'],
            # # D–P–G–G–P–S（0.10）
            # ['disease', 'gene', 'GO', 'GO', 'gene', 'drug'],
            # ['drug', 'gene', 'GO', 'GO', 'gene', 'disease'],
            # # D–P–G–P–G–P–S（0.04）
            # ['disease', 'gene', 'GO', 'gene', 'GO', 'gene', 'drug'],
            # ['drug', 'gene', 'GO', 'gene', 'GO', 'gene', 'disease']

            # TCM-Symptom
            ['drug', 'gene', 'disease'],
            ['disease', 'gene', 'drug'],
            ['drug', 'gene', 'gene', 'disease'],
            ['disease', 'gene', 'gene', 'drug'],
            ['drug', 'gene', 'gene','gene', 'disease'],
            ['disease', 'gene', 'gene', 'gene', 'drug'],
            ['drug', 'gene', 'GO', 'gene', 'disease'],
            ['disease', 'gene', 'GO', 'gene', 'drug'],
            ['drug', 'gene', 'GO', 'GO', 'gene', 'disease'],
            ['disease', 'gene', 'GO', 'GO', 'gene', 'drug']
        ]

    # =========================
    # Default weights (corresponding to metapaths one-to-one)
    # - Higher weight = higher probability of selecting this metapath
    # - No normalization required (relative ratio)
    # =========================
    if metapath_weights is None:
        # Example: Prefer shorter paths (first two), lower weights for longer paths
        # metapath_weights = [0.1, 0.1, 0.40, 0.40, 0.40, 0.40,0.10,0.10]
        # metapath_weights = [
        #     0.11, 0.11,
        #     0.47, 0.47,
        #     0.28, 0.28,
        #     0.14, 0.14
        # ]

        # metapath_weights = [0.05, 0.05, 0.10, 0.10, 0.65, 0.65, 0.20,0.20]
        # metapath_weights = [
        #     0.11, 0.11,
        #     0.05, 0.05,
        #     0.05, 0.05,
        #     0.66, 0.66,
        #     0.09, 0.09,
        #     0.04, 0.04
        # ]

        metapath_weights = [0.05, 0.05, 0.30, 0.30, 0.25, 0.25, 0.25,0.25,0.15,0.15]

    if len(metapath_weights) != len(metapaths):
        raise ValueError(
            f"Length of metapath_weights ({len(metapath_weights)}) must equal length of metapaths ({len(metapaths)})"
        )

    print('Generating metapath-guided walks...')
    walks = generate_metapath_walks(
        G=G,
        node_types=node_types,
        metapaths=metapaths,
        metapath_weights=metapath_weights,
        walk_length=walk_length,
        num_walks=num_walks,
        seed=seed,
        workers=workers
    )

    os.makedirs('walks', exist_ok=True)
    with open('walks/tmp_walk_file.pkl', 'wb') as fw:
        pickle.dump(walks, fw)

    print('Generating node embeddings...')
    use_hetSG = True if nodetypef is not None else False
    embeddings = HeterogeneousSG(
        use_hetSG,
        walks,
        set(G.nodes()),
        nodetypef=nodetypef,
        embedding_size=dimension,
        window_length=window_size,
        workers=workers
        # workers=1
    )

    with open(outputf, 'wb') as fw:
        pickle.dump(embeddings, fw)

    print(f'Node embeddings saved: {outputf}')