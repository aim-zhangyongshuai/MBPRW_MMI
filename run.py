from Random_Walk.generate_embeddings_M_RW_update import save_embedding_files
from Random_Walk.predict_associations_M_RW import predict_asso
from Random_Walk.generate_embeddings_control_group import run_random_walk_control_experiments

if __name__ == '__main__':
    # Network dataset
    networkf = 'data/Random walk network data/network  data -graph/processed_graph_add_TCM.csv'
    # networkf = 'data/Random walk network data/network  data -graph/processed_graph_add_CTD.csv'
    # networkf = 'data/Random walk network data/network  data -graph/processed_graph_add_drug.csv'

    # Node type dataset
    nodetypef = 'data/Random walk network data/Network node type/node_types_TCM_SID.csv'
    # nodetypef = 'data/Random walk network data/Network node type/node_types_CTD.csv'
    # nodetypef = 'data/Random walk network data/Network node type/node_types_drug.csv'

    # Label dataset
    pairf = 'data/Three treatment associations with Labeled Dataset/dda_TCM.csv'
    # pairf = 'data/Three treatment associations with Labeled Dataset/dda_CTD.csv'
    # pairf = 'data/Three treatment associations with Labeled Dataset/dda_drug.csv'

    # Clinical label dataset
    clinical_pairs = 'data/Clinical Label Dataset/clinical occurrence_pairs.tsv'
    effective_pairs = 'data/Clinical Label Dataset/clinical effectiveness_pairs.tsv'
    psm_pairs= 'data/Clinical Label Dataset/clinical PSM effectiveness_pairs.tsv'

    # Trained model file
    modelf = fr'output/clf.pkl'

    # Output path
    clinic_predict_file_path = f'output/clinical_pairs.csv'
    Effective_predict_file_path = f'output/effective_pairs.csv'
    psm_predict_file_path=f'output/psm_pairs.csv'
    predict_file_path = f'output/pairs.csv'

    embeddingf = fr"output/embeddingf/embedding_file_TCM.pkl"

    # Generate node embeddings. Modify meta-path and sampling probability inside the function if needed.
    # Western medicine and compound groups: 10 random walks with length 50
    # TCM group: 100 random walks with length 50
    # MBPRW model
    # save_embedding_files(netf=networkf, outputf=embeddingf, nodetypef=nodetypef, walk_length=50, num_walks=100,seed=334,dimension=128)

    for seed in range(334,344):
        predict_asso(embeddingf, pairf, predict_file_path, modelf, valid_ratio=0.1, test_ratio=0.1,train=True,seed=seed)

        # Clinical effectiveness prediction
        predict_asso(
            embedding_file=embeddingf,
            pair_file=effective_pairs,
            clinic_predict_file_path=Effective_predict_file_path,
            model_checkpoint=modelf,
            seed=seed,
            valid_ratio=0,  # No validation set required
            test_ratio=1.0,  # Use all data as test set
            train=False  # Disable training mode
        )
        print('=' * 50)

        # Clinical occurrence prediction
        predict_asso(
            embedding_file=embeddingf,
            pair_file=clinical_pairs,
            clinic_predict_file_path=clinic_predict_file_path,
            model_checkpoint=modelf,
            seed=seed,
            valid_ratio=0,  # No validation set required
            test_ratio=1.0,  # Use all data as test set
            train=False  # Disable training mode
        )
        # PSM effectiveness prediction
        predict_asso(
            embedding_file=embeddingf,
            pair_file=psm_pairs,
            clinic_predict_file_path=psm_predict_file_path,
            model_checkpoint=modelf,
            seed=seed,
            valid_ratio=0,  # No validation set required
            test_ratio=1.0,  # Use all data as test set
            train=False  # Disable training mode
        )