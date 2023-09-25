from feature_engineering.engine import FeatureEngineeringEngine


if __name__ == '__main__':
    #engine = FeatureEngineeringEngine(pdb_path = 'data/pdb', decoy_path = 'dataset_A/rosetta_processed', graph_path = 'dataset_A/rosetta_graphs')
    engine = FeatureEngineeringEngine(pdb_path = 'dataset_C/pdbs', decoy_path = 'dataset_C/smc_processed', graph_path = 'dataset_C/smc_graphs')
    #engine.run_debug(n_decoys = 5)
    engine.run(n_processes = 4)
    
    
    


