import subprocess
import os

agg_group_sizes = [1000] # default value used in paper's experiments

for agg_group_size in agg_group_sizes:
    arguments = [
        'sample_aggregates.py',
        '--dataset_name', 'Milano',
        '--data_dictionary_pickle_name', 'data_dictionaries/data_dictionary_Milano.pickle', # enter filepath to preprocessed data dictionary
        '--save_dir', '', # enter path to directory where aggregates will be saved
        '--n_seed', str(42), # enter randomization seed for reproducibility
        '--num_processes', str(8), # enter desired number of processes for parallelization
        '--n_targets', str(50), # Default value used in paper's experiments
        '--agg_group_size', str(agg_group_size),
        '--ref_size', str(2500), # Default value used in paper's experiments
        '--train_size', str(400), # Default value used in paper's experiments
        '--test_size', str(100), # Default value used in paper's experiments
        '--validation_size', str(100), # Default value used in paper's experiments
    ]

    # Execute the Python script with arguments
    subprocess.run(['python'] + arguments)

