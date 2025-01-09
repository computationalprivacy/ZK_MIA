#ZK - k - fraction of the target trace - reuse
for epsilon in 1\.0
do
    for frac in 0\.1 0\.25 0\.5 0\.75 1\.0
    do
        python main.py --MIA_name=zero_knowledge\
            --dataset_name=Milano_twitter\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=zero_knowledge_milano_partial_trace_reuse\
            --pca_components=0\
            --bucket_threshold=1\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
            --frac_target_trace=$frac\
            --partial_trace_mode=random\
            --n_synthetic_trace=5000\
            --chunk_size=20\
            --nbr_cores_chunk=4\
            --synthetic_trace_mode=unicity_marginals\
            --activity_marginal=exp\
            --true_mean=False\
            --scaler_type=Standard\
            --reuse_synthetic_traces=True\
            --poly_transformation=True\
            --log_compression=False
    done
done

#Knock2 - fraction of the target trace
for epsilon in 1\.0
do
    for frac in 0\.1 0\.25 0\.5 0\.75 1\.0
    do
        python main.py --MIA_name=knock2\
            --dataset_name=Milano_twitter\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=knock2_milano_partial_trace\
            --pca_components=0\
            --bucket_threshold=1\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
            --frac_target_trace=$frac\
            --partial_trace_mode=random\
            --n_synthetic_trace=5000\
            --chunk_size=20\
            --nbr_cores_chunk=4\
            --synthetic_trace_mode=unicity_marginals\
            --activity_marginal=exp\
            --true_mean=False\
            --scaler_type=Standard\
            --reuse_synthetic_traces=False\
            --poly_transformation=True\
            --log_compression=False
    done
done

#ZK - k - fraction of the target trace
for epsilon in 1\.0
do
    for frac in 0\.1 0\.25 0\.5 0\.75 1\.0
    do
        python main.py --MIA_name=zero_knowledge\
            --dataset_name=Milano_twitter\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=zero_knowledge_milano_partial_trace\
            --pca_components=0\
            --bucket_threshold=1\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
            --frac_target_trace=$frac\
            --partial_trace_mode=random\
            --n_synthetic_trace=5000\
            --chunk_size=20\
            --nbr_cores_chunk=4\
            --synthetic_trace_mode=unicity_marginals\
            --activity_marginal=exp\
            --true_mean=False\
            --scaler_type=Standard\
            --reuse_synthetic_traces=False\
            --poly_transformation=True\
            --log_compression=False
    done
done