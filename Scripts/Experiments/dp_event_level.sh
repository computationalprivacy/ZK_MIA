#ZK - Dp - event - reuse
for epsilon in 0\.1 0\.5 1\.0 5\.0 10\.0
do
    for k in 0
    do
        python main.py --MIA_name=zero_knowledge\
            --dataset_name=Milano_twitter_resized\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=zero_knowledge_milano_DP_event_level_reuse\
            --pca_components=0\
            --bucket_threshold=$k\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
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

#Knock2 - Dp - event
for epsilon in 0\.1 0\.5 1\.0 5\.0 10\.0
do
    for k in 0
    do
        python main.py --MIA_name=knock2\
            --dataset_name=Milano_twitter_resized\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=knock2_milano_DP_event_level\
            --pca_components=0\
            --bucket_threshold=$k\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
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

#ZK - Dp - event
for epsilon in 0\.1 0\.5 1\.0 5\.0 10\.0
do
    for k in 0
    do
        python main.py --MIA_name=zero_knowledge\
            --dataset_name=Milano_twitter_resized\
            --save_dir=results \
            --delaunay_nbrs_dict=pre_processing/data_dictionaries/delaunay_nbrs_dict_Milano.pickle\
            --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
            --train_size=400\
            --n_groups=100\
            --validation_size=100\
            --experiment_name=zero_knowledge_milano_DP_event_level\
            --pca_components=0\
            --bucket_threshold=$k\
            --DP_eps=$epsilon\
            --DP_sens=1\
            --classification=LR\
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