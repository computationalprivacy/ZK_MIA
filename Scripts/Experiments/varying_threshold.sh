#ZK - k - reuse
for k in 0 1 2 3 4 5
do
     python main.py --MIA_name=zero_knowledge\
     --dataset_name=Milano_twitter\
     --save_dir=results \
     --experiment_name=zero_knowledge_milano_k_reuse\
     --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
     --train_size=400 \
     --n_groups=100 \
     --validation_size=100\
     --pca_components=0 \
     --bucket_threshold=$k \
     --classification=LR \
     --reuse_synthetic_traces=True\
     --scaler_type=Standard \
     --nbr_cores=1
done

#Knock2 - k
for k in 0 1 2 3 4 5
do
     python main.py --MIA_name=knock2\
     --dataset_name=Milano_twitter\
     --save_dir=results \
     --experiment_name=knock2_milano_k\
     --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
     --train_size=400 \
     --n_groups=100 \
     --validation_size=100\
     --pca_components=0 \
     --bucket_threshold=$k \
     --classification=LR \
     --scaler_type=Standard \
     --nbr_cores=1
done

#ZK - k
for k in 0 1 2 3 4 5
do
     python main.py --MIA_name=zero_knowledge\
     --dataset_name=Milano_twitter\
     --save_dir=results \
     --experiment_name=zero_knowledge_milano_k\
     --saved_aggregates_filename=pre_processing/aggregates/Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle\
     --train_size=400 \
     --n_groups=100 \
     --validation_size=100\
     --pca_components=0 \
     --bucket_threshold=$k \
     --classification=LR \
     --reuse_synthetic_traces=False\
     --scaler_type=Standard \
     --nbr_cores=1
done