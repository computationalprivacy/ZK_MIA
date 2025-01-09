from src.dataset_sampler import *
from src.attacks import *
from src.utils import *
import time
from tqdm import tqdm
from multiprocessing import Pool, Manager, current_process
import random

    
''' 
Implementation of the Knock Knock MIA experiment 
Computation is optimized by using saved training, validation and test aggregates,
which were previously created during pre-processing, using sample_aggregates.py

To evaluate the MIA on privacy-aware aggregates (suppression of small counts or DP noise),
or given an adversary with partial trace knowledge, the saved aggregates and target traces
are adjusted accordingly using the clean_saved_aggregates function from data_sampler.py
'''

def run_knock2_from_saved_aggs(save_dir, args):
    '''
    Runs the knock2 attack on a number of targets. Uses pre-saved aggregates
    '''
    # access dictionary of saved aggregates for training, validating, and testing the MIA
    global data_sampler
    data_sampler = DatasetSampler.from_saved_aggregates(args.saved_aggregates_filename, args.seed, 'knock2')
    # adjusts the aggregates based on the attack settings (ex. using only a partial target trace or privacy settings like DP)
    data_sampler.clean_saved_aggregates('knock2', args.bucket_threshold, train_size=args.train_size, validation_size=args.validation_size, test_size=args.n_groups, n_targets=args.n_targets, DP_eps = args.DP_eps, DP_sens = args.DP_sens, paired_sampling = args.paired_in_out_sampling, partial_trace_mode = args.partial_trace_mode, frac_target_trace = args.frac_target_trace,  top_k=args.top_k)
    # get target ids and other attack parameters from dictionary
    targets, n_rois, n_epochs, train_size, validation_size, test_size, group_size = data_sampler.load_statistics_from_saved_aggs('knock2')
    # reset relevant parameters
    args.n_targets = len(targets)
    args.train_size = train_size
    args.validation_size = validation_size
    args.test_size = test_size
    args.group_size = group_size
    # initialize dictionary for results
    acc_auc_scores, membership_labels = initialize_result_dictionaries(args)
    t1_ = time.time()
    for target in targets:
        print(f'now attacking user {target}')
        # retrieve all relevant aggregates for the given target
        train_aggregates, train_labels, validation_aggregates, validation_labels, test_aggregates, test_labels = data_sampler.load_target_information(
            target, 'knock2')
        if args.DP_eps is not None and args.DP_sens is not None:
            scale = args.DP_sens / args.DP_eps
        else:
            scale = None
        if args.bucket_threshold == 0 and scale is None and args.partial_trace_mode is None:
            # hybridize ML MIA with baseline rule if the test aggregate is raw
            predictions = []
            p_scores = []
            target_trace = data_sampler.data_dictionary[target]
            # set up and train the ML classifier given the loaded train_aggregates computed with reference traces
            knock2_attack = Binary_classifier(target, train_aggregates, train_labels,
                                              validation_aggregates=validation_aggregates,
                                              validation_labels=validation_labels)
            if args.classification == 'RF':
                knock2_attack.set_RF_hyperparameters(args.n_trees, args.max_depth)
            elif args.classification == 'LR':
                knock2_attack.set_LR_hyperparameters(args.LR_C, args.LR_max_iter, args.LR_tol)
            classifier, pca, scaler, validation_threshold = knock2_attack.train_model(args.classification, args.scaler_type,
                                                                                      args.pca_components)
            for test_aggregate, test_label in zip(test_aggregates, test_labels):
                if np.any(target_trace.toarray() == np.ones(test_aggregate.shape) - (
                        test_aggregate / group_size).toarray()):
                    # apply baseline contradiction rule for non-membership
                    predictions.append(0)
                    p_scores.append(0)
                else:
                    knock2_attack.set_test_aggregate(test_aggregate, test_label)
                    p_score, prediction, validation_threshold = knock2_attack.test_model(classifier, pca, scaler,
                                                                                         args.pca_components,
                                                                                         validation_threshold)
                    p_scores.append(p_score[0])
                    predictions.append(prediction[0])
            acc, auc = compute_acc_auc(test_labels, predictions, p_scores)
        else:
            # use pure ML MIA
            knock2_attack = Binary_classifier(target, train_aggregates, train_labels, test_aggregates, test_labels,
                                              validation_aggregates, validation_labels)
            if args.classification == 'RF':
                knock2_attack.set_RF_hyperparameters(args.n_trees, args.max_depth)
            elif args.classification == 'LR':
                knock2_attack.set_LR_hyperparameters(args.LR_C, args.LR_max_iter, args.LR_tol)
            # run the attack and retrieve the performance statistics
            p_scores, predictions, validation_threshold = knock2_attack.run_attack(args.classification, args.scaler_type,
                                                                                   args.pca_components)
            acc, auc = compute_acc_auc(test_labels, predictions, p_scores)
        print(f'accuracy of Knock2 MIA on target {target}: {acc}')
        print(f'AUC of Knock2 MIA on target {target}: {auc}')
        acc_auc_scores[target] = (acc, auc)
    t2_=time.time()
    print(f'Runtime // : {t2_-t1_}')
    save_all_results(acc_auc_scores, membership_labels, save_dir, args)

''' 
The following functions implement the Zero Auxiliary Knowledge MIA experiment 
Computation is optimized by using the sam test aggregates, which were created using sample_aggregates.py
By default, the ZK adversary generates their own training and validation aggregates for each test aggregate.

If privacy settings were applied, then the saved aggregates will get adjusted accordingly using clean_saved_aggregates from data_sampler.py

The implementation is parallelized at the level of the test aggregates for each target.
'''


def run_zero_knowledge_from_saved_aggs(save_dir, args):
    '''
    Runs the zero knowledge MIA on multiple targets and other predefined parameters, where Adv approximates marginals from the released aggregate, uses them to create synthetic traces,
    which are then used to train a membership classifier. The function is parallelized at the level of released (test) aggregates with nested parallelization
    '''
    # This object will control the computation of train and test aggregeates, the generation of synthetic traces, and other dataset specific tasks
    global data_sampler
    data_sampler = DatasetSampler.from_saved_aggregates(args.saved_aggregates_filename, args.seed, 'zero_knowledge')
    data_sampler.clean_saved_aggregates('zero_knowledge', args.bucket_threshold, test_size=args.n_groups, n_targets=args.n_targets, DP_eps = args.DP_eps, DP_sens = args.DP_sens)
    # get target ids and other stats from dictionary
    targets, n_rois, n_epochs, test_size, group_size = data_sampler.load_statistics_from_saved_aggs('zero_knowledge')
    # reset relevant paremeters
    args.n_targets = len(targets)
    args.test_size = test_size
    args.group_size = group_size
    data_sampler.apply_delaunay_clustering_from_saved_dictionary(args.delaunay_nbrs_dict)
    print(f'{args.n_targets} users have been selected as targets for the MIA')
    # initialize dictionary objects for storing MIA results
    acc_auc_scores, membership_labels = initialize_result_dictionaries(args)
    # we generate the activity marginal n_visits_user accordingly
    if args.true_mean == True:
        n_visits_user = compute_activity_marginal_mean(data_sampler, args.activity_marginal)
    else:
        n_visits_user = None
    t1_ = time.time()
    for target in tqdm(targets):
        t1 = time.time()
        print(f'now attacking user {target}')
        # this modifies the target's trace for training if applicable
        data_sampler.modify_target_trace(target, args.frac_target_trace, args.partial_trace_mode, args.top_k)
        # obtain test aggregates and labels from saved aggregates
        test_aggregates, test_labels = data_sampler.load_target_information(target, 'zero_knowledge')
        if args.DP_eps is not None and args.DP_sens is not None:
            scale = args.DP_sens / args.DP_eps
        else:
            scale = None
        if args.reuse_synthetic_traces:
            # create synthetic traces from the first test aggregate. In this implementation, we will re-use
            # these same synthetic traces to test the ZK MIA on other test aggregates pertaining to the same target
            test_aggregate = test_aggregates[0]
            train_aggregates, train_labels, validation_aggregates, validation_labels = obtain_zk_training_aggregates(data_sampler, target, test_aggregate, group_size, n_rois, n_epochs, scale, args)
            if args.bucket_threshold == 0 and scale is None and args.partial_trace_mode is None:
                # hybridize ML MIA with baseline rule if the test aggregate is raw
                predictions = []
                p_scores = []
                target_trace = data_sampler.data_dictionary[target]
                # set up and train one ML classifier given the train_aggregates
                zk_attack = Binary_classifier(target, train_aggregates, train_labels,
                                                  validation_aggregates=validation_aggregates,
                                                  validation_labels=validation_labels)
                if args.classification == 'RF':
                    zk_attack.set_RF_hyperparameters(args.n_trees, args.max_depth)
                elif args.classification == 'LR':
                    zk_attack.set_LR_hyperparameters(args.LR_C, args.LR_max_iter, args.LR_tol)
                classifier, pca, scaler, validation_threshold = zk_attack.train_model(args.classification,
                                                                                          args.scaler_type,
                                                                                          args.pca_components)
                for test_aggregate, test_label in zip(test_aggregates, test_labels):
                    if np.any(target_trace.toarray() == np.ones(test_aggregate.shape) - (
                            test_aggregate / group_size).toarray()):
                        # apply baseline contradiction rule for non-membership
                        predictions.append(0)
                        p_scores.append(0)
                    else:
                        zk_attack.set_test_aggregate(test_aggregate, test_label)
                        p_score, prediction, validation_threshold = zk_attack.test_model(classifier, pca, scaler,
                                                                                             args.pca_components,
                                                                                             validation_threshold)
                        p_scores.append(p_score[0])
                        predictions.append(prediction[0])
                acc, auc = compute_acc_auc(test_labels, predictions, p_scores)
            else:
                zk_attack = Binary_classifier(target, train_aggregates, train_labels, test_aggregates, test_labels,
                                                  validation_aggregates, validation_labels)
                zk_attack.set_RF_hyperparameters(args.RF_n_trees, args.RF_max_depth)
                zk_attack.set_LR_hyperparameters(args.LR_C, args.LR_max_iter, args.LR_tol)
                p_scores, predictions, validation_threshold = zk_attack.run_attack(args.classification,
                                                                                       args.scaler_type,
                                                                                       args.pca_components)
                membership_labels[target] = (test_labels, predictions, p_scores, validation_threshold)
                acc, auc = compute_acc_auc(test_labels, predictions, p_scores)
        else:
            # we train a separate classifier for each test aggregate, using synthetic traces generated from the aggregate
            p_scores = []
            predictions = []
            results = []
            y_labels = []
            synthetic_data_dictionary_from_aggregate = {}
            # set up parallelization at the level of aggregate groups
            if args.chunk_size > 1:
                list_args = [(data_sampler, test_aggregate, test_label, args.group_size, args.n_synthetic_traces, target, data_sampler.n_rois, data_sampler.n_epochs, args.nbr_cores_chunk, args.train_size, args.paired_in_out_sampling, args.DP_eps, args.DP_sens, args.bucket_threshold, args.classification, args.scaler_type, args.pca_components, args.synthetic_trace_mode, args.activity_marginal, args.log_compression, args.poly_transformation, n_visits_user, args.RF_n_trees, args.RF_max_depth, args.LR_C, args.LR_max_iter, args.LR_tol, args.validation_size, args.cluster_size, synthetic_data_dictionary_from_aggregate, args.partial_trace_mode) for test_aggregate, test_label in zip(test_aggregates, test_labels)]
                chunk_size = args.chunk_size
                n_chunks = len(list_args) // chunk_size
                for chunk_idx in tqdm(range(n_chunks)):
                    # define chunk
                    start_idx = chunk_idx * chunk_size
                    end_idx = (chunk_idx + 1) * chunk_size
                    chunk_list_args = list_args[start_idx:end_idx]
                    with Pool(chunk_size, initializer=init_pool) as pool:
                        chunk_results = list(pool.starmap(run_zero_knowledge_test_aggregate_from_saved_agg, chunk_list_args))
                    results.extend(chunk_results)
            else:
                for test_aggregate, test_label in zip(test_aggregates, test_labels):
                    result = run_zero_knowledge_test_aggregate_from_saved_agg(data_sampler, test_aggregate, test_label, args.group_size, args.n_synthetic_traces, target, data_sampler.n_rois, data_sampler.n_epochs, args.nbr_cores_chunk, args.train_size, args.paired_in_out_sampling, args.DP_eps, args.DP_sens, args.bucket_threshold, args.classification, args.scaler_type, args.pca_components, args.synthetic_trace_mode, args.activity_marginal, args.log_compression, args.poly_transformation, n_visits_user, args.RF_n_trees, args.RF_max_depth, args.LR_C, args.LR_max_iter, args.LR_tol, args.validation_size, args.cluster_size, synthetic_data_dictionary_from_aggregate, args.partial_trace_mode)
                    results.append(result)
            for result in results:
                y_labels.append(result[0])
                predictions += result[1]
                p_scores += result[2]
                validation_threshold = result[3]
            membership_labels[target] = (y_labels, predictions, p_scores, validation_threshold)
            acc, auc = compute_acc_auc(y_labels, predictions, p_scores)
        print(f'accuracy of ZK MIA on target {target}: {acc}')
        print(f'AUC of ZK MIA on target {target}: {auc}')
        t2 = time.time()
        acc_auc_scores[target] = (acc, auc, round(t2-t1,2))
        print(f'Time to attack target {target}: {t2-t1}')
    t2_=time.time()
    print(f'Time to run experiment: {t2_-t1_}')
    save_all_results(acc_auc_scores, membership_labels, save_dir, args)

def run_zero_knowledge_test_aggregate_from_saved_agg(data_sampler, test_aggregate,\
                                     test_label,\
                                     group_size,\
                                     n_synthetic_traces,\
                                     target,\
                                     n_rois,\
                                     n_epochs,\
                                     nbr_cores,\
                                     train_size,\
                                     paired_in_out_sampling,\
                                     DP_eps,\
                                     DP_sens,\
                                     bucket_threshold,\
                                     classification,\
                                     scaler_type,\
                                     pca_components,\
                                     synthetic_trace_mode,\
                                     activity_marginal,\
                                     log_compression,\
                                     poly_transformation,\
                                     n_visits_user,\
                                     n_trees,\
                                     max_depth,\
                                     LR_C, \
                                     LR_max_iter, \
                                     LR_tol, \
                                     validation_size,\
                                     cluster_size, \
                                     synthetic_data_dictionary_from_aggregate, \
                                     partial_trace_mode):
    '''
    Runs the zero knowledge MIA on one released (test) aggregate, where Adv approximates marginals from the released aggregate, uses them to create synthetic traces, 
    which are then used to train a membership classifier. This function calls further levels of parallelization in process_group and data_sampler.generate_synthetic_traces_unicity
    '''
    if DP_eps is not None and DP_sens is not None:
        scale = DP_sens/DP_eps
    else:
        scale = None
    if bucket_threshold == 0 and scale is None and partial_trace_mode is None:
        # hybridize ML MIA with baseline rule if the test aggregate is raw
        target_trace = data_sampler.data_dictionary[target]
        if np.any(target_trace.toarray() == np.ones(test_aggregate.shape)-(test_aggregate/group_size).toarray()):
            # return baseline contradiction
            prediction = [0]
            p_score = [0]
            validation_threshold = 0.5
            return test_label, prediction, p_score, validation_threshold
    # otherwise, proceed with the standard zero knowledge MIA
    if synthetic_trace_mode == "unicity_marginals":
        # approximate space marginal from test aggregate 
        roi_dist = np.asarray(test_aggregate.sum(axis=1)/test_aggregate.sum()).flatten()
        # approximate time marginal from aggregate
        epoch_dist = np.asarray(test_aggregate.sum(axis=0)/test_aggregate.sum()).flatten()
        # log compression should be automatically called if small counts are suppressed
        if log_compression:
            roi_dist = apply_log_transformation(roi_dist)
            epoch_dist = apply_log_transformation(epoch_dist)
        # If the aggregates are noisy, we should denoise the resulting space marginal with the power transformation
        if DP_eps is not None:
            if poly_transformation:
                # take n_rois samples of Unif(0,1) for visits to each ROI
                roi_visits_samples = np.random.uniform(0,1, n_rois)
                normalized_roi_visits_samples = roi_visits_samples / np.sum(roi_visits_samples)
                # Adv estimates the variance of the space marginal from naive independent uniform samples
                variance_space_ref = np.var(normalized_roi_visits_samples)
                # choose corresponding p for denoising space marginal given desired variance
                p = select_p_knowing_variance(roi_dist, variance_space_ref, scale, 10)
                # transform the space marginal
                roi_dist = apply_poly_transformation(roi_dist, p)
                # take n_epochs samples of Unif(0,1) for visits to each epoch
                epoch_visits_samples = np.random.uniform(0,1, n_epochs)
                normalized_epoch_visits_samples = epoch_visits_samples / np.sum(epoch_visits_samples)
                # Adv estimates the variance of the time marginal from naive independent uniform samples
                variance_time_ref = np.var(normalized_epoch_visits_samples)
                # choose corresponding p for denoising time marginal given desired variance
                p = select_p_knowing_variance(epoch_dist, variance_time_ref, scale, 10)
                # transform the time marginal
                epoch_dist = apply_poly_transformation(epoch_dist, p)
        # estimate the population's mean number of visits
        if n_visits_user is None:
            mean = estimate_true_mean(data_sampler, test_aggregate, roi_dist, epoch_dist, test_aggregate.sum()/group_size, group_size, activity_marginal, bucket_threshold, n_rois, n_epochs, 2023, noise_scale = scale)
            # estimate activity marginal given the estimated mean
            n_visits_user = compute_activity_marginal_mean(data_sampler, activity_marginal, mean=mean)
        # generate synthetic traces using the unicity model, with the approximated marginals as inputs
        synthetic_data_dictionary_from_aggregate = data_sampler.generate_synthetic_traces_unicity(roi_dist, epoch_dist, n_visits_user, n_synthetic_traces, nbr_cores, cluster_size)
    reference = [target] + list(synthetic_data_dictionary_from_aggregate.keys())
    # Sample train aggregation groups
    t1_agg = time.time()
    train_agg_groups, train_labels = data_sampler.sample_agg_groups(target, reference, group_size, train_size, paired_in_out_sampling)
    if paired_in_out_sampling:
        # compute aggregates in pairs
        grouped_train_agg_groups = [(train_agg_groups[i], train_agg_groups[i + 1]) for i in range(0, len(train_agg_groups), 2)]
        group_label_args = [(data_sampler, group_pair, DP_eps, DP_sens, bucket_threshold, target, synthetic_data_dictionary_from_aggregate) for group_pair in grouped_train_agg_groups]
        with Pool(nbr_cores) as pool:
            # Process train aggregation groups in parallel
            aggregate_pairs = pool.map(process_group_pair, group_label_args)
        train_aggregates = [aggregate for pair in aggregate_pairs for aggregate in pair]
        train_labels = np.empty((train_size,),int)
        train_labels[::2] = 1
        train_labels[1::2] = 0
    else:
        # compute aggregates one by one if not paired sampling
        group_label_args = [(data_sampler, group, label, DP_eps, DP_sens, bucket_threshold, target, synthetic_data_dictionary_from_aggregate) for group, label in zip(train_agg_groups, train_labels)]
        with Pool(nbr_cores) as pool:
            # Process validation aggregation groups in parallel
            results = pool.map(process_group, group_label_args)
        # Separate the results into validation_aggregates and validation_labels
        train_aggregates, train_labels = zip(*[(train_aggregate, label) for train_aggregate, label in results])
    # Sample validation aggregation groups
    if validation_size > 0:
        validation_agg_groups, validation_labels = data_sampler.sample_agg_groups(target, reference, group_size, validation_size, paired_in_out_sampling = False)
    # We will create validation aggregates and labels in parallel: first, create a list of tuples containing group, label, and other parameters
    if validation_size > 0:
        group_label_args = [(data_sampler, group, label, DP_eps, DP_sens, bucket_threshold, target, synthetic_data_dictionary_from_aggregate) for group, label in zip(validation_agg_groups, validation_labels)]
        with Pool(nbr_cores) as pool:
            # Process validation aggregation groups in parallel
            results = pool.map(process_group, group_label_args)
        # Separate the results into validation_aggregates and validation_labels
        validation_aggregates, validation_labels = zip(*[(validation_aggregate, label) for validation_aggregate, label in results])
    else:
        validation_aggregates, validation_labels = [], []
    t2_agg = time.time()
    t1_a=time.time()
    zk_attack = Binary_classifier( target, train_aggregates, train_labels, [test_aggregate], [test_label], validation_aggregates, validation_labels)
    if classification == 'RF':
        zk_attack.set_RF_hyperparameters(n_trees, max_depth)
    elif classification == 'LR':
        zk_attack.set_LR_hyperparameters(LR_C, LR_max_iter, LR_tol)
    p_score, prediction, validation_threshold = zk_attack.run_attack(classification, scaler_type, pca_components)
    t2_a = time.time()
    if test_label == prediction.tolist()[0]:
        print(f'Correct, Adv guesses {prediction.tolist()[0]} after seeing p_score {p_score.tolist()[0]} vs. validation threshold {validation_threshold}')
    else:
        print(f'Wrong, Real label was: {test_label}, but Adv guessed {prediction.tolist()[0]} after seeing p_score of {p_score.tolist()[0]} vs. validation threshold {validation_threshold}')
    print(f'Time to sample and create train and val aggregates: {round(t2_agg-t1_agg, 2)} seconds. \n Time to train and test attack: {round(t2_a-t1_a, 2)} seconds')
    return test_label, prediction.tolist(), p_score.tolist(), validation_threshold
    
    

    
    
''' The following are helper functions for parallelization'''

def process_group(args):
    data_sampler, group, label, DP_eps, DP_sens, bucket_threshold, target, synthetic_data_dictionary= args
    aggregate = data_sampler.compute_aggregate(group, bucket_threshold, DP_eps = DP_eps, DP_sens = DP_sens, training=True, target=target, synthetic_data_dictionary=synthetic_data_dictionary)
    return aggregate, label

def process_group_pair(args):
    data_sampler, group_pair, DP_eps, DP_sens, bucket_threshold, target, synthetic_data_dictionary = args
    in_aggregate, out_aggregate = data_sampler.compute_aggregate_pair(group_pair, bucket_threshold, DP_eps = DP_eps, DP_sens = DP_sens, training=True, target=target, synthetic_data_dictionary=synthetic_data_dictionary)
    return (in_aggregate, out_aggregate)
def init_pool():
    '''
    Helper method for initializing pools: Ensures that a multiprocessing pool can create another multiprocessing pool
    '''
    current_process().daemon = False

""" Helper functions for ZK MIA """

def obtain_zk_training_aggregates(data_sampler, target, test_aggregate, group_size, n_rois, n_epochs, scale, args):
    roi_dist, epoch_dist, n_visits_user = estimate_marginals_from_aggregate(args, data_sampler, test_aggregate, n_rois,
                                                                            n_epochs, scale)
    # generate synthetic traces using the unicity model, with the approximated marginals as inputs
    synthetic_data_dictionary_from_aggregate = data_sampler.generate_synthetic_traces_unicity(roi_dist,
                                                                                              epoch_dist,
                                                                                              n_visits_user,
                                                                                              args.n_synthetic_traces,
                                                                                              args.nbr_cores,
                                                                                              args.cluster_size)
    reference = [target] + list(synthetic_data_dictionary_from_aggregate.keys())
    # Sample groups of synthetic users for the training aggregates
    train_agg_groups, train_labels = data_sampler.sample_agg_groups(target, reference, group_size, args.train_size,
                                                                    args.paired_in_out_sampling)
    if args.paired_in_out_sampling:
        # compute aggregates in pairs
        grouped_train_agg_groups = [(train_agg_groups[i], train_agg_groups[i + 1]) for i in
                                    range(0, len(train_agg_groups), 2)]
        group_label_args = [(data_sampler, group_pair, args.DP_eps, args.DP_sens, args.bucket_threshold, target,
                             synthetic_data_dictionary_from_aggregate) for group_pair in
                            grouped_train_agg_groups]
        with Pool(args.nbr_cores) as pool:
            # Process train aggregation groups in parallel
            aggregate_pairs = pool.map(process_group_pair, group_label_args)
        train_aggregates = [aggregate for pair in aggregate_pairs for aggregate in pair]
        train_labels = np.empty((args.train_size,), int)
        train_labels[::2] = 1
        train_labels[1::2] = 0
    else:
        # compute aggregates one by one if not paired sampling
        group_label_args = [
            (data_sampler, group, label, args.DP_eps, args.DP_sens, args.bucket_threshold, target,
             synthetic_data_dictionary_from_aggregate)
            for group, label in zip(train_agg_groups, train_labels)]
        with Pool(args.nbr_cores) as pool:
            # Process validation aggregation groups in parallel
            results = pool.map(process_group, group_label_args)
        # Separate the results into validation_aggregates and validation_labels
        train_aggregates, train_labels = zip(*[(train_aggregate, label) for train_aggregate, label in results])
    # Sample validation aggregation groups
    if args.validation_size > 0:
        validation_agg_groups, validation_labels = data_sampler.sample_agg_groups(target, reference, group_size,
                                                                                  args.validation_size,
                                                                                  paired_in_out_sampling=False)
    # We will create validation aggregates and labels in parallel: first, create a list of tuples containing group, label, and other parameters
    if args.validation_size > 0:
        group_label_args = [
            (data_sampler, group, label, args.DP_eps, args.DP_sens, args.bucket_threshold, target,
             synthetic_data_dictionary_from_aggregate)
            for group, label in zip(validation_agg_groups, validation_labels)]
        with Pool(args.nbr_cores) as pool:
            # Process validation aggregation groups in parallel
            results = pool.map(process_group, group_label_args)
        # Separate the results into validation_aggregates and validation_labels
        validation_aggregates, validation_labels = zip(
            *[(validation_aggregate, label) for validation_aggregate, label in results])
    else:
        validation_aggregates, validation_labels = [], []
    return train_aggregates, train_labels, validation_aggregates, validation_labels