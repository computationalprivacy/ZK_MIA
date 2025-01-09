from scipy import stats
from scipy.stats import expon, powerlaw, pareto
import numpy as np
from math import sqrt, isqrt
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import errno
import os
import pickle


def estimate_marginals_from_aggregate(args, data_sampler, test_aggregate, n_rois, n_epochs, DP_scale):
    # approximate space marginal from test aggregate
    roi_dist = np.asarray(test_aggregate.sum(axis=1) / test_aggregate.sum()).flatten()
    # approximate time marginal from aggregate
    epoch_dist = np.asarray(test_aggregate.sum(axis=0) / test_aggregate.sum()).flatten()
    # log compression should be automatically called if small counts are suppressed
    if args.log_compression:
        roi_dist = apply_log_transformation(roi_dist)
        epoch_dist = apply_log_transformation(epoch_dist)
    # If the aggregates are noisy, we should denoise the resulting space marginal with the power transformation
    if DP_scale is not None:
        if args.poly_transformation:
            # take n_rois samples of Unif(0,1) for visits to each ROI
            roi_visits_samples = np.random.uniform(0, 1, n_rois)
            normalized_roi_visits_samples = roi_visits_samples / np.sum(roi_visits_samples)
            # Adv estimates the variance of the space marginal from naive independent uniform samples
            variance_space_ref = np.var(normalized_roi_visits_samples)
            # choose corresponding p for denoising space marginal given desired variance
            p = select_p_knowing_variance(roi_dist, variance_space_ref, DP_scale, 10)
            # transform the space marginal
            roi_dist = apply_poly_transformation(roi_dist, p)
            # take n_epochs samples of Unif(0,1) for visits to each epoch
            epoch_visits_samples = np.random.uniform(0, 1, n_epochs)
            normalized_epoch_visits_samples = epoch_visits_samples / np.sum(epoch_visits_samples)
            # Adv estimates the variance of the time marginal from naive independent uniform samples
            variance_time_ref = np.var(normalized_epoch_visits_samples)
            # choose corresponding p for denoising time marginal given desired variance
            p = select_p_knowing_variance(epoch_dist, variance_time_ref, DP_scale, 10)
            # transform the time marginal
            epoch_dist = apply_poly_transformation(epoch_dist, p)
    # estimate the population's mean number of visits
    if args.true_mean:
        n_visits_user = compute_activity_marginal_mean(data_sampler, args.activity_marginal, mean=None)
    else:
        mean = estimate_true_mean(data_sampler, test_aggregate, roi_dist, epoch_dist,
                                  test_aggregate.sum() / args.group_size, args.group_size, args.activity_marginal,
                                  args.bucket_threshold, n_rois, n_epochs, 2023, noise_scale=DP_scale)
        # estimate activity marginal given the estimated mean
        n_visits_user = compute_activity_marginal_mean(data_sampler, args.activity_marginal, mean=mean)
    return roi_dist, epoch_dist, n_visits_user

def estimate_true_mean(data_sampler, test_aggregate, roi_dist, epoch_dist, initial_mean, group_size, activity_marginal_setting, k, n_rois, n_epochs, n_seed, noise_scale = None, nbr_cores = 5, max_iterations=5, tol=0.001, step_weight=2, circadian = None, cluster_size = 10):
    if k == 0 and noise_scale is None:
        mean = test_aggregate.sum()/group_size
        # print('from k=0:', mean)
        return mean
    its = 1
    mean = initial_mean
    while its <= max_iterations:
        # print(its, mean)
        n_visits_user = compute_activity_marginal_mean(data_sampler,activity_marginal_setting, mean=mean)
        # create aggregate from synthetic traces
        aggregate = csr_matrix((n_rois, n_epochs), dtype=np.int8)
        synthetic_data_dictionary_from_aggregate = data_sampler.generate_synthetic_traces_unicity(roi_dist, epoch_dist, n_visits_user, group_size, nbr_cores, cluster_size)
        for user_id in synthetic_data_dictionary_from_aggregate:
            aggregate += synthetic_data_dictionary_from_aggregate[user_id]
        aggregate = apply_privacy(aggregate, k, noise_scale, group_size)
        its +=1
        if noise_scale is None:
            if k > 0:
                tamp = mean + step_weight*k*(test_aggregate.count_nonzero() - aggregate.count_nonzero())/(group_size)
                if tamp <= 0 :
                    return mean
                else:
                    new_mean = tamp
            else:
                tamp = mean + step_weight*(test_aggregate.sum() - aggregate.sum())/(group_size)
                if tamp <= 0 :
                    return mean
                else:
                    new_mean = tamp
        else:
            if k > 0:
                tamp = mean + step_weight*k*(test_aggregate.sum() - aggregate.sum())/(group_size)
                if tamp <= 0 :
                    return mean
                else:
                    new_mean = tamp
            else:
                tamp = mean + step_weight*(test_aggregate.sum() - aggregate.sum())/(group_size)
                if tamp <= 0 :
                    return mean
                else:
                    new_mean = tamp
        tolerance = abs(mean-new_mean)
        if tolerance < tol:
            # print(its,mean)
            return mean
        else:
            mean = new_mean
    # print(its, mean)
    if mean <= 0:
        return initial_mean
    else:
        return mean

def compute_activity_marginal_mean(data_sampler, activity_marginal_setting, skew = 3, mean=None):
    '''
    Estimates the population's activity marginal. If mean is None, Adv uses the true mean
    '''
    # code for case where Adv has access to true mean
    if mean is None:
        if activity_marginal_setting == 'constant':
            n_visits = data_sampler.get_activity_distribution()
            n_visits_user = [int(np.mean(n_visits))]
        elif activity_marginal_setting == 'true':
            n_visits_user = data_sampler.get_activity_distribution()
        elif activity_marginal_setting == 'lognormal':
            n_visits = data_sampler.get_activity_distribution()
            # give Adv access to true mean
            mean = np.mean(n_visits)
            # using the true mean and desired skewness, Adv approximates the activity distribution with a lognormal distribution
            sigma, median = solve_sigma_median_from_mean_skew(skew, mean)
            n_visits_user = stats.lognorm(s=sigma, scale=median)
        elif activity_marginal_setting == 'exp':
            n_visits = data_sampler.get_activity_distribution()
            # give Adv access to true mean
            mean = np.mean(n_visits)
            # Create a frozen exponential random variable with the specified scale parameter
            n_visits_user = expon(scale=mean)
        elif activity_marginal_setting == 'powerlaw':
            n_visits = data_sampler.get_activity_distribution()
            # give Adv access to true mean
            mean = np.mean(n_visits)
            alpha = 1+1/mean
            n_visits_user = pareto(alpha)
    else:
        # code for case where Adv inputs their estimated mean
        if activity_marginal_setting == 'constant':
            n_visits_user = [int(mean)]
        elif activity_marginal_setting == 'lognormal':
            sigma, median = solve_sigma_median_from_mean_skew(skew, mean)
            n_visits_user = stats.lognorm(s=sigma, scale=median)
        elif activity_marginal_setting == 'exp':
            # Create a frozen exponential random variable with the specified scale parameter
            n_visits_user = expon(scale=mean)
        elif activity_marginal_setting == 'powerlaw':
            alpha = 1+1/mean
            n_visits_user = pareto(alpha)
    return n_visits_user


def apply_log_transformation(y):
    '''
    Compresses dynamic range of input y (array-like) using logarithmic compression
    '''
    y_nonzero = [el for el in y if el > 0]
    min_y = min(y_nonzero)
    a = 1/min_y
    transform_y = [np.log(a * el + 1) for el in y]
    return transform_y/np.sum(transform_y)

def select_p_knowing_variance(dist, variance, noise_scale, scale):
    #careful, treshold parameter may run the code into an infinite loop
    #agg is the noisy version of the aggregates
    '''
    this function outputs the polynomial degree p that will be used for denoising a DP aggregate
    other parameters:
    variance?
    scale
    '''
    p = 0
    noisy_var = np.var(dist)
    while np.abs(variance - noisy_var) > variance:#/scale:
        noisy_var = np.var(apply_poly_transformation(dist, p))
        p += 0.001
    return p

def apply_poly_transformation(dist, p):
    transform_y = [el**p for el in dist]
    return transform_y/np.sum(transform_y)



def solve_sigma_median_from_mean_skew(s, mean):
    '''
    A lognormal distribution is defined by two parameters, typically given by mu and sigma (the mean and std of the log of the r.v., which is normal)
    scipy.stats uses sigma and exp(mu) instead, where exp(mu) is the median of the lognormal distribution
    Inputs:
    s: skewness. All lognormal distributions are positively skewed (large valued outliers causing heavy tail). Skewness typically varies from 1 to 5
    mean: the mean of the lognormal random variable
    Output:
    sigma: the standard deviation of the logarithm of the lognormal distribution
    median: the median of the lognormal distribution
    '''
    sigma_squared = np.log((s**2 + sqrt(s**4 + 4*s**2) + 2)**(1/3)/2**(1/3) + 2**(1/3)/(s**2 + sqrt(s**4 + 4*s**2) + 2)**(1/3) - 1)
    median = mean/np.exp(sigma_squared/2)
    return sqrt(sigma_squared), median



def apply_privacy(aggregate, k, noise_scale, group_size):
    '''
    Adversary uses this method to imitate the privacy settings on the released aggregate
    and applies this on an aggregate that they have created using synthetic traces.
    Used to estimate the mean number of visits per user in the population in estimate_true_mean
    '''
    if noise_scale is not None:
        shape = aggregate.shape
        noise = csr_matrix(np.random.laplace(scale=noise_scale, size=shape).astype(np.int8))
        aggregate+=noise
        # post-processing
        aggregate.data[(aggregate.data < 0)] = 0
        aggregate.data[(aggregate.data > group_size)] = group_size
    if k > 0:
        aggregate.data[(aggregate.data > 0) & (aggregate.data <= k)] = 0
    return aggregate


''' The following are helper functions used for creating and saving results from the experiment'''


def initialize_result_dictionaries(args):
    acc_auc_scores = {}
    membership_labels = {}
    # reserve the 0 key for parameters about the experiment
    tag = create_tag(args)
    acc_auc_scores[0] = tag
    membership_labels[0] = tag
    return acc_auc_scores, membership_labels


def create_tag(args):
    tag = (args.saved_aggregates_filename, f'Attack: {args.MIA_name}',
           f'Seed:{args.seed}', f'k:{args.bucket_threshold}', f'train_size:{args.train_size}',
           f'test_size: {args.n_groups}', f'Group size: {args.group_size}', f'PCA: {args.pca_components}',
           f'Classifier: {args.classification}', f'LR_tol: {args.LR_tol}', f'LR_C: {args.LR_C}',
           f'LR_max_iter: {args.LR_max_iter}', f'Scaler: {args.scaler_type}',
           f'partial_target_trace_mode: {args.partial_trace_mode}', f'frac_target_trace: {args.frac_target_trace}',
           f'Paired_in_out_sampling: {args.paired_in_out_sampling}', f'synthetic_traces: {args.n_synthetic_traces}',
           f'synthetic_trace_mode: {args.synthetic_trace_mode}',
           f'activity marginal: {args.activity_marginal}', f'log_compression: {args.log_compression}',
           f'validation: {args.validation_size}', f'skew: {args.skew}', f'k: {args.bucket_threshold}',
           f'DP_params: {args.DP_eps}, {args.DP_sens}', f'true_mean: {args.true_mean}',
           f'cluster_size: {args.cluster_size}')
    return tag


def save_all_results(acc_auc_scores, membership_labels, save_dir, args):
    # Filter out the key-value pair with key = 0 since this stores exp information as string
    filtered_acc_auc_scores = {k: v for k, v in acc_auc_scores.items() if k != 0}
    # Get the AUC values
    acc_auc = list(zip(*filtered_acc_auc_scores.values()))
    aucs = acc_auc[1]
    accs = acc_auc[0]
    # Calculate the mean AUC
    mean_auc = np.mean(aucs)
    mean_acc = np.mean(accs)
    print('Average accuracy across all targets: ', mean_acc)
    print('Average AUC score across all targets: ', mean_auc)
    results_directory = save_dir + f'/results_{args.MIA_name}_{args.dataset_name}/'
    acc_auc_filename, membership_labels_filename = create_save_filenames(results_directory, args)
    new_acc_auc_filename = save_results(acc_auc_scores, results_directory, acc_auc_filename, args.experiment_name)
    print(f'Accuracy and AUC scores have been saved under {new_acc_auc_filename}')
    new_membership_labels_filename = save_results(membership_labels, results_directory + 'membership_labels/',
                                                  membership_labels_filename, args.experiment_name)
    print(f'Membership labels and p scores have been saved under {new_membership_labels_filename}')


def create_save_filenames(results_directory, args):
    # main_params
    params = f'agg-{args.group_size}_n_targets-{args.n_targets}_train-{args.train_size}_val-{args.validation_size}_test-{args.n_groups}'
    # add privacy params
    if args.DP_eps is not None and args.DP_sens is not None:
        params += f'_DP_e-{args.DP_eps}_DP_sens-{args.DP_sens}_k-{args.bucket_threshold}'
    else:
        params += f'_k-{args.bucket_threshold}'
    # add partial target trace params if applicable
    if args.partial_trace_mode == "random":
        if args.frac_target_trace != 1:
            params += f"_frac_target_trace-{args.frac_target_trace}"
    elif args.partial_trace_mode == "mask":
        params += f"_target_epoch_filter-{args.deleted_epochs_season}"
    # add synthetic params if applicable
    if args.n_synthetic_traces > 0:
        params += f'_syn_traces-{args.n_synthetic_traces}_mode-{args.synthetic_trace_mode}_act_marg-{args.activity_marginal}_log-{args.log_compression}'
    if args.MIA_name == 'knock2' or args.MIA_name == 'zero_knowledge':
        if args.classification == 'RF':
            params += f"_{args.RF_n_trees}t_{args.RF_max_depth}md"
        elif args.classification == 'LR':
            params += f"_C-{args.LR_C}_tol-{args.LR_tol}"
    if args.paired_in_out_sampling == False:
        params += '_no_pair'
    if args.true_mean == True:
        params += '_true_mean'
    if args.cluster_size != 10:
        params += f'clu-{args.cluster_size}'
    if 'batch-' in args.saved_aggregates_filename:
        batch_index = args.saved_aggregates_filename.index('batch-')
        character_after_batch = args.saved_aggregates_filename[batch_index + len('batch-')]
        params += f'_batch-{character_after_batch}'
    acc_auc_filename = 'acc_auc_' + params
    membership_labels_filename = 'membership_labels_' + params
    return acc_auc_filename, membership_labels_filename


def save_results(scores, results_directory, scores_filename, experiment_name=None):
    """
    Save the dictionary of results for the given experiment
    Input:
        scores = dictionary of the form {user_id: (acc, auc), ...} or {user_id: (y_true, y_pred), ...}
        results_directory: path (string) to save directory

        scores_filename: filename to save results
    """
    # create appropriate directory if it does not exist
    if experiment_name is not None:
        results_directory = results_directory + f'{experiment_name}/'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    # ensure that we do not overwrite an existing file
    counter = 1
    new_scores_filename = results_directory + scores_filename
    while os.path.exists(new_scores_filename):
        new_scores_filename = results_directory + f"{scores_filename}_({counter})"
        counter += 1
    try:
        with open(new_scores_filename, 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return new_scores_filename
    except IOError as e:
        if e.errno == errno.ENOSPC:  # No space left on device
            # Save the file to a different directory
            alternate_results_directory = "/home/florent/Git/pald_synthetic_data/backup"
            if not os.path.exists(alternate_results_directory):
                os.makedirs(alternate_results_directory)
            alternate_filename = os.path.join(alternate_results_directory, os.path.basename(new_scores_filename))
            with open(alternate_filename, 'wb') as handle:
                pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return alternate_filename
        else:
            raise e  # Reraise the exception if it's not error code 28
    return new_scores_filename