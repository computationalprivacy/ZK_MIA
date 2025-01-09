from math import ceil
from random import sample, seed, shuffle, choices, choice
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import scipy.spatial as sp
from collections import defaultdict
import time 

class DatasetSampler(object):
    
    def __init__(self, path_name, n_seed):
        """
        The DatasetSampler object stores vital information about the location dataset that is used for MIA, including but not limited to:
            - data_dictionary: a dictionary that has user_ids as dictionary keys and their traces (csr_matrices) as values
            - partial_trace_dictionary: a similar dictionary where the values are partial traces (some actual visits may be zeroed)
            - user_ids: a list of user_ids that we will sample from for reference and test set
            - n_rois, n_epochs, n_users: the number of regions of interest (ROIs), epochs, and users considered
            - seed: To ensure reproducibility
        We assume that the location dataset has already been pre-processed, and that the data is stored 
        as a dictionary of the form {user_id: trace as csr_matrix, ...}, which can be retrieved by path_name
        """
        self.data_filename = path_name
        with open(path_name, 'rb') as handle:
            data_dictionary_csr = pickle.load(handle)
        # we represent the users' location traces as np arrays
        self.data_dictionary = data_dictionary_csr
        # to facilitate the case where the adversary only has access to a partial trace
        # this dictionary will have target ids as dictionary keys and the target's partially retained traces as their values
        self.partial_trace_dictionary = {}
        print(f'Dataset has been successfully loaded from {path_name}')
        self.user_ids = list(self.data_dictionary.keys())
        print(f'The full (master) dataset includes the location data of {len(self.user_ids)} different users')
        seed(n_seed)
        np.random.seed(n_seed)
        self.seed = n_seed
        print(f'Seed for dataset sampler has been set to {n_seed}')
        self.n_rois, self.n_epochs = self.data_dictionary[1].shape
        self.n_users = len(self.user_ids)
        print(f'The dataset ranges over {self.n_rois} different ROIs and {self.n_epochs} different epochs')

    @classmethod
    def from_saved_aggregates(cls, filename, n_seed, attack, data=None):
        obj = cls.__new__(cls)
        obj.saved_aggs = True
        if data is None:
            with open(filename, 'rb') as handle:
                aggregates_dict = pickle.load(handle)
            print(f'Loaded saved aggregates from {filename}')
        else:
            aggregates_dict = data
        obj.data_dictionary = aggregates_dict['target_traces']
        targets = list(aggregates_dict['target_traces'].keys())
        obj.targets = targets
        target_1 = targets[0]
        target_in = str(target_1) + '_in'
        target_out = str(target_1) + '_out'
        n_rois, n_epochs = aggregates_dict['target_traces'][target_1].shape
        obj.n_rois = n_rois
        obj.n_epochs = n_epochs
        print(f'The dataset ranges over {n_rois} different ROIs and {n_epochs} different epochs')
        seed(n_seed)
        np.random.seed(n_seed)
        print(f'Seed for dataset sampler has been set to {n_seed}')
        obj.seed = n_seed
        obj.partial_trace_dictionary = {}
        size_substring = filename.split('size-')[1]
        group_size = int(size_substring.split('_')[0])
        obj.group_size = group_size
        obj.test_size = len(aggregates_dict['test'][target_in]) + len(aggregates_dict['test'][target_out])
        obj.test_dict = aggregates_dict['test']
        if attack == 'knock2':
            obj.ref_size = 5000
            obj.train_dict = aggregates_dict['train']
            obj.validation_dict = aggregates_dict['validation']
            obj.train_size = len(aggregates_dict['train'][target_in]) + len(aggregates_dict['train'][target_out])
            obj.validation_size = len(aggregates_dict['validation'][target_in]) + len(
                aggregates_dict['validation'][target_out])
        return obj
    
    def clean_saved_aggregates(self, attack, bucket_threshold = 0, DP_sens = None, DP_eps = None, train_size = None, validation_size = None, test_size = None, n_targets = None, paired_sampling = True, partial_trace_mode = None, frac_target_trace = 1, deleted_epochs_season = None, n_seasons_epoch = None, top_k=None): 
        '''
        Used only for the case where we are using saved aggregates. Allows subsampling of train, validation, test aggregates, enacts bucket suppression, and modifies the DatasetSampler attributes for train_dict, validation_dict, and test_dict so that the 
key is a target_ids and value is a tuple of corresponding aggregates, with all IN aggregates first, followed by all 
OUT aggregates
        '''
        assert self.saved_aggs, "This function should only be called if we are using saved aggregeates"
        # prepare new dictionaries
        if n_targets is not None and n_targets > 0:
            targets = sample(self.targets, n_targets)
            self.targets = targets
        train_dict = {}
        validation_dict = {}
        test_dict = {}
        print('applying privacy measures on saved aggregates for each target')
        # clean each target's aggregates
        for target in tqdm(self.targets):
            target_in = str(target) + '_in'
            target_out = str(target) + '_out'
            if attack == "knock2":
                if train_size is not None:
                    # subsampling aggregates with provided parameter size if applicable
                    assert train_size % 2 == 0 and train_size <= self.train_size, "cannot subsample an amount larger than the number of pre-saved aggregates or a non-even number of aggregates"
                    if paired_sampling:
                        # preserve relative ordering between in and out train aggs
                        in_train_aggregates = self.train_dict[target_in][:train_size//2] # assumes that index i of train_dict[target_in][i] and  train_dict[target_out][i] differ in one entry
                        out_train_aggregates = self.train_dict[target_out][:train_size//2]
                    else:
                        # randomly sample training aggregates
                        in_train_aggregates = sample(self.train_dict[target_in], train_size//2)
                        out_train_aggregates = sample(self.train_dict[target_out], train_size//2)
                    self.train_size = train_size
                else:
                    in_train_aggregates = self.train_dict[target_in]
                    out_train_aggregates = self.train_dict[target_out]
                # modify the target's trace for training if applicable
                if self.modify_target_trace(target, frac_target_trace, partial_trace_mode, top_k):
                    # modifies the target trace appropriately on the raw training aggregates if we are using partial target trace
                    in_train_aggregates = [
                        in_train_agg - self.data_dictionary[target] + self.partial_trace_dictionary[target] for
                        in_train_agg in in_train_aggregates]

                # the following code applies privacy measures on the raw training aggregates for the KK adversary
                if paired_sampling:
                    # iterate through each pair (order was preserved previously) and apply privacy measures to each pair
                    paired_train_aggregates = [apply_privacy_on_aggregate_pair(in_aggregate, out_aggregate, self.group_size, bucket_threshold, DP_eps, DP_sens) for (in_aggregate, out_aggregate) in zip(in_train_aggregates,out_train_aggregates)]
                    in_aggregates = [pair[0] for pair in paired_train_aggregates]
                    out_aggregates = [pair[1] for pair in paired_train_aggregates]
                    # Concatenate modified in_aggregates and out_aggregates to form train_aggregates
                    train_aggregates = in_aggregates + out_aggregates
                    train_dict[target] = train_aggregates
                else:
                    # iterate through each aggregate and apply privacy measures on each aggregate
                    in_aggregates = [apply_privacy_on_aggregate(in_aggregate, self.group_size, bucket_threshold, DP_eps, DP_sens) for in_aggregate in in_train_aggregates]
                    out_aggregates = [apply_privacy_on_aggregate(out_aggregate, self.group_size, bucket_threshold, DP_eps, DP_sens) for out_aggregate in out_train_aggregates] 
                    # Concatenate modified in_aggregates and out_aggregates to form train_aggregates
                    train_aggregates = in_aggregates + out_aggregates
                    train_dict[target] = train_aggregates
                # load validation aggregates for Knock-Knock adversary
                if validation_size is not None:
                    # subsampling aggregates with provided parameter size if applicable
                    assert validation_size % 2 == 0 and validation_size <= self.validation_size, "cannot subsample an amount larger than the number of pre-saved aggregates or a non-even number of aggregates"
                    in_validation_aggregates = sample(self.validation_dict[target_in], validation_size//2)
                    out_validation_aggregates = sample(self.validation_dict[target_out], validation_size//2)
                    self.validation_size = validation_size
                else:
                    in_validation_aggregates = self.validation_dict[target_in]
                    out_validation_aggregates = self.validation_dict[target_out]
                validation_aggregates = [apply_privacy_on_aggregate(aggregate, self.group_size, bucket_threshold, DP_eps, DP_sens) for aggregate in in_validation_aggregates + out_validation_aggregates]
                validation_dict[target] = validation_aggregates
            # done for both knock2 and zero knowledge
            if test_size is not None:
                # subsampling aggregates with provided parameter size if applicable
                assert test_size % 2 == 0 and test_size <= self.test_size, "cannot subsample an amount larger than the number of pre-saved aggregates or a non-even number of aggregates"
                in_test_aggregates = sample(self.test_dict[target_in], test_size//2)
                out_test_aggregates = sample(self.test_dict[target_out], test_size//2)
            else:
                in_test_aggregates = self.test_dict[target_in]
                out_test_aggregates = self.test_dict[target_out]
            #print(f'true mean avg: {np.mean(mean_)}') #suppress ?
            test_aggregates = [apply_privacy_on_aggregate(aggregate, self.group_size, bucket_threshold, DP_eps, DP_sens) for aggregate in in_test_aggregates + out_test_aggregates]
            test_dict[target] = test_aggregates
        self.test_dict = test_dict
        self.test_size = test_size
        self.true_mean_ = 21.54
        if attack == 'knock2':
            self.train_dict = train_dict
            self.validation_dict = validation_dict
        print('privacy measures have been applied to saved aggregates')
        
    def modify_target_trace(self, target, frac_target_trace, partial_trace_mode, top_k):
        '''
        Adv may not possess the full target trace during the inference period. This function simulates two possible scenarios:
        1) Adv is missing a random fraction of the target trace,
        2) Adv only knows the target's vists from their top k ROIs
        3)
        '''
        if partial_trace_mode is not None:
            if partial_trace_mode == "random":
                self.set_random_partial_target_trace(target, frac_target_trace)
            elif partial_trace_mode == "time_and_top_k_locations":
                self.set_top_k_target_trace(target, top_k)
            elif partial_trace_mode == "only_rois_greedy":
                self.set_onlyrois_greedy_target_trace(target)
            elif partial_trace_mode == "only_rois_top_k_locations":
                self.set_onlyrois_top_k_target_trace(target, top_k)
            return True
        return False

    def load_statistics_from_saved_aggs(self, attack):
        assert self.saved_aggs, "This function should only be called if we are using saved aggregeates"
        if attack == 'knock2':
            return self.targets, self.n_rois, self.n_epochs, self.train_size, self.validation_size, self.test_size, self.group_size
        elif attack == 'zero_knowledge':
            return self.targets, self.n_rois, self.n_epochs, self.test_size, self.group_size
    
    def load_target_information(self, target, attack):
        assert self.saved_aggs, "This function should only be called if we are using saved aggregates"
        test_aggregates = self.test_dict[target]
        test_labels = np.concatenate([np.ones(len(test_aggregates)//2), np.zeros(len(test_aggregates)//2)])
        if attack == 'knock2':
            train_aggregates = self.train_dict[target]
            train_labels = np.concatenate([np.ones(len(train_aggregates)//2), np.zeros(len(train_aggregates)//2)])
            validation_aggregates = self.validation_dict[target]
            validation_labels = np.concatenate([np.ones(len(validation_aggregates)//2), np.zeros(len(validation_aggregates)//2)])
            return train_aggregates, train_labels, validation_aggregates, validation_labels, test_aggregates, test_labels
        elif attack == 'zero_knowledge' or attack == 'baseline':
            return test_aggregates, test_labels
        else:
            return None
        
        
    def sort_users(self):
        """
        sorts self.user_ids from largest trace size to smallest
        """
        new_dic = {k: v.count_nonzero() for k,v in self.data_dictionary.items()}
        sorted_user_ids = sorted(new_dic, key=new_dic.get, reverse = True)
        print("user_ids have been sorted from largest trace size to smallest")
        self.user_ids = sorted_user_ids
    
    

    def partition_user_ids(self, k):
        """
        partitions the user_ids into k sublists. It is recommended to sort user_ids first to have meaningful partitions by mobility groups
           sorted_user_ids: a list of user_ids sorted by decreasing location trace size
        Output:
            p_list: a list of k sublists of user_ids, partitioned from sorted_user_ids
        """
        p_list = [self.user_ids[i:i + ceil(len(self.user_ids)/k)] for i in range(0, len(self.user_ids), ceil(len(self.user_ids)/k))]
        return p_list
    
    
    def set_random_partial_target_trace(self, target, alpha):
        """
        Input:
            target: user_id of the target trace that will be reduced to alpha fraction of the original trace
            alpha: fraction of target trace's non zero entries that will be retained via random sampling
        Output:
            target_trace: partial target trace obtained via random sampling alpha % of the non-zero entries of original trace
        """ 
        target_trace = self.data_dictionary[target].tolil()
        non_zero_entries = list(np.argwhere(target_trace > 0))
        m = int(len(non_zero_entries) * (1-alpha))
        deleted_entries = sample(non_zero_entries, m)
        for idx in deleted_entries:
            target_trace[idx[0], idx[1]] = 0
        self.partial_trace_dictionary[target] = target_trace
        return target_trace
    
    def set_top_k_target_trace(self, target, top_k):
        target_trace = self.data_dictionary[target].tolil()
        non_zero_entries = list(np.argwhere(target_trace > 0))
        size_trace = len(non_zero_entries)
        ROIs_with_counts =  {}
        for entry in non_zero_entries:
            ROI = entry[0]
            if ROI in ROIs_with_counts.keys():
                continue
            else:
                count = 0
                queue = []
                for other_entry in non_zero_entries:
                    if other_entry[0] == ROI:
                        queue.append(other_entry[1])
                        count += 1
                ROIs_with_counts[ROI] = (count,queue)
        with_counts_non_zero_entries = [(ROI,ROIs_with_counts[ROI][0]) for ROI in ROIs_with_counts.keys()]
        with_counts_non_zero_entries = sorted(with_counts_non_zero_entries, key= lambda x: x[1], reverse = True)
        proportion_dropped = {}
        for top_k in range(len(with_counts_non_zero_entries)):
            count_global = 0
            for (ROI, count) in with_counts_non_zero_entries[top_k:] : 
                count_global += len(ROIs_with_counts[ROI][1])
            proportion_dropped[top_k] = count_global / size_trace
        proportion_dropped[f'target {target}'] = ROIs_with_counts
        for (ROI,count) in with_counts_non_zero_entries[top_k:]:
            for epoch in ROIs_with_counts[ROI][1]:
                target_trace[ROI,epoch] = 0
        self.partial_trace_dictionary[target] = target_trace
        return target_trace

    def set_onlyrois_top_k_target_trace(self, target, top_k):
        """
        Called if Adv only has access to the target trace, with only the knowledge of the ROIs.
        The greedy version of this experiment set 1 to every epochs.
        Input:
            target: user_id of the target trace that will be reduced to alpha fraction of the original trace
        Output:
            target_trace: greedy version of the target trace, with 1 for all epochs for all the ROIs seen by the target. To only apply during training, we create a copy of it that will be return.
        """ 
        target_trace = self.data_dictionary[target].tolil()
        Rois = target_trace[:].nonzero()[0]
        max_rois = {}
        for roi in Rois:
            if roi in max_rois.keys():
                max_rois[roi]+=1
            else :
                max_rois[roi]=1
        rois = sorted(max_rois,reverse=True)
        try:
            rois = rois[:top_k]
        except:
            pass
        epochs = np.ones(self.n_epochs).tolist()
        target_trace[:] = 0
        target_trace[rois,:] = epochs
        # we keep a separate partial trace dictionary 
        self.partial_trace_dictionary[target] = target_trace
        return target_trace
    
    def set_onlyrois_greedy_target_trace(self, target):
        """
        Called if Adv only has access to the target trace, with only the knowledge of the ROIs.
        The greedy version of this experiment set 1 to every epochs.
        Input:
            target: user_id of the target trace that will be reduced to alpha fraction of the original trace
        Output:
            target_trace: greedy version of the target trace, with 1 for all epochs for all the ROIs seen by the target. To only apply during training, we create a copy of it that will be return.
        """ 
        target_trace = self.data_dictionary[target].tolil()
        Rois = target_trace[:].nonzero()[0] 
        epochs = np.ones(self.n_epochs).tolist()
        target_trace[:] = 0
        target_trace[Rois,:] = epochs
        # we keep a separate partial trace dictionary 
        self.partial_trace_dictionary[target] = target_trace
        return target_trace

    def get_activity_distribution(self):
        """
        This method returns the number of visits reported per user in the entire dataset
        Output:
            list of visits reported per user in the dataset
        """ 
        n_visits_user = []
        for user_id in self.data_dictionary:
            trace = self.data_dictionary[user_id]
            n_visits_user.append(trace.count_nonzero())
        return n_visits_user
    
    
    def sample_targets(self, p_list, n_p_list):
        """
        Input:
            p_list: a list of k sublists [p_1, ..., p_k] of the user_ids
            n_p_list: a list of k natural numbers [n_1, ..., n_k], where the ith number corresponds to the number of users to be sampled from the ith partition
        Output:
            targets: the list of users that will be attacked by an MIA, such that n_i members are sampled from the sublist of users p_i for each i=1, ..., k
        In the simplest case, we may sample n_attack_users randomly from the total list of user_ids (i.e. p_list = [self.user_ids] and n_p_list = [n_targets])
        """ 
        targets = []
        for partition, n_attack in zip(p_list, n_p_list):
            assert len(partition) >= n_attack, "the number of users selected from each partition should not exceed the size of the partition"
            targets += sample(partition, n_attack)
        return targets
    
    
    def sample_reference(self, target, ref_size):
        """
        Randomly sample a specified number of users for the attacker's reference group, which includes the target
        The remaining users will be reserved for other_users (test)
        Input:
            ref_size: number of users in the reference
            target: the user_id of the target, who will be included in the reference group
        Output:
            reference: A list of ref_size users (including the target) that makes up the attacker's reference group
            other_users: The list of all users that are not in the reference group to be used for testing
        """
        users_without_target = self.user_ids.copy()
        shuffle(users_without_target)
        users_without_target.remove(target)
        if ref_size > 0:
            reference = users_without_target[: ref_size-1] + [target]
            other_users = users_without_target[ref_size-1:]
        else:
            reference = [target]
            other_users = users_without_target
        return reference, other_users
    
    def sample_reference_static(self, ref_size):
        """
        Randomly sample a specified number of users for the attacker's reference group, which will
        be constant across all attacked targets, except for the inclusion of the target user_id
            ref_size: number of users in the reference
        Output:
            reference: A list of ref_size users that makes up the attacker's reference group
            other_users: The list of all users that are not in the reference group
        """
        user_ids = self.user_ids.copy()
        shuffle(user_ids)
        reference = user_ids[: ref_size]
        other_users = user_ids[ref_size:]
        return reference, other_users
    
    
    def include_target_in_reference(self, reference, test_set, target):
        """
        This method is to be used in conjunction with sample_reference_static. It modifies
        reference and test set such that the target will be included in reference without making 
        any unnecessary changes to the reference.
        If the target is not included in reference, replace a random user in reference with the target.
        We note that real users have a postive user_id and negative user_ids are for synthetic traces
        We ensure that we do not include any synthetic users in test_set
        """
        # in this case, the reference contains at least some real traces (positive indices are real traces, negative indices are synthetic)
        if target not in reference and len([x for x in reference if x > 0])>0:
            random_user = -1
            # ensure that a real trace is chosen
            while random_user < 0:
                random_user = choice(reference)
            reference.remove(random_user)
            test_set.append(random_user)
            test_set.remove(target)
            reference.append(target)
        # in this case, the reference contains only synthetic traces
        if target not in reference and len([x for x in reference if x > 0])==0:
            reference.append(target)
            test_set.remove(target)
        return reference, test_set
        
        
    def sample_agg_groups(self, target, users_list, group_size, n_groups, paired_in_out_sampling):
        """
        Sample the groups that are used to compute the aggregates
        The target belongs to the sampled group half of the time
        Input:
            target: the target's user_id, which must belong to half of the aggregation groups
            users_list: the list of user_ids from which the aggregation groups are sampled (either reference or test set + target)
            group_size: the number of users in each aggregation group
            n_groups: the number of aggregation groups 
            paired_in_out_sampling: boolean variable determining whether or not IN and OUT agg groups have all but one traces (target trace) overlapping
        Output:
            user_groups: list of the sampled groups (each group is a list of users)
            target_in_group: list of n_groups booleans (1 if the target is in the group, else 0)
        """
        assert n_groups % 2 == 0, "the number of aggregation groups for training and testing must be an even number to create balanced IN and OUT samples"
        #list of aggregation groups of users
        list_groups = []
        users_without_target = users_list.copy()
        shuffle(users_without_target)
        users_without_target.remove(target)
        if paired_in_out_sampling:
            for i in range(n_groups // 2):
                # sample all users but one and reserve this for both an in group and an out group
                common_users = sample(users_without_target, group_size - 1)
                in_group = common_users + [target]
                list_groups.append(in_group)
                other_users = list(set(users_without_target) - set(common_users))
                out_group = common_users + sample(other_users, 1)
                list_groups.append(out_group)
            target_in_group = np.empty((n_groups,),int)
            target_in_group[::2] = 1
            target_in_group[1::2] = 0
        else:
            # create aggregation groups that contain the target
            for i in range(n_groups // 2):
                user_group = sample(users_without_target, group_size - 1) + [target]
                list_groups.append(user_group)
            # create aggregation groups that do not contain the target
            for i in range(n_groups // 2):
                user_group = sample(users_without_target, group_size)
                list_groups.append(user_group)
        # create true labels (1 if target is a member of the aggregation group, 0 otherwise)
            target_in_group = np.append(np.ones(n_groups // 2), np.zeros(n_groups // 2))
        # shuffling labels
        # gets done downstream in Knock2 training in attacks.py
        # groups_with_labels = list(zip(list_groups, target_in_group))
        # shuffle(groups_with_labels)
        # list_groups, target_in_group = zip(*groups_with_labels)
        # # order doesn't matter for logistic regression (https://stats.stackexchange.com/questions/15307/does-the-order-of-explanatory-variables-in-logistic-regression-change-the-result)
        return list_groups, target_in_group
        

    def compute_aggregate(self, group, k, training = False, target = 0, return_suppression_stats = False, synthetic_data_dictionary={}, DP_eps = None, DP_sens = None):
        """
        Compute the raw aggregated data from the users in the aggregation groups
        Input:
            group: list of group_size users to be aggregated
            k: bucket suppression threshold
            training: boolean variable determining whether this aggregate is used for training an MIA
            target: target_id of user being attacked
            return_suppression_stats: if True, we return the fraction of entries that are suppressed 
            as well as the fraction of entries that are non-zero 
            synthetic_data_dictionary: 
        Output:
            aggregate: csr_matrix (n_rois x n_epochs) representing the aggregate trace of users in group
            fraction_suppressed: fraction of non-zero entries that have been bucket suppressed to 0
        """
        # initialize aggregate in full form
        aggregate = csr_matrix((self.n_rois, self.n_epochs), dtype=np.int8)
        # add trace from each user in group
        for user_id in group:
            if user_id > 0:
                user_trace = self.data_dictionary[user_id]
            else:
                assert training == True, "A synthetic trace may only be included in a training aggregate"
                user_trace = synthetic_data_dictionary[user_id]
            aggregate += user_trace
        # if this function is called on to produce a training aggregate, use the target's partial trace if applicable
        if training and target in group and target in self.partial_trace_dictionary:
            partial_trace = self.partial_trace_dictionary[target]
            # delete the full target trace from the aggregate that was previously added
            full_trace = self.data_dictionary[target]
            # replace full target trace with partial target trace
            aggregate -= full_trace    
            aggregate += partial_trace
        if DP_eps is not None and DP_sens is not None:
            scale = DP_sens / DP_eps
            shape = (self.n_rois, self.n_epochs)
            noise = csr_matrix(np.random.laplace(scale=scale, size=shape).astype(np.int8))
            group_size = len(group)
            aggregate += noise
            aggregate.data[(aggregate.data < 0)] = 0
            aggregate.data[(aggregate.data > group_size)] = group_size
        # apply bucket suppression with threshold k
        if k > 0:
            aggregate.data[(aggregate.data > 0) & (aggregate.data <= k)] = 0
        return aggregate

    
    
    
    def compute_aggregate_pair(self, group_pair, k, training = False, target = 0, return_suppression_stats = False, synthetic_data_dictionary={}, DP_eps = None, DP_sens = None):
        """
        Compute the aggregated data from the users in the aggregation groups
        Input:
            group_pair: tuple of (IN group, OUT group), each of group_size users to be aggregated
            k: bucket suppression threshold
            training: boolean variable determining whether this aggregate is used for training an MIA
            target: target_id of user being attacked
            as well as the fraction of entries that are non-zero 
            synthetic_data_dictionary: 
        Output:
            aggregate: csr_matrix (n_rois x n_epochs) representing the aggregate trace of users in group
            fraction_suppressed: fraction of non-zero entries that have been bucket suppressed to 0
        """
        in_group, out_group = group_pair
        in_aggregate = csr_matrix((self.n_rois, self.n_epochs), dtype=np.int8)
        out_aggregate = csr_matrix((self.n_rois, self.n_epochs), dtype=np.int8)
        #  aggregate all user traces common to both aggregate groups
        for user_id in in_group[:-1]:
            if user_id > 0:
                user_trace = self.data_dictionary[user_id] 
            else:
                assert training == True, "A synthetic trace may only be included in a training aggregate"
                user_trace = synthetic_data_dictionary[user_id]
            # use in_aggregate to accumulate common traces
            in_aggregate += user_trace
        # add last non-target user to out_aggregate
        last_trace = synthetic_data_dictionary[out_group[-1]]
        out_aggregate = in_aggregate + last_trace
        assert(target == in_group[-1])
        # add appropriate target trace to in_aggregate
        if training and target in in_group and target in self.partial_trace_dictionary:
            partial_trace = self.partial_trace_dictionary[target]
            in_aggregate += partial_trace
        else:
            full_trace = self.data_dictionary[target]
            in_aggregate += full_trace
        # apply DP if applicable
        if DP_eps is not None and DP_sens is not None:
            scale = DP_sens / DP_eps
            group_size = len(in_group)
            shape = (self.n_rois, self.n_epochs)
            noise = csr_matrix(np.random.laplace(scale=scale, size=shape).astype(np.int8))
            # add same noise to both aggregates
            in_aggregate += noise
            in_aggregate.data[(in_aggregate.data < 0)] = 0
            in_aggregate.data[(in_aggregate.data > group_size)] = group_size
            out_aggregate += noise
            out_aggregate.data[(out_aggregate.data < 0)] = 0
            out_aggregate.data[(out_aggregate.data > group_size)] = group_size
        # apply bucket suppression with threshold k
        if k > 0:
            in_aggregate.data[(in_aggregate.data > 0) & (in_aggregate.data <= k)] = 0
            out_aggregate.data[(out_aggregate.data > 0) & (out_aggregate.data <= k)] = 0
        return in_aggregate, out_aggregate


        
    def apply_delaunay_clustering(self, fname):
        """This function reads lat and lon information from a file and returns
        a dictionary linking antennas (ROIs) to neighbouring antennas (ROIs).

        Inputs:
            - fname: str, name of file containing antenna ids, lat and long
        Outputs:
            - dict with keys being integers which enumerate antennas in the order
              they appear in fname. Values are sets of antenna ids which reside in
              the towers which neighbour the tower of the antenna which is the
              dictionary entry key.
        -------
        adapted from code by AF
        """
        # we assume that the csv obeys the form site_id arr_id lon lat and uses ',' as a separator
        pdf = pd.read_csv(fname, names=['site_id', 'arr_id', 'lon', 'lat'], sep=',')
        pdf = pdf.iloc[1: , :]
        pdf['site_id'] = np.int16(pdf['site_id'])
        # filter rois if required (Note: for D4D this conveniently reduces to keeping antennas 1 to 435)
        pdf = pdf.loc[pdf['site_id'] <= self.n_rois]
        tower_dict = dict(pdf.groupby(['lon', 'lat']).indices)
        # antennas_lon_lat = {v.tolist()[0]: tuple(map(float, k)) for k,v in tower_dict.items()}
        points, ant_in_tower = zip(*tower_dict.items())
        ant_in_tower = [np.int16(i) for i in ant_in_tower]
        points = np.array(points)
        tri = sp.Delaunay(points)
        ant_neighbour_ant = defaultdict(set)
        for i in range(len(points)):
            t_neighbours = find_neighbors(i, tri)
            a = set()
            for t in t_neighbours:
                ants = ant_in_tower[t]
                a = a.union(set(ants))
                for ant in ant_in_tower[i]:
                    ant_neighbour_ant[ant] = a
        self.delaunay_cluster_dict = dict(ant_neighbour_ant)
        return dict(ant_neighbour_ant)
    
    def apply_delaunay_clustering_from_saved_dictionary(self, fname):
        with open(fname, 'rb') as f:
            self.delaunay_cluster_dict = pickle.load(f)


    def generate_synthetic_traces_unicity(self, roi_dist, epoch_dist, n_visits_user, n_traces, nbr_cores, n_unique_rois = 10):
        """
        Generates n_traces synthetic traces (stored as a dictionary with negative user_ids as keys) using marginal distributions and delaunay cluster (unicity model)
        Input:
            roi_dist: normalized frequencies for visits to each roi
            epoch_dist: normalized frequencies for visits to each epoch
            n_visits_user: list consisting of numbers of reported visits (to be sampled from directly as size of synthetic trace)
            n_traces: number of synthetic traces to be created
            nbr_cores: number of cores for parallelization
        Output:
            synthetic_data_dictionary: dictionary where keys are negative user ids (negative denotes synthetic) and value are synthetic traces
        """
        # we will use negative indices for synthetic traces to distinguish them in data_dictionary from real traces
        synthetic_data_dictionary={}
        if isinstance(n_visits_user, list):
            list_args = [(i, self.n_rois, self.n_epochs, roi_dist, epoch_dist, np.random.choice(n_visits_user), self.delaunay_cluster_dict, n_unique_rois, self.seed-i) for i in range(-n_traces, 0)]
        elif isinstance(n_visits_user, stats._distn_infrastructure.rv_frozen):
            list_args = [(i, self.n_rois, self.n_epochs, roi_dist, epoch_dist, int(n_visits_user.rvs()), self.delaunay_cluster_dict, n_unique_rois, self.seed-i) for i in range(-n_traces, 0)]
        with Pool(nbr_cores) as pool:
            results = [result for result in pool.map(generate_synthetic_trace_unicity, list_args)]
        pool.close()
        pool.join()
        for result in results:
            # result[0] is the key of the synthetic user, result[1] is its trace (csr_matrix)
            synthetic_data_dictionary[result[0]] = result[1]
        return synthetic_data_dictionary 
    


    
    
#################### Utility Functions #############################################

def generate_synthetic_trace_unicity(args):
    """
    Called by generate_synthetic_traces_unicity to create a single synthetic trace using the unicity model
    """      
    idx, n_rois, n_epochs, roi_dist, epoch_dist, n_visits, delaunay_cluster_dict, n_unique_rois, n_seed= args
    np.random.seed(n_seed)
    seed(n_seed)
    synthetic_trace = lil_matrix((n_rois, n_epochs), dtype=np.int8)
    start_roi = np.random.choice(np.arange(n_rois), p=roi_dist)
    # print('start_roi:', start_roi)
    cluster_array = gen_cluster(n_unique_rois, delaunay_cluster_dict, start_roi, n_seed)
    cluster_dist = [roi_dist[i] for i in cluster_array]
    cluster_dist = cluster_dist/np.sum(cluster_dist)
    # sample with replacement the roi and epoch observations, where rois are drawn from the connected cluster of ROIs
    # and epochs are drawn from all epochs over the inference period
    rois = np.random.choice(cluster_array, n_visits, p=cluster_dist)
    epochs = np.random.choice(np.arange(n_epochs), size=n_visits, p=epoch_dist, replace=True)
    for i,j in zip(rois, epochs):
        synthetic_trace[i, j] = 1
    return idx, synthetic_trace.tocsr()
def gen_cluster(size, ana, start, n_seed):
    """This generates a cluster of unique antennas of fixed size.

    Inputs:
        - size: int, indicates size of cluster
        - ana: delaunay_cluster_dict
        - start: starting antenna_id
        - n_seed: seed number for fixing randomizatoin

    Outputs:
        - ndarray of ints (roi_ids) which consitutes a connected path of fixed size

    -------
    adapted from code by AF
    """
    # start = rnd.sample(ana_keys, 1)[0]
    seed(n_seed)
    np.random.seed(n_seed)
    ana_keys = ana.keys()
    current_ant = start
    visited = {current_ant}
    choices = set()
    while len(visited) != size:
        choices = choices.union(ana[current_ant]) - visited
        if len(choices) == 0:  # if this happens then restart
            current_ant = sample(ana_keys, 1)[0]
            visited = {current_ant}
            choices = choices.union(ana[current_ant]) - visited
        current_ant = sample(choices, 1)[0]
        visited.add(current_ant)
    v = list(visited)
    v = np.array(v, dtype=np.int32)
    np.random.shuffle(v)
    return v
    
    
    
def find_neighbors(pindex, triang):
    """
    This function uses the built in scipy.spatial.Delaunay objects features
    to fetch the neighbouring nodes of a particular point.

    Inputs:
        - pindex: int, index of point to find neighbours for
        - triang: scipy.spatial.Delaunay object

    Outputs:
        - ndarray of inidces of points which neighbour pindex
    -------
    AF
    """
    a = triang.vertex_neighbor_vertices[1]
    b = triang.vertex_neighbor_vertices[0][pindex]
    c = triang.vertex_neighbor_vertices[0][pindex + 1]
    return a[b:c]

def apply_privacy_on_aggregate(aggregate, group_size, k, DP_eps = None, DP_sens = None):
    #apply DP first if applicable
    if DP_eps is not None and DP_sens is not None:
        scale = DP_sens / DP_eps
        shape = aggregate.shape
        noise = csr_matrix(np.random.laplace(scale=scale, size=shape).astype(np.int8))
        aggregate += noise
        # post-processing
        aggregate.data[(aggregate.data < 0)] = 0
        aggregate.data[(aggregate.data > group_size)] = group_size
    #apply bucket suppression if applicable
    if k > 0:
        aggregate.data[(aggregate.data > 0) & (aggregate.data <= k)] = 0
    return aggregate

def apply_privacy_on_aggregate_pair(in_aggregate, out_aggregate, group_size, k, DP_eps = None, DP_sens = None):
    #apply DP first if applicable
    if DP_eps is not None and DP_sens is not None:
        scale = DP_sens / DP_eps
        shape = in_aggregate.shape
        noise = csr_matrix(np.random.laplace(scale=scale, size=shape).astype(np.int8))
        in_aggregate += noise
        out_aggregate += noise
        # post-processing
        in_aggregate.data[(in_aggregate.data < 0)] = 0
        in_aggregate.data[(in_aggregate.data > group_size)] = group_size
        out_aggregate.data[(out_aggregate.data < 0)] = 0
        out_aggregate.data[(out_aggregate.data > group_size)] = group_size
    #apply bucket suppression if applicable
    if k > 0:
        in_aggregate.data[(in_aggregate.data > 0) & (in_aggregate.data <= k)] = 0
        out_aggregate.data[(out_aggregate.data > 0) & (out_aggregate.data <= k)] = 0
    return in_aggregate, out_aggregate
