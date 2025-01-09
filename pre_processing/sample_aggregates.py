import numpy as np
import argparse
import os
import time
from math import floor
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
from multiprocessing import Pool
from random import seed, shuffle, sample
import pickle


class DatasetSampler(object):

    def __init__(self, path_name, n_seed):
        with open(path_name, 'rb') as handle:
            # obeys the format {user_id: trace as csr_matrix, ... }
            self.data_dictionary = pickle.load(handle)
        # store the list of user_ids as {1, ..., n_users} 
        self.user_ids = list(self.data_dictionary.keys())
        # set randomization seed for reproducibility
        self.seed = n_seed
        seed(n_seed)
        print(f'Seed for dataset sampler has been set to {n_seed}')
        # gather dimensions of each trace
        self.n_rois, self.n_epochs = self.data_dictionary[1].shape
        self.n_users = len(self.user_ids)
        print(f'The dataset ranges over {self.n_rois} different ROIs and {self.n_epochs} different epochs')

        
    def sample_targets(self, n_targets, min_n_visits = 10):
        '''
        Randomly sample n_targets target users for the MIA experiments such that they have at least min_n_visits visits
        '''
        self.n_targets = n_targets
        target_ids = []
        while len(target_ids) < n_targets:
            user_id = sample(self.user_ids, 1)[0]
            while user_id in target_ids:
                user_id = sample(self.user_ids, 1)[0]
            if self.data_dictionary[user_id].count_nonzero() >= min_n_visits:
                target_ids.append(user_id)
        return target_ids
        
        
    def sample_reference(self, target, ref_size):
        """
        Randomly sample a specified number of users for the attacker's reference group, which includes the target
        The remaining users will be reserved for other_users (test_set = other_users + [target])
        Input:
            ref_size: number of users in the reference
            target: the user_id of the target, who will be included in the reference group
        Output:
            reference: A list of ref_size users (including the target) that makes up the attacker's reference group
            other_users: The list of all users that are not in the reference group to be used for testing
        """
        # ensure reproducible reference group for each target
        seed(self.seed + target)
        users_without_target = self.user_ids.copy()
        shuffle(users_without_target)
        users_without_target.remove(target)
        if ref_size > 0:
            reference = users_without_target[: ref_size-1] + [target]
            other_users = users_without_target[ref_size-1:]
        return reference, other_users


    def sample_agg_groups_training(self, target, users_list, group_size, n_groups):
        """
        Sample training aggregation groups for given target, which obey paired sampling:
            Each IN aggregate has a corresponding OUT aggregegate, which varies only in one trace (target vs other)
        Input:
            target: the target's user_id, which must belong to half of the aggregation groups
            users_list: the list of user_ids from which the train aggregation groups are sampled (reference)
            group_size: the number of users in each aggregation group
            n_groups: the number of aggregation groups (half will contain target, half will not)
        Output:
            paired_groups: list of pairs of training aggregate groups [(in_1, out_1), ... ]
        """
        # ensure reproducible training aggregation groups for each target
        seed(self.seed + self.n_targets + target)
        assert n_groups % 2 == 0, "the number of aggregation groups for training and testing must be an even number to create balanced IN and OUT samples"
        paired_groups = []
        users_without_target = users_list.copy()
        shuffle(users_without_target)
        users_without_target.remove(target)
        for i in range(n_groups // 2):
            # sample all users but one and reserve this for both an in group and an out group
            common_users = sample(users_without_target, group_size - 1)
            in_group = common_users + [target]
            other_users = list(set(users_without_target) - set(common_users))
            out_group = common_users + sample(other_users, 1)
            paired_groups.append((in_group, out_group))
        return paired_groups
    
    
    def sample_agg_groups(self, target, users_list, group_size, n_groups):
        """
        Sample aggregation groups (test or validation) for given target, which are each randomly sampled
        Input:
            target: the target's user_id, which must belong to half of the aggregation groups
            users_list: the list of user_ids from which the aggregation groups are sampled (either reference or test set + target)
            group_size: the number of users in each aggregation group
            n_groups: the number of aggregation groups (half will contain target, half will not)
        Output:
            in_groups: list of the sampled training groups that contain target(each group is a list of users) 
            out_groups: list of the sampled training groups that don't contain target (each group is a list of users) 
        """
        # ensure reproducible test and validation aggregation groups for each target
        seed(self.seed + 2* self.n_targets + target)
        users_without_target = users_list.copy()
        shuffle(users_without_target)
        users_without_target.remove(target)
        assert n_groups % 2 == 0, "the number of aggregation groups for training and testing must be an even number to create balanced IN and OUT samples"
        in_groups = []
        for i in range(n_groups // 2):
            # print('Population:', len(users_without_target))
            # print('Sample size:', group_size-1)
            user_group = sample(users_without_target, group_size - 1) + [target]
            in_groups.append(user_group)
        # create aggregation groups that do not contain the target
        out_groups = []
        for i in range(n_groups // 2):
            user_group = sample(users_without_target, group_size)
            out_groups.append(user_group)
        # # create true labels (1 if target is a member of the aggregation group, 0 otherwise)
        # target_in_group = np.append(np.ones(n_groups // 2), np.zeros(n_groups // 2))
        return in_groups, out_groups
        
    
    def compute_aggregate_paired(self, in_group, out_group):
        """
        Used for computing training aggregates
        Compute the raw aggregated data from users in an IN group and an OUT group that differentiate in only one trace
        Input:
            group: list of group_size users to be aggregated
        Output:
            aggregate: csr_matrix (n_rois x n_epochs) representing the aggregate trace of users in group
        """
        # print('IN:', in_group)
        # print('OUT:', in_group)
        assert in_group[:-1] == out_group[:-1], "in_group and out_group should match except in the last element"
        group_size = len(in_group)
        common_users = in_group[:group_size-1]
        aggregate = csr_matrix((self.n_rois, self.n_epochs), dtype=np.int8)
        # aggregate all user traces where user_id is in group
        for user_id in common_users:
            aggregate += self.data_dictionary[user_id] 
        target = in_group[-1]
        in_aggregate = aggregate + self.data_dictionary[target] 
        other_user = out_group[-1]
        out_aggregate = aggregate + self.data_dictionary[other_user] 
        return in_aggregate, out_aggregate


    def compute_aggregate(self, group):
        """
        Compute the raw aggregated data from the users in the aggregation group
        Input:
            group: list of users, whose traces will be aggregated
        Output:
            aggregate: csr_matrix (n_rois x n_epochs) representing the aggregate trace of users in group
        """                                    
        aggregate = csr_matrix((self.n_rois, self.n_epochs), dtype=np.int8)
        # aggregate all user traces where user_id is in group
        for user_id in group:
            aggregate += self.data_dictionary[user_id] 
        return aggregate
            

def process_group(data_sampler, group):
    '''
    Parallelization helper function used for creating either a validation or test aggregate from a group of users
    '''
    aggregate = data_sampler.compute_aggregate(group)
    return aggregate

            
def process_training_group(data_sampler, in_group, out_group):
    '''
    Parallelization helper function used for creating a pair of training aggregates (IN and OUT) from a pair of groups of users
    '''
    in_aggregate, out_aggregate = data_sampler.compute_aggregate_paired(in_group, out_group)
    return in_aggregate, out_aggregate 

    
def create_aggregates_target(data_sampler, target, agg_group_size, ref_size, train_size, validation_size, test_size, num_cores):
    '''
    Creates all of the train, validation, and test aggregates for the specified target and agg_group size
    '''
    #create a reference and test for the given target
    reference, other_users = data_sampler.sample_reference(target, ref_size)
    # print('size of other users:', len(other_users))
    # create paired IN/OUT user groups for training aggregates from reference group
    paired_groups = data_sampler.sample_agg_groups_training(target, reference, agg_group_size, train_size)
    paired_group_args = [(data_sampler, group_pair[0], group_pair[1]) for group_pair in paired_groups]
    print(f'creating train aggregates for target {target}')
    # compute train aggregates and split them into IN and OUT
    with Pool(num_cores) as pool:
        results = pool.starmap(process_training_group, paired_group_args)
    in_train_aggregates, out_train_aggregates = zip(*results)
    # create randomly sampled IN/OUT user groups for validation aggregates from reference group
    in_validation_groups, out_validation_groups = data_sampler.sample_agg_groups(target, reference, agg_group_size, validation_size)
    print(f'creating validation aggregates for target {target}')
    # create IN and OUT validation aggregates
    with Pool(num_cores) as pool:
        in_val_aggregates = pool.starmap(process_group, [(data_sampler, group,) for group in in_validation_groups])
    with Pool(num_cores) as pool:
        out_val_aggregates = pool.starmap(process_group, [(data_sampler, group,) for group in out_validation_groups])
    # create randomly sampled IN/OUT user groups for test aggregates from other_users + [target]
    in_test_groups, out_test_groups = data_sampler.sample_agg_groups(target, other_users + [target], agg_group_size, test_size)
    print(f'creating test aggregates for target {target}')
    # create IN and OUT test aggregates
    with Pool(num_cores) as pool:
        in_test_aggregates = pool.starmap(process_group, [(data_sampler, group,) for group in in_test_groups])
    with Pool(num_cores) as pool:
        out_test_aggregates = pool.starmap(process_group, [(data_sampler, group,) for group in out_test_groups])
    return in_train_aggregates, out_train_aggregates, in_val_aggregates, out_val_aggregates, in_test_aggregates, out_test_aggregates 

# data_sampler = DatasetSampler('data_dictionaries/old_data_dictionary_Milano.pickle',42)
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dictionary_pickle_name', type = str, help='Path to pickled data_dictionary of user traces produced from obtain_dictionaries')   
    parser.add_argument('--dataset_name', type = str, help='Dataset name')
    parser.add_argument('--save_dir', type = str, help='file path to where aggregates folder will be created') 
    parser.add_argument('--n_seed', type=int, default='2023', help='Seed number for reproducibility')
    parser.add_argument('--num_processes', type=int, default='1', help='Number of processes for parallelization')
    parser.add_argument('--n_targets', type=int, default='50', help='Number of targets')
    parser.add_argument('--agg_group_size', type=int, default='1000', help='Size of aggregates')
    parser.add_argument('--ref_size', type=int, default='5000', help='Adversary reference size')
    parser.add_argument('--train_size', type=int, default='1000', help='Number of training groups')
    parser.add_argument('--test_size', type=int, default='200', help='Number of test groups')
    parser.add_argument('--validation_size', type=int, default='200', help='Number of validation groups')
    args = parser.parse_args()
    # set up dictionary to store all aggregates for the given agg_group_size
    aggregates_agg_dict = {}
    # initialize data_sampler object that will be shared across methods
    # global data_sampler
    data_sampler = DatasetSampler(args.data_dictionary_pickle_name, args.n_seed)
    targets = data_sampler.sample_targets(args.n_targets, min_n_visits = 10)
    # initialize subdictionaries for train, validation, and test
    aggregates_agg_dict['train'] = {}
    aggregates_agg_dict['validation'] = {}
    aggregates_agg_dict['test'] = {}
    aggregates_agg_dict['target_traces'] = {}
    for target in tqdm(targets):
        # save shuffled target trace
        aggregates_agg_dict['target_traces'][target] = data_sampler.data_dictionary[target]
        # print('target trace:', data_sampler.data_dictionary[target])
        # compute all relevant aggregates
        in_train_aggregates, out_train_aggregates, in_val_aggregates, out_val_aggregates, in_test_aggregates, out_test_aggregates = create_aggregates_target(data_sampler, target, args.agg_group_size, args.ref_size, args.train_size, args.validation_size, args.test_size, args.num_processes)
        # save them to respective dictionaries
        target_in = str(target) + '_in'
        target_out = str(target) + '_out'
        aggregates_agg_dict['train'][target_in] = in_train_aggregates
        aggregates_agg_dict['train'][target_out] = out_train_aggregates
        aggregates_agg_dict['validation'][target_in] = in_val_aggregates
        aggregates_agg_dict['validation'][target_out] = out_val_aggregates
        aggregates_agg_dict['test'][target_in] = in_test_aggregates
        aggregates_agg_dict['test'][target_out] = out_test_aggregates
        aggregates_agg_dict['info'] = f'({args.dataset_name} ,{args.agg_group_size}, {args.n_targets}, {args.n_seed}, {args.ref_size}, {args.train_size}, {args.validation_size}, {args.test_size})'
    save_results(aggregates_agg_dict, args.dataset_name, args.save_dir, args.agg_group_size, args.n_targets, args.train_size, args.validation_size, args.test_size)

def save_results(aggregates_agg_dict, dataset_name, save_dir, agg_group_size, n_targets, train_size, validation_size, test_size):
    '''
    Saves aggregates to the appropriate directory
    Inputs:
    aggregates_agg_dict: dictionary storing all of the relevant aggregates
    dataset_name: name of dataset
    save_dir: str file path to where aggregates folder will be created 
    agg_group_size: the number of user traces in each aggregate (only parameter that will be varied for different runs)
    n_targets: the number of targets
    train_size: the number of train aggregates
    validation_size: the number of validation aggregates
    test_size: the number of test aggregates
    '''
    output_directory = save_dir + f'aggregates/'
    # create directory for saving aggregates if it doesn't already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, mode=0o755, exist_ok=True)
    save_filename = f'{dataset_name}_aggregates_size-{agg_group_size}_n_targets-{n_targets}_train-{train_size}_val-{validation_size}_test-{test_size}.pickle'
    output_file = os.path.join(output_directory, save_filename)
    with open(output_file,'wb') as f:
        pickle.dump(aggregates_agg_dict, f)
    print(f'All aggregates of size {agg_group_size} for all targets has been saved under {output_file}')

    
if __name__ == '__main__':
    main()