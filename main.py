import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import multiprocessing

def str2bool(s):
    # This is for boolean type in the parser
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2list(s):
    # Produce a list from the str passed in argument
    sub = s[1:len(s) - 1]
    l = []
    first = True
    tamp = 0
    for i, c in enumerate(sub):
        if c == ",":
            l.append(int(sub[tamp:i]))
            tamp=i+1
            continue
        if i == len(sub)-1:
            l.append(int(sub[i:i+1]))
    return l


def args_to_string(args):
    # Produce the list of arguments in a str, to be printed
    arg_strings = []
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            arg_strings.append(f'{arg_name}={arg_value}')
    return ', '.join(arg_strings)
        
def get_parser():
    parser = argparse.ArgumentParser(description=
            'Membership inference attacks against aggregate location data, introducing the zero auxiliary knowledge attack')

    # Arguments for selecting the MIA and naming/saving the experiment
    parser.add_argument('--MIA_name', type=str, help='Which MIA to run. Either knock2 or zero_knowledge', default='knock2')
    parser.add_argument("--experiment_name", help="Input the name of the experiment", type=str, default=None)
    parser.add_argument("--dataset_name", help="name of location dataset", type=str, default='')
    parser.add_argument("--save_dir",
                        help="name of the directory for saving the results of the given current experiment", type=str,
                        default="")
    parser.add_argument("--seed", help="set a seed to get reproducible results", type=int, default=2023)

    # Arguments to load files
    parser.add_argument("--saved_aggregates_filename",
                        help="filename of the pre-saved aggregates, which will be fed directly into the aggregate",
                        type=str, default="")
    parser.add_argument("--delaunay_nbrs_dict", help="file path of the pickled delaunay neighbourhood dictionary ", type=str, default = '')

    
    # Parallelization arguments
    parser.add_argument("--nbr_cores", help="Maximum number of cores the program will use", type=int, default=1)
    parser.add_argument("--chunk_size", help="Number of test aggregates to be attacked at once. Only applicable for the zero auxiliary knowledge attack", type=int,default=1)
    parser.add_argument("--nbr_cores_chunk", help="Number of cores to be used at the second level of parallelization. Only applicable for zero auxiliary knowledge attack", type=int,default=4)

    # Suppression of small counts (aka bucket suppression) arguments
    parser.add_argument("--bucket_threshold", help="The minimal number of visits for a point (roi, epoch) to avoid bucket suppression", type = int, default = 0)
    # Differential privacy arguments
    parser.add_argument("--DP_eps", help="epsilon privacy budget for DP", type = float, default = None)
    parser.add_argument("--DP_sens", help="The global sensitivity to consider for DP (event DP corresponds to 1)", type = int, default = None)

    # Arguments to select adversarial setting
    parser.add_argument("--n_targets", help="Number of targets sampled from the dataset for the MIA", type = int, default = None)
    parser.add_argument("--group_size", help="number of users m in the aggregates under attack", type=int, default=None)
    parser.add_argument("--n_groups", help="number of test aggregates attacked for each target",
                        type=int, default=100)
    parser.add_argument("--train_size",
                        help="number of aggregation groups used to train the model",
                        type=int, default=400)
    parser.add_argument("--validation_size", help="How many aggregation groups will be dedicated for validation?", type = int, default = 100)
    parser.add_argument("--paired_in_out_sampling",
                        help="should we pair all but one (target) users in each IN group with an OUT group for training aggregates?",
                        type=str2bool, default='True')


    # hyperparameters for ML binary classifier
    parser.add_argument("--classification",
                        help="Type of classifier used for MIA: Logistic Regression (LR), Random Forest (RF), Multi-layer Perceptron (MLP)",
                        type=str, default="LR")
    parser.add_argument("--LR_max_iter", help = "maximum number of iterations for Logistic Regression", type=int, default=100)
    parser.add_argument("--LR_tol", help = "tolerance for Logistic Regression, default = 1e-4", type=float, default=1e-4)
    parser.add_argument("--LR_C",
                        help="Regularization coefficient for Logistic Regression classifier. Default=1.0, smaller means more regularized",
                        type=float, default=1.0)
    parser.add_argument("--RF_n_trees", help = "Number of trees for RF classifier", type=int, default=100)
    parser.add_argument("--RF_max_depth", help = "Max depth for RF classifier", type=int, default=5)
    parser.add_argument("--scaler_type", help="Type of scaler used for classifier: Standard, MinMax", type=str,
                        default="Standard")
    parser.add_argument("--pca_components", help=" components kept by the PCA (% explained variance)", type=float,
                        default=0)

    # additional arguments for ZK MIA
    parser.add_argument("--reuse_synthetic_traces", help="whether or not to reuse the synthetic traces across each target's test aggregates", type=str2bool, default = True)
    parser.add_argument("--n_synthetic_traces", help="number of synthetic traces to be created", type=int, default=5000)
    parser.add_argument("--cluster_size", help="maximum number of unique ROIs visited per synthetic trace", type=int, default=10)
    parser.add_argument("--synthetic_trace_mode", help="How to generate synthetic data?", type=str,
                        default="unicity_marginals")
    parser.add_argument("--activity_marginal",
                        help="type of probability distribution should be used to model the activity marginal",
                        type=str, default='exp')
    parser.add_argument("--log_compression",
                        help="should we use log to compress empirical marginals from observed aggregates?",
                        type=str2bool, default='False')
    parser.add_argument("--poly_transformation",
                        help="should we use poly transformation on empirical marginals from observed aggregates?",
                        type=str2bool, default='False')
    parser.add_argument("--true_mean",
                        help="should Adv has access to the true mean number of visits per user to approximate the activity marginal?",
                        type=str2bool, default='False')
    parser.add_argument("--skew",
                        help="skewness score that Adv uses to fit the approximation of the activity marginal if using lognormal",
                        type=float, default=3.0)


    # Arguments to perform the experiment where the adversary only know a fraction of the target trace
    parser.add_argument("--frac_target_trace", help="fraction of target trace to be retained", type=float, default= 1.0)
    parser.add_argument("--partial_trace_mode", help="How target's partial trace is modeled: random, time_and_top_k_locations, only_rois_greedy, only_rois_top_k_locations", type=str, default= None)
    parser.add_argument("--top_k", help="Number of top ROIs to keep in the trace", type=int, default=None)

    return parser


def check_args(args):
    # Verify the parse arguments refer to existing code
    assert args.MIA_name in ['knock2', 'zero_knowledge'], \
        f'Invalid MIA name {args.MIA_name}'
    assert args.partial_trace_mode in ["random", "time_and_top_k_locations", "only_rois_greedy", "only_rois_top_k_locations", None], f'Invalid partial_trace_mode parameter, your parameter must be: random or mask'
    max_cores = multiprocessing.cpu_count()
    assert args.nbr_cores >= 1 and args.nbr_cores <= max_cores, f'Invalid nbr_cores parameter, your parameter must be between 1 and 40'


if __name__ == '__main__' :
    args = get_parser().parse_args()
    check_args(args)
    partial_trace_args = ['frac_target_trace', 'top_k']
    RF_args = ['RF_n_trees', 'RF_max_depth']
    synthetic_args = ['log_compression', 'activity_marginal', 'synthetic_trace_mode', 'true_mean', 'skew']

    # Automatically ensure that parameters are valid
    if args.bucket_threshold == 0:
        for arg_name in ['log_compression']:
            setattr(args, arg_name, 'False')
    if args.bucket_threshold > 0:
        if args.DP_eps is not None:
            for arg_name in ['log_compression']:
                setattr(args, arg_name, 'False')
        else:
            for arg_name in ['log_compression']:
                setattr(args, arg_name, 'True')
    if args.DP_eps is not None:
        setattr(args,'poly_transformation', 'True')


    if args.partial_trace_mode is None:
        for arg_name in partial_trace_args:
            setattr(args, arg_name, None)
    if args.classification != 'RF':
        for arg_name in RF_args:
            setattr(args, arg_name, None)
    if args.n_synthetic_traces == 0:
        print('setting synthetic arguments to None')
        for arg_name in synthetic_args:
            setattr(args, arg_name, None)

    args_string = args_to_string(args)
    print(args_string)

    from src.MIA_experiments import run_knock2_from_saved_aggs, run_zero_knowledge_from_saved_aggs
    # Always save results of an attack in a .txt or .pickle file
    save_dir = os.path.join(args.save_dir, args.MIA_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Save directory: {save_dir}')
    if args.MIA_name == 'knock2':
        run_knock2_from_saved_aggs(save_dir, args)
    elif args.MIA_name == 'zero_knowledge':
        run_zero_knowledge_from_saved_aggs(save_dir, args)
    print("finished")