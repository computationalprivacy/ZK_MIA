# A Zero Auxiliary Knowledge Membership Inference Attack on Aggregate Location Data.

This code repository enables users to assess the privacy of aggregate location data with the Zero Auxiliary Knowledge
membership inference attack (ZK MIA) from the paper (https://petsymposium.org/popets/2024/popets-2024-0108.php). The
repository also supports an implementation of the Knock-Knock membership inference attack (KK MIA) from the paper
(https://arxiv.org/abs/1708.06145), which uses real in-dataset location traces as an adversarial prior.

By default, this repository allows users to reproduce the results from the paper on the Milan Twitter dataset, which can
be downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/9IZALB. For ease, the raw
data is also included in this repository in the pre_processing folder. To reproduce the results, one should:

1. Ensure that the raw data social-pulse-milano.geojson is located in the repo's pre_processing folder.
2. Run preprocessing.py, which will produce two relevant files: data_dictionary_Milano.pickle and
   delaunay_nbrs_dict_Milano.pickle. The first is a dictionary of the form {user_id: location trace matrix}. It will be
   used in the next step to create the relevant location aggregates. The second is a representation of the Delaunay
   triangulation of the regions of interest (rois), which is used by the ZK MIA when running the attack.
3. Run run_sample_aggregates.py. By default, it is initialized with settings that are consistent with the experiments
   from the paper. This will create the
   file [Milano_aggregates_size-1000_n_targets-50_train-400_val-100_test-100.pickle], which contains the test aggregates
   for testing the MIAs, as well as pre-computed training and validation aggregates for KK MIA, sampled from an
   in-dataset reference of 2500 traces.
4. Finally, to run the MIA experiments from the paper, appropriate bash scripts should be run. See below for details.

*Note*: Bash scripts support two MIAs: ZK MIA and KK MIA. An additional setting has been introduced to reuse the ZK
synthetic traces aggregates across test aggregates, to significantly save computational time while reproducing similar
AUC and accuracy scores. This option is implemented by setting reuse_synthetic_traces=True with
MIA_name=zero_knowledge. Note that by default, this option creates synthetic traces from the first test aggregate, which
consistently includes the target trace. We therefore tend to observe slightly higher scores with this option. One can 
also modify the code to randomize at this level to achieve better approximations.

## 1 - Various Privacy Measures.

### 1.A - Varying the suppression thresholds.

The goal is to vary the suppression thresholds ($k=0$ to $5$).

To reproduce the results, run the following command.

```
bash scripts/varying_threshold.sh 
```

*Note 1*: You can change the threshold by modifying the value $k$ in the bash file.


### 1.B - DP Experiments.

#### 1.B.1 - DP Event Level

The goal is to vary the $\epsilon$ value between 0.1 and 10.0, with a sensitivity of 1.

To reproduce the results, run the following command.

```
bash scripts/dp_event_level.sh 
```

*Note*: You can change the threshold by modifying the value $\epsilon$ in the bash file. By default, a threshold k=1
is applied.


#### 1.B.1 - DP User-Day level

The goal is to vary the $\epsilon$ value between 0.1 and 10.0, when the sensitivity is now of 10.

To reproduce the results, run the following command.

```
bash scripts/dp_user_day.sh 
```

*Note*: You can change the threshold by modifying the value $\epsilon$ in the bash file. By default, a threshold k=1
is applied.


## 2 - Fraction of the trace.

The goal of this experiment is to reduce the knowledge of the attacker by reducing the amount of point known by the
adversary from the target trace.

To reproduce the results, run the following command.

```
bash scripts/fraction_trace.sh 
```

*Note*: You can change the fraction by modifying the value in the bash file. By default, we applied an $\espilon =1.0$
DP event level protection on the aggregate for this experiment.



