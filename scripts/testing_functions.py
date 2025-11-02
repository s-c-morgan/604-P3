import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm # For progression bar

def maximal_mean_dif_perm_test(data, test_attr, label_attr, n_sample=1000, \
                               with_verbose=True, return_perm_diffs=True, seed=None, ranking=False):
    '''
    Perform multi-label permutation testing of different mean value.
    Permute group labels and calculate the maximal mean difference between groups
    to approximate the null distribution. 

    Parameters
    ----------
    data : pandas.DataFrame
        input dataframe with integer index from 0 to N-1.
    test_attr : str
        attribute to be used for testing.
    label_attr : str
        attribute to be used for labelling.
    n_sample : int (default = 1000)
        number of permutations used to approximate the null distribution.
    with_verbose : bool (default = True)
        option to print progress when run.
    return_perm_diffs : bool (default = True)
        option to return the test statistics for the permuted label data.
    seed : Optional int (default = None)
        random seed for Numpy.
    ranking : bool (default = False)
        option to use the rank of the `test_attr` attribute
        instead of the attribute itself.

    Return
    ------
    dictionary
        permutation test output: maximum mean difference,
        maximum mean difference in permuted label data (if return_perm_diffs == True),
        seed (if seed is not None),
        and 2-tailed p value
    '''
    # Copy data & handle ranking
    data_ = data.copy()
    data_['_label_'] = data_[label_attr].copy()
    if ranking: data_[test_attr] = data_[test_attr].rank()
        
    # Calculate maximal mean diff
    mean_by_label = list(data_.groupby('_label_')[test_attr].mean())
    mean_by_label.sort()
    max_mean_dif_observe = mean_by_label[-1] - mean_by_label[0]
    
    # Verbose
    if with_verbose:
        value_count = data_[label_attr].value_counts()
        print('='*11+' Running permutation test ... '+'='*11)
        print(f'    Testing attribute: {test_attr}')
        print(f'    Labelling attribute: {label_attr}')
        print(f'    Unique labels: {list(value_count.index)}')
        print(f'    Unique label count: {list(value_count)}')
        print('='*52)
    
    # Main loop prep
    if seed: np.random.seed(seed)
    max_mean_dif_samples = []
    count_ge = 0

    # Main loop
    for _ in tqdm(range(n_sample)) if with_verbose else range(n_sample):

        # Generate a permutation
        shuffled_values = data_['_label_'].values
        np.random.shuffle(shuffled_values)
        data_['_label_'] = shuffled_values

        # Record mean diff
        mean_by_label = list(data_.groupby('_label_')[test_attr].mean())
        mean_by_label.sort()
        max_mean_dif_sample = mean_by_label[-1] - mean_by_label[0]
        max_mean_dif_samples.append(max_mean_dif_sample)
        if max_mean_dif_sample >= max_mean_dif_observe: count_ge += 1
        
    # Return
    output = {
        'max_mean_dif_observe' : max_mean_dif_observe,
        'p_value' : count_ge/n_sample
    }
    if return_perm_diffs: output['max_mean_dif_samples'] = max_mean_dif_samples
    if seed is not None: output['seed'] = seed
    return output

def f_test(data, test_attr, label_attr, ranking=False):
    '''
    Perform multiple-sample F-test
    '''
    # Copy data & handle ranking
    data_ = data.copy()
    if ranking: data_[test_attr] = data_[test_attr].rank()

    # Create a list of group samples
    groups = []
    for label in list(data_[label_attr].unique()):
        groups.append(list(data_[data_[label_attr]==label][test_attr]))
    f_stats_anova, p_val_anova = stats.f_oneway(*groups)
    return (f_stats_anova, p_val_anova)

def binary_perm_mean_test_blocking(data, test_attrs, label_attr, block_attrs, data_label=None, n_sample=1000, \
                                   with_verbose=True, return_perm_diffs=True, seed=None, ranking=False):
    '''
    Perform 2-label permutation testing of different mean value of the attribuite `test_attr`
    in the data, with label `label_attr` and blocking

    Parameters
    ----------
    data : pandas.DataFrame
        input dataframe with integer index from 0 to N-1.
    test_attrs : list of strs
        names of the testing attributes.
    label_attr : str
        name of the labeling attribute.
        Attribute should be int 0 or 1.
    block_attrs : list of strs
        list containing names of blocking attributes.
        Attributes should all be numerical.
    data_label : Optional pandas.DataFrame (default = None)
        input dataframe with inter index from 0 to N-1.
        refered to for label attributes and block attributes.
        Contains `label_attr` and list of `block_attrs`
        If None: set to be data
    n_sample : int (default = 1000)
        number of permutation sampling.
    with_verbose : bool (default = True)
        flag variable, True indicating printing progression bar.
    return_perm_diffs : bool (default = True)
        if True, return the differences generated by the permutations
        in a DataFrame of size (n_sample, len(test_attrs)
    seed : Optional int (default = None)
        random seed for Numpy.
    ranking : bool (dfault = False)
        use a rank test

    Output
    ------
    dictionary
        permutation test output: mean difference,
        mean difference in sampled permutations (if return_perm_diffs == True),
        and 2-tailed p value
    '''
    # Handling data_test
    if data_label is None: data_label = data
        
    # Handling blocking
    if block_attrs:

        # Create blocking index
        unique_values = [tuple(row) for _, row in data_label[block_attrs].drop_duplicates().iterrows()]
        block_ind = dict()
        for ind, value in enumerate(unique_values):
            block_ind[value] = ind
    
        # Save row indexes and count label in each blocking index
        row_inds = [[] for _ in range(len(unique_values))]
        count_label = [0]*len(unique_values)
        for index, row in data_label[block_attrs].iterrows():
            ind = block_ind[tuple(row)]
            count_label[ind] += data_label[label_attr][index]
            row_inds[ind].append(index)

    # Not blocking case
    else:
        count_label = [data_label[label_attr].sum()]
        row_inds = [data_label.index.values.tolist()]

    # Ranking
    if not ranking: testing_data = data[test_attrs]
    else: testing_data = data[test_attrs].rank(axis=0)
        
    # Calculate mean diff
    label_ind = np.array(list(data_label[label_attr] == 1))
    mean_diff = testing_data[~label_ind].mean() \
               - testing_data[label_ind].mean()
    
    # Verbose
    print('='*11+' Running permutation test ... '+'='*11)
    print(f'    Number of block: {len(count_label)}')
    print(f'    Block length: {[len(l) for l in row_inds]}')
    print(f'    Label 1 per block: {count_label}')
    print('='*52)
    
    # Main loop preparation
    if seed: np.random.seed(seed)
    if return_perm_diffs: mean_diffs = pd.DataFrame(columns=test_attrs)
    count_tail = pd.Series(0, index=test_attrs, dtype=float)
    N = data.shape[0]

    # Main loop
    for _ in tqdm(range(n_sample)) if with_verbose else range(n_sample):

        # Generate a permutation
        permutation = np.array([False]*N)
        for count, inds in zip(count_label, row_inds):
            for ind in np.random.choice(inds, size=count, replace=False):
                permutation[ind] = True

        # Record mean diff
        mean_diff_perm = testing_data[~permutation].mean() \
                        - testing_data[permutation].mean()
        if return_perm_diffs: mean_diffs.loc[len(mean_diffs)] = mean_diff_perm
        count_tail += (mean_diff.abs() - mean_diff_perm.abs() < 1e-9).astype(float)  
        
    # Return
    output = {
        'mean_diff' : mean_diff,
        'p_value' : count_tail/n_sample
    }
    if return_perm_diffs: output['mean_diff_list'] = mean_diffs
    return output

def BH_correction(df, p_col, new_p_col='p_value_adjusted'):
    '''
    Performs BH correction for multidimentional testing

    Parameters
    ----------
    df : pandas.DataFrame
        inputing data
    p_col : string
        name of the p_value column in df
    new_p_col : string
        generate a column with this name in df
        and put the adjusted value there

    Return
    ------
    pandas.Series
        panda.Series object with adjusted p values
    '''
    
    # Get p values
    p_values = [[p, ind] for ind, p in enumerate(df[p_col])]
    d = len(p_values) # dimension

    # Sort and perform BH correction
    p_values.sort()
    temp_p = 1.
    for ind in range(d, 0, -1):
        temp_p = min(p_values[ind-1][0]/ind*d, temp_p)
        p_values[ind-1][0] = temp_p

    # Put back into dataframe and return
    p_values.sort(key = lambda x:x[1])
    new_p_values = pd.Series([p for p, _ in p_values])
    df[new_p_col] = new_p_values
    return new_p_values
