import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from testing_functions import maximal_mean_dif_perm_test, binary_perm_mean_test_blocking, f_test
from utils import calculate_delta

df_colors = pd.read_csv('data/cilantro_stats.csv')
df_weights = pd.read_csv('data/weights.csv')

criteria = ['saturation', 'hue', 'value', 'green']

def plot_difference_bagvsnot_img_feature(criteria=criteria, save_dir='plots'):
    '''
    Plot the histogram of daily differences in image features for in-fridge groups
    colored by bagged vs unbagged data

    Parameters
    ----------
    criteria :  str or list[str]
        features to be plotted.
        Must be an element or a sub-list of ['saturation', 'hue', 'value', 'green']

    save_dir : Optional(str)
        directory to save data (default to be 'plots')
        if is None, instead of saving, show the image.
    '''
    if type(criteria) == str: criteria = [criteria]

    for criterion in criteria:

        # Get differential data
        df_colors_dif = df_colors[['fridge_layer', 'bagged', 'day', f'mean_{criterion}']]
        df_colors_dif_fridged = calculate_delta(df_colors_dif[df_colors_dif['fridge_layer']!='OOF'],
                                                label_attrs=['fridge_layer', 'bagged'], record_attr=f'mean_{criterion}', time_attr='day', n=11, dif_name='dif')
        
        # Plot histogram
        data_bagged = list(df_colors_dif_fridged[df_colors_dif_fridged['bagged']=='yes']['dif'])
        data_unbagged = list(df_colors_dif_fridged[df_colors_dif_fridged['bagged']=='no']['dif'])
        left = min(np.min(data_bagged), np.min(data_unbagged))
        right = max(np.max(data_bagged), np.max(data_unbagged))
        bins_count = 15
        epsilon = (left-right)/bins_count/10.
        bins = np.linspace(left-epsilon, right, bins_count+1)
        plt.hist(data_bagged, bins=bins, label='In bag', alpha=0.5, color='orange')
        plt.hist(data_unbagged, bins=bins+epsilon, label='Out of bag', alpha=0.5, color='springgreen')
        plt.title(f'Histogram of in bag vs out of bag \n{criterion} deltas for fridged cilantro sprigs')
        plt.legend()
        if save_dir is not None: plt.savefig(f'{save_dir}/hist_delta_fridged_{criterion}.png', bbox_inches='tight')
        else: plt.show()
        plt.close()

def bagvsnot_test_img_feature(criteria=criteria, save_dir='plots'):
    '''
    Test the daily differences in image features for all in-fridge groups
    constrasting bagged and un-bagged groups
    conduct a permutation test and an f-test

    Parameters
    ----------
    criteria :  str or list[str]
        features to be plotted.
        Must be an element or a sub-list of ['saturation', 'hue', 'value', 'green']

    save_dir : Optional(str)
        directory to save data (default to be 'plots')
        if is None, instead of saving, show the image.
    '''
    if type(criteria) == str: criteria = [criteria]
    final_output = dict()
    for criterion in criteria:

        # Get differential data
        df_colors_dif = df_colors[['fridge_layer', 'bagged', 'day', f'mean_{criterion}']]
        df_colors_dif_fridged = calculate_delta(df_colors_dif[df_colors_dif['fridge_layer']!='OOF'],
                                                label_attrs=['fridge_layer', 'bagged'], record_attr=f'mean_{criterion}', time_attr='day', n=11, dif_name='dif')
        
        # Run permutation test
        output = maximal_mean_dif_perm_test(data=df_colors_dif_fridged, test_attr='dif', label_attr='bagged', n_sample=10000,
                                            with_verbose=True, return_perm_diffs=True, seed=True, ranking=False)
        
        # Run f test
        (f_stats_anova, p_val_anova) = f_test(data=df_colors_dif_fridged, test_attr='dif', label_attr='bagged', ranking=False)
        print(f'ANOVA F-statistic: {f_stats_anova}, ANOVA P-value: {p_val_anova}')

        # Plotting
        plt.hist(output['max_mean_dif_samples'], bins=30, color='skyblue')
        plt.title(f'Abs. mean {criterion} dif.  for fridged not-bagged vs bagged cilantros\np value: {output["p_value"]:.4f}; f-test p value: {p_val_anova:.4f}')
        plt.axvline(x=output['max_mean_dif_observe'], label='Observed', color='salmon')
        plt.legend()
        if save_dir is not None: plt.savefig(f'{save_dir}/mean_dif_test_{criterion}_baggedvsnot.png', bbox_inches='tight')
        else: plt.show()
        plt.close()

        # Save for return
        final_output[criterion] = (output["p_value"], p_val_anova)

    return final_output

def plot_difference_bagvsnot_weight(save_dir='plots'):
    '''
    Plot the histogram of daily differences in weights for in-fridge groups 
    colored by bagged vs unbagged data

    Parameters
    ----------
    save_dir : Optional(str)
        directory to save data (default to be 'plots')
        if is None, instead of saving, show the image.
    '''

    # Get differential data
    df_weights_dif = df_weights[['fridge_layer', 'bagged', 'day', 'weight']]
    df_weights_dif_fridged = calculate_delta(df_weights_dif[df_weights_dif['fridge_layer']!='OOF'],
                                             label_attrs=['fridge_layer', 'bagged'], record_attr='weight', time_attr='day', n=11, dif_name='dif')
    
    # Plot histogram
    data_bagged = list(df_weights_dif_fridged[df_weights_dif_fridged['bagged']=='yes']['dif'])
    data_unbagged = list(df_weights_dif_fridged[df_weights_dif_fridged['bagged']=='no']['dif'])
    left = min(np.min(data_bagged), np.min(data_unbagged))
    right = max(np.max(data_bagged), np.max(data_unbagged))
    bins_count = 15
    epsilon = (left-right)/bins_count/10.
    bins = np.linspace(left-epsilon, right, bins_count+1)
    plt.hist(data_bagged, bins=bins, label='In bag', alpha=0.5, color='orange')
    plt.hist(data_unbagged, bins=bins+epsilon, label='Out of bag', alpha=0.5, color='springgreen')
    plt.title('Histogram of in bag vs out of bag \nweight deltas for fridged cilantro sprigs')
    plt.legend()
    if save_dir is not None: plt.savefig(f'{save_dir}/hist_delta_fridged_weight.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

def bagvsnot_test_weight(save_dir='plots'):
    '''
    Test the daily differences in weights for all in-fridge groups
    constrasting bagged and un-bagged groups
    conduct a permutation test and an f-test

    Parameters
    ----------
    save_dir : Optional(str)
        directory to save data (default to be 'plots')
        if is None, instead of saving, show the image.
    '''
    # Get differential data
    df_weights_dif = df_weights[['fridge_layer', 'bagged', 'day', 'weight']]
    df_weights_dif_fridged = calculate_delta(df_weights_dif[df_weights_dif['fridge_layer']!='OOF'],
                                             label_attrs=['fridge_layer', 'bagged'], record_attr='weight', time_attr='day', n=11, dif_name='dif')
    
    # Run permutation test
    output = maximal_mean_dif_perm_test(data=df_weights_dif_fridged, test_attr='dif', label_attr='bagged', n_sample=10000,
                                        with_verbose=True, return_perm_diffs=True, seed=True, ranking=False)

    # Run f test
    (f_stats_anova, p_val_anova) = f_test(data=df_weights_dif_fridged, test_attr='dif', label_attr='bagged', ranking=False)
    print(f'ANOVA F-statistic: {f_stats_anova}, ANOVA P-value: {p_val_anova}')

    # Plotting
    plt.hist(output['max_mean_dif_samples'], bins=50, color='skyblue')
    plt.title(f'Abs. mean weight dif. for fridged not-bagged vs bagged cilantros\np value: {output["p_value"]:.4f}; f-test p value: {p_val_anova:.4f}')
    plt.axvline(x=output['max_mean_dif_observe'], label='Observed', color='salmon')
    plt.legend()
    if save_dir is not None: plt.savefig(f'{save_dir}/mean_dif_test_weights_baggedvsnot.png', bbox_inches='tight')
    else: plt.show()
    plt.close()

    # Returns
    return (output["p_value"], p_val_anova)


if __name__ == '__main__':
    plot_difference_bagvsnot_img_feature()
    bagvsnot_test_img_feature()
    plot_difference_bagvsnot_weight()
    bagvsnot_test_weight()
