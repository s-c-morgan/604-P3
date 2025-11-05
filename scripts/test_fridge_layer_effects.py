import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from testing_functions import maximal_mean_dif_perm_test, binary_perm_mean_test_blocking, f_test
from utils import calculate_delta

df_colors = pd.read_csv('data/cilantro_stats.csv')
df_weights = pd.read_csv('data/weights.csv')

criteria = ['saturation', 'hue', 'value', 'green']

def layer_difference_test_img_feature(criteria=criteria, save_dir='plots'):
    '''
    Perform hypothesis testing of weather different layers express difference in image features.
    Run a maximum-mean-different permutation test and an f-test, and then save/show the output plot of the null permutation distribution.

    Parameters
    ----------
    criteria :  str or list[str]
        features to be used to run permutation test and f-test
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
        
        # Run permutation test
        output = maximal_mean_dif_perm_test(data=df_colors_dif_fridged[df_colors_dif_fridged['bagged']=='no'], test_attr='dif', label_attr='fridge_layer', n_sample=10000,
                                            with_verbose=True, return_perm_diffs=True, seed=True, ranking=False)
        
        # Run f test
        (f_stats_anova, p_val_anova) = f_test(data=df_colors_dif_fridged[df_colors_dif_fridged['bagged']=='no'], test_attr='dif', label_attr='fridge_layer', ranking=False)
        print(f'ANOVA F-statistic: {f_stats_anova}, ANOVA P-value: {p_val_anova}')

        # Plotting
        plt.hist(output['max_mean_dif_samples'], bins=30, color='skyblue')
        plt.title(f'Max mean difference in {criterion} for fridged not-bagged cilantros\np value: {output["p_value"]:.4f}; f-test p value: {p_val_anova:.4f}')
        plt.axvline(x=output['max_mean_dif_observe'], label='Observed', color='salmon')
        plt.legend()
        if save_dir is not None: plt.savefig(f'{save_dir}/max_mean_dif_test_{criterion}_unbagged.png')
        else: plt.show()
        plt.close()
    

def layer_difference_test_weight(save_dir='plots'):
    '''
    Perform hypothesis testing of weather different layers express difference in weighting.
    Run a maximum-mean-different permutation test and an f-test, and then save/show the output plot of the null permutation distribution.

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
    output = maximal_mean_dif_perm_test(data=df_weights_dif_fridged[df_weights_dif_fridged['bagged']=='no'], test_attr='dif', label_attr='fridge_layer', n_sample=10000,
                                        with_verbose=True, return_perm_diffs=True, seed=True, ranking=False)

    # Run f test
    (f_stats_anova, p_val_anova) = f_test(data=df_weights_dif_fridged[df_weights_dif_fridged['bagged']=='no'], test_attr='dif', label_attr='fridge_layer', ranking=False)
    print(f'ANOVA F-statistic: {f_stats_anova}, ANOVA P-value: {p_val_anova}')

    # Plotting
    plt.hist(output['max_mean_dif_samples'], bins=50, color='skyblue')
    plt.title(f'Max mean difference in weights for fridged not-bagged cilantros\np value: {output["p_value"]:.4f}; f-test p value: {p_val_anova:.4f}')
    plt.axvline(x=output['max_mean_dif_observe'], label='Observed', color='salmon')
    plt.legend()
    if save_dir is not None: plt.savefig(f'plots/max_mean_dif_test_weights_unbagged.png')
    else: plt.show()
    plt.close()

if __name__ == '__main__':
    layer_difference_test_img_feature()
    layer_difference_test_weight()


