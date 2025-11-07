# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project's combined notebook
#
# This notebook combines the analysis carried out for this experiment. We explore the data, attempt to conduct statistical analyses, and provide conclusions and remarks at the end.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from pathlib import Path

# %%
# Import testing and plotting functionalities
import scripts.plot_eda 
import scripts.plot_group_series
import scripts.test_bag_effect_for_infridge
import scripts.test_fridge_layer_effects 

# %% [markdown]
# ## Load Data

# %%
df = pd.read_csv("data/cilantro_stats.csv")
print(df.head())

# %% [markdown]
# ## EDA

# %%
scripts.plot_eda.plot_eda_heatmaps(features=['hue', 'saturation', 'value'])

# %%
scripts.plot_eda.plot_eda_catplot(features=['value'])

# %% [markdown]
# ## Model based method

# %%
# Mapping dictionaries
location_map = {
    1: 'out of fridge', 2: 'out of fridge',
    3: 'layer 1', 4: 'layer 1', 5: 'layer 2',
    6: 'layer 2', 7: 'layer 3', 8: 'layer 3'}
bag_map = {
    1: 'out of bag', 2: 'in bag',
    3: 'out of bag', 4: 'in bag',
    5: 'out of bag', 6: 'in bag',
    7: 'out of bag', 8: 'in bag'}

# Create feature columns
df['Location'] = df['group'].map(location_map)
df['Bag'] = df['group'].map(bag_map)

# %%
## OLS model
groups = df['group'].unique()

slope_results = []

# Loop through each group and fit a model
for group_name in groups:
    group_data = df[df['group'] == group_name]
        
    # Fit the OLS model
    # 'mean_saturation ~ day' is the formula "Mean Saturation = B0 + B1*Day"
    model = smf.ols('mean_value ~ day', data=group_data).fit()
        
    # Extract the slope coefficient for 'day'
    slope = model.params['day']
        
    slope_results.append({
        'group': group_name,
        'slope': slope
    })

df_slopes = pd.DataFrame(slope_results)
df_slopes['Location'] = df_slopes['group'].map(location_map)
df_slopes['Bag'] = df_slopes['group'].map(bag_map)

print("Successfully extracted 8 slopes:")
print(df_slopes)

# %%
## ANOVA model
model_full = smf.ols('slope ~ Location + Bag ', data=df_slopes).fit()
        
# Get the ANOVA table (Type 2 is robust)
anova_full_table = anova_lm(model_full, type=2)
        
print("\nFull Model (with Interaction):")
print(anova_full_table)

# %% [markdown]
# ## Daily difference used as data
#
# Taking avantage of the linear model assumption, we used daily changes as data to make up for the lack of data measured per one day. The two plots below hints that the layer effect may be much less significant compared to zipbag effects; therefore, data from different layers may be pulled together when investigate zipbag effects.

# %%
scripts.plot_group_series.generate_plots_img_feature_by_group(criteria=['green'], window=3, save_dir=None)
scripts.plot_group_series.generate_plot_weight_by_group(window=3, save_dir=None)

# %% [markdown]
# Permutation test and f test to investigate fridge layer effects. We only used out-of-bag data for this section, hence testing on a 10 vs 10 vs 10 3-sample data.

# %%
scripts.test_fridge_layer_effects.layer_difference_test_img_feature(criteria=['value'], save_dir=None)

# %%
scripts.test_fridge_layer_effects.layer_difference_test_weight(save_dir=None)

# %% [markdown]
# From what we have seen, we pulled the data from different layers together to test for bag effects

# %%
scripts.test_bag_effect_for_infridge.bagvsnot_test_img_feature(criteria = 'green',save_dir=None)

# %%
scripts.test_bag_effect_for_infridge.bagvsnot_test_weight(save_dir=None)

# %% [markdown]
# The zipobag effects seem to be more significant compared to the fridge layer effects.

# %% [markdown]
# ## Human model
#
# * Out-of-bag cilantros dried out and lose weight quickly; they became very light at the end. But “dried out” does not imply "not fresh" in a lot of cases, such as dried hot pepper. In the same way, not dried out does not imply fresh. The cilantro outside the fridge and in the bag started smelling clearly rotten around day 3-4, hence the experimenters took in-bag photos since day 5 instead of taking them out. This potentially gives light-refracting issues, which may partially explain the plot below.
#
# * Taking daily measurements may disturb the condition of the experiment, especially for the in-zipbag groups, where the effect of zipbags on humidity would be potentially skewed.
#
# * In general, the in-zipbag groups seem to clearly do better than the ones outside.
#
# * Layer-wise, the cilantro sprigs on the top layer looked a little freezer-burned: they are darker than the rest and had ice crystals on them. The middle layer performed the best by far: not only did they look good, but the stems also retained the stiffness (the cilantro did not lay perfectly flat on the paper). The in-zipbag group does better for this layer. The bottom layer was somewhere in between.
#
# In conclusion, we attempt to conduct analyses using objective metrics and statistical methods in the two previous section. This has some drawbacks, as the gathered data was small in size, and the measured metrics were not intuitive to measure the quality of the cilantro sprigs. There was no metric we could take that being larger would imply better quality than being smaller. Nonetheless, we were able to make observations that were consistent with what was observed subjectively by the experimenters.

# %%
scripts.plot_group_series.generate_plots_img_feature_by_group(criteria=['green'], window=3, save_dir=None)
