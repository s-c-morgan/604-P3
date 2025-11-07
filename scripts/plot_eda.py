import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from pathlib import Path

CRITERIA = ['value', 'saturation', 'hue', 'green']

def plot_eda_heatmaps(features=CRITERIA, save_dir=None):
    '''
    Plot heatmaps of image features for EDA purposes
    '''
    # Load&handle data
    df = pd.read_csv("data/cilantro_stats.csv")
    df['group'] = pd.to_numeric(df['group'])

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

    # Aggregate the data for the heatmaps
    heatmap_data = df.groupby(['group', 'day']).agg(
        avg_mean_saturation=('mean_saturation', 'mean'),
        avg_mean_hue=('mean_hue', 'mean'),
        avg_mean_value = ('mean_value', 'mean'),
        avg_mean_green = ('mean_green', 'mean'),
        ).reset_index()
    
    # Plot the heatmaps
    for feature in features:
        heatmap_feature = heatmap_data.pivot(
            index='group', 
            columns='day', 
            values=f'avg_mean_{feature}')
        plt.figure(figsize=(12, 8)) # Set the size of the figure
        ax = sns.heatmap(
            heatmap_feature, 
            annot=True,     # Write the data value in each cell
            fmt=".1f",      # Format numbers to one decimal place
            cmap="viridis"  # A common color map (green-to-yellow)
        )
        plt.title(f"Average {feature} by Day and Condition")
        plt.xlabel("Day")
        plt.ylabel("Experimental Group")
        if save_dir is None: plt.show()
        else: plt.savefig(f'{save_dir}/heatmap_{feature}.png', bbox_inches='tight')
        plt.close()

def plot_eda_catplot(features=CRITERIA, save_dir=None):
    '''
    Plot catplots of image features for EDA purposes
    '''
    # Load&handle data
    df = pd.read_csv("data/cilantro_stats.csv")
    df['group'] = pd.to_numeric(df['group'])

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
    bag_order = ['out of bag', 'in bag']
    location_order = ['out of fridge', 'layer 1', 'layer 2', 'layer 3']

    for feature in features:
        g = sns.catplot(
                data=df,
                kind='strip',          # Use 'strip' to plot all 11 points
                x='Location',          # The 4 groups on the x-axis
                order=location_order,  # Fix the x-axis order
                y='mean_value',        # The value to plot
                col='Bag',             # This creates the "two big types" as separate columns
                col_order=bag_order,   # Fix the column order
                hue='day',             # Color the 11 points by day
                palette='viridis',     # A good sequential colormap for days
                jitter=0.25,           # Spread the points out so they don't overlap
                s=15,
                height=4,              # Make the plot a good size
                aspect=1.2             # Make each plot a bit wider than it is tall
            )


        g.figure.suptitle(f"Mean of attribute '{feature}' Over 11 Days by Location and Bag Status", y=1.03, fontsize=16)
        g.set_axis_labels("Location", f"Mean attribute '{feature}'")
        g.set_xticklabels(rotation=25) # Rotate x-axis labels
        g.set_titles("Bag: {col_name}") # Set individual titles for the two plots
        if save_dir is None: plt.show()
        else: plt.savefig(f'{save_dir}/catmap_{feature}.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    plot_eda_heatmaps(save_dir='plots')
    plot_eda_catplot(save_dir='plots')