#!/usr/bin/env python3
"""
Plot mean green values by group across days.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

def generate_plots_img_feature_by_group(criteria=['green', 'saturation', 'hue'], window=0):

    # Read the CSV data
    df = pd.read_csv('data/cilantro_stats.csv')
    df['in_bag'] = df['bagged'] == 'yes'
    df['layer'] = (df['fridge_layer'] == 'top') + \
                (df['fridge_layer'] == 'middle')*2 + \
                (df['fridge_layer'] == 'bottom')*3

    markers = {True: 'o-', False: 's--'}
    colors = ['salmon', 'orange', 'springgreen', 'royalblue']

    # Plotting
    for criterion in criteria:

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot a line for each group
        for layer, in_bag in itertools.product(list(df['layer'].unique()), list(df['in_bag'].unique())):
            group_data = df[(df['layer']==layer) & (df['in_bag']==in_bag)].sort_values('day')
            if window <= 1:
                plt.plot(group_data['day'], group_data[f'mean_{criterion}'], \
                        markers[in_bag], color=colors[layer], \
                        label=f'Layer {layer} '+('in bag' if in_bag else 'out bag'), linewidth=2)
            else:
                temp = [np.sum(group_data[f'mean_{criterion}'][i:i+window])/window for i in range(len(list(group_data['day']))-window+1)]
                plt.plot(list(group_data['day'])[window-1:], temp, \
                        markers[in_bag], color=colors[layer], \
                        label=f'Layer {layer} '+('in bag' if in_bag else 'out bag'), linewidth=2)

            plt.xlabel('Day', fontsize=12)  
            plt.ylabel(f'Mean {criterion} Value', fontsize=12)
            plt.title(f'Mean {criterion} Value by Group Across Days'+(f'\nwindow size {window}' if window > 1 else ''), fontsize=14, fontweight='bold')
            plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        # Save the plot
        window_name = f'_winsize{window}' if window > 1 else ''
        plt.savefig(f'plots/plot_mean_{criterion}_by_group{window_name}.png', dpi=300, bbox_inches='tight')
        print(f'Plot saved to: plots/plot_mean_{criterion}_by_group{window_name}.png')
        plt.close()

def generate_plot_weight_by_group(window=0):

    # Read the CSV data
    df = pd.read_csv('data/weights.csv')
    df['in_bag'] = df['bagged'] == 'yes'
    df['layer'] = (df['fridge_layer'] == 'top') + \
                (df['fridge_layer'] == 'middle')*2 + \
                (df['fridge_layer'] == 'bottom')*3

    markers = {True: 'o-', False: 's--'}
    colors = ['salmon', 'orange', 'springgreen', 'royalblue']

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot a line for each group
    for layer, in_bag in itertools.product(list(df['layer'].unique()), list(df['in_bag'].unique())):
        group_data = df[(df['layer']==layer) & (df['in_bag']==in_bag)].sort_values('day')
        if window <= 1:
            plt.plot(group_data['day'], group_data['weight']+0.05*layer, \
                    markers[in_bag], color=colors[layer], \
                    label=f'Layer {layer} '+('in bag' if in_bag else 'out bag'), linewidth=2)
        else:
            temp = [np.sum(np.array(group_data[f'weight'][i:i+window]))/window+0.05*layer for i in range(len(list(group_data['day']))-window+1)]
            if group_data['weight'].isna().any(): print(temp, group_data[f'weight'])
            plt.plot(list(group_data['day'])[window-1:], temp, \
                    markers[in_bag], color=colors[layer], \
                    label=f'Layer {layer} '+('in bag' if in_bag else 'out bag'), linewidth=2)

        plt.xlabel('Day', fontsize=12)  
        plt.ylabel(f'Weight, in g', fontsize=12)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
        plt.title(f'Weight by Group Across Days'+(f'\nwindow size {window}' if window > 1 else ''), fontsize=14, fontweight='bold')
        plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # Save the plot
    window_name = f'_winsize{window}' if window > 1 else ''
    plt.savefig(f'plots/plot_weight_by_group{window_name}.png', dpi=300, bbox_inches='tight')
    print(f'Plot saved to: plots/plot_weight_by_group{window_name}.png')
    plt.close()

if __name__ == '__main__':
    generate_plots_img_feature_by_group()
    generate_plots_img_feature_by_group(window=3)
    generate_plot_weight_by_group()
    generate_plot_weight_by_group(window=3)
