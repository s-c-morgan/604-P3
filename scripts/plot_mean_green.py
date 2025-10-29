#!/usr/bin/env python3
"""
Plot mean green values by group across days.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
df = pd.read_csv('data/cilantro_stats.csv')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot a line for each group
for group in sorted(df['group'].unique()):
    group_data = df[df['group'] == group].sort_values('day')
    plt.plot(group_data['day'], group_data['mean_green'],
             marker='o', label=f'Group {group}', linewidth=2)

plt.xlabel('Day', fontsize=12)
plt.ylabel('Mean Green Value', fontsize=12)
plt.title('Mean Green Value by Group Across Days', fontsize=14, fontweight='bold')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('plots/mean_green_by_group.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to: plots/mean_green_by_group.png")

# Show the plot
plt.show()
