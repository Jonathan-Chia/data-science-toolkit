import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_correlation_heatmap(
    df, title="Correlation Heatmap", figsize=(10, 8), cmap="coolwarm"
):
    """
    Generates a correlation heatmap for a given DataFrame using Seaborn.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title of the heatmap.
        figsize (tuple): Size of the figure (width, height).
        cmap (str): Colormap to use for the heatmap.

    Returns:
        None: Displays the heatmap.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=figsize)

    # Create the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True, square=True)

    # Add title
    plt.title(title, fontsize=16)

    # Show the plot
    plt.show()


def grouped_barplot(df, cat, subcat, val, err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.0)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(
            x + offsets[i],
            dfg[val].values,
            width=width,
            label="{} {}".format(subcat, gr),
            yerr=dfg[err].values,
        )
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, u)
    plt.legend()
    plt.show()

    # got this from stack overflow: https://stackoverflow.com/questions/42017049/how-to-add-error-bars-on-a-grouped-barplot-from-a-pandas-column


# GET two line plots and add correlation
# # Resample the data by week, summing the visitors and sales
# weekly_data = data.resample('W').sum()

# # Calculate the correlation
# correlation, _ = pearsonr(weekly_data['UNIQUE_VISITORS'], weekly_data['GROSS_SALES'])

# # Create the plot
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Plot UNIQUE_VISITORS using Seaborn
# sns.lineplot(x='DAY_OF', y='UNIQUE_VISITORS', data=weekly_data, ax=ax1, color='tab:blue', label='Unique Visitors')
# ax1.set_xlabel('Day')
# ax1.set_ylabel('Unique Visitors', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Create a second y-axis for GROSS_SALES
# ax2 = ax1.twinx()
# sns.lineplot(x='DAY_OF', y='GROSS_SALES', data=weekly_data, ax=ax2, color='tab:red', label='Gross Sales')
# ax2.set_ylabel('Gross Sales', color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# # Add the correlation to the title
# plt.title(f'Unique Visitors and Gross Sales by Week\nCorrelation: {correlation:.2f}')
# fig.tight_layout()  # Adjust the layout to make room for the title

# # Show the plot
# plt.show()
