import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

def histplot_with_boxplot(df, column_name, target=None, figsize=(10, 6)):
    """
    Plots the histogram and box plot of numerical columns.
    """
    plt.figure(figsize=figsize)
    grid = GridSpec(1, 2, width_ratios=[3, 1])
    ax_hist = plt.subplot(grid[0])
    sns.histplot(df, x=column_name, hue=target, ax=ax_hist, kde=True, stat='density')
    ax_hist.set_title(f'Histogram and KDE of {column_name}', fontsize=14)
    ax_hist.set_xlabel(column_name)
    ax_hist.set_ylabel('Density')
    ax_box = plt.subplot(grid[1])
    sns.boxplot(data=df, y=column_name, hue=target, ax=ax_box, orient='h', linewidth=1)
    ax_box.set_title('Box Plot', fontsize=14)
    ax_box.set_xlabel('')
    ax_box.set_ylabel(column_name)
    
    plt.tight_layout()
    plt.show()


def barplot(df, column_name, target=None, figsize=(10,6)):
    """
    Plots a barplot for the column.
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=column_name, hue=target)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

def barplot_target(df, column_name, target, figsize=(10,6)):
    """
    Plots a barplot for the column with percentages.
    """
    plt.figure(figsize=figsize)
    counts = df.groupby([column_name, target]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages.plot(kind='bar', stacked=True, colormap='viridis', figsize=figsize)
    plt.title(f'Distribution of {column_name} by {target}')
    plt.xlabel(column_name)
    plt.ylabel('Percentage')
    plt.xticks(rotation=90)
    plt.legend(title=target)
    plt.show()

def correlation(df, figsize=(10,8), cmap='coolwarm', annot=True):
    """
    Calculates the correlation matrix for the numerical features.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def kdeplot_by_target(df, feature, target, figsize=(10, 6)):
    """
    Plots a KDE plot for the feature, grouped by the target.
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df, x=feature, hue=target, fill=True, common_norm=False, palette='Set2')
    plt.title(f'Distribution of {feature} by {target}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

def plot_histogram_with_kde(data, target, feature, figsize=(10, 4)):
    """
    Plots a histogram plot for the feature, grouped by the target.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, x=target, hue=feature,  element='step', stat='density')
    plt.title(f'Histogram with KDE of {target} and {feature}')
    plt.xlabel(target)
    plt.ylabel('Density')
    plt.show()

def scatterplot(data, target, feature1, feature2,figsize=(10, 4)):
    """
    Plots a scatter plot of two numerical features, with hue based on target.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data, x=feature1, y=feature2, hue=target)
    plt.title(f'Scatter plot of {feature1} vs {feature2} by {target}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

def scatterplot_by(data, target, feature1, figsize=(10, 4)):
    """
    Plots a scatter plot of two numerical features, with hue based on target.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data, x=feature1, by=target, hue=target)
    plt.title(f'Scatter plot of {feature1} by {target}')
    plt.xlabel(feature1)
    plt.show()

def plot_time_series(df, date_col, amount_col, account_id=None, figsize=(12, 6)):
    """
    Plots the time series of transactions.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[amount_col], label=f'Account ID: {account_id}', color='b')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title(f'{amount_col}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
