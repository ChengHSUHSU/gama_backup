import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from sklearn.linear_model import SGDClassifier
from bdds_recommendation.src.feature_visualization._base_vis_conf_ import FIGSIZE
from bdds_recommendation.utils.logger import logger


def plot_boxplot(
    data: list,
    label: str,
    title: str = 'BoxplotFigure',
):
    """Draw boxplot function. Only support one boxplot in one figure.

    Args:
        df (ArrayLike): Support ArrayLike data, such as dataframe or list.
        labels (list): name of dataframe_col to show on diagram.
        title (str, optional): Figure name show on the top of diagram. Defaults to 'BoxplotFigure'.
    Examples:
        1. plot_boxplot([1, 2, 3, 4, 5, 4, 3, 2, 1], 'col_name')

    """
    fig = plt.figure(figsize=FIGSIZE)
    plt.boxplot(data, showmeans=True)
    plt.xticks([1], [label])
    plt.title(title, figure=fig)
    return fig


def plot_distribution_diagram(
    data: list,
    labels: list,
    orientation: str = 'vertical',
    width_of_bar: float = 0.3,
    title: str = 'DistributionFigure',
):
    """Draw distribution histogram. Only support one data in one distribution histogram.

    Args:
        data (list): list type data. Note that input data should transform to statistics data first.
        labels (list): name of dataframe_col to show on diagram.
        orientation (str, optional): Orientation of Diagram, can be vertical or horizontal. Defaults to 'vertical'.
        width_of_bar (float, optional): Width of bar in diagram. Defaults to 0.3.
        title (str, optional): Figure name show on the top of diagram. Defaults to 'DistributionFigure'.
    """
    fig = plt.figure(figsize=FIGSIZE)
    plt.title(title, figure=fig)
    align = 'center'
    ax = plt.subplot()
    ax.set_xticks(labels)

    if orientation == 'horizontal':
        plt.barh(labels, data, height=width_of_bar, align=align, figure=fig)

    elif orientation == 'vertical':
        plt.bar(labels, data, width=width_of_bar, align=align, figure=fig)
    
    else:
        logger.info('[Visualize Diagram] Setting orientation Error!')

    return fig


def plot_continuous_distribution_diagram(
    data: pd.DataFrame,
    bins: int = 10,
    density: bool = False,
    cumulative: bool = False,
    title: str = 'ContinuousDistributionFigure',
):
    """Draw continuous diagram. Support one data in one continuous distribution fig.

    Args:
        data (pd.DataFrame): Data should be a column data of dataframe, example: df[col], and this data should be continuous numerical data.
        bins (int, optional): bins to draw diagram. Defaults to 10.
        density (bool, optional): Normalize data to 1. Defaults to False.
        cumulative (bool, optional): Compute all bins for smaller values. Defaults to False.
        title (str, optional): Diagram Title. Defaults to 'ContinuousDistributionFigure'.
    """
    fig = plt.figure(figsize=FIGSIZE)
    plt.title(title, figure=fig)
    plt.xticks(np.arange(0, bins, step=0.5), figure=fig)
    plt.hist(data, bins, density=density, cumulative=cumulative, figure=fig)
    return fig


def plot_feature_correlation(
    data: pd.DataFrame,
    title: str = 'TestFeatureCorrelation',
):
    """Draw feature correlations. 

    Args:
        data (pd.DataFrame): All features need to calculate correlations.
        title (str, optional): Diagram Title. Defaults to 'TestFeatureCorrelation'.
    """
    fig = plt.figure(figsize=FIGSIZE)
    heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
    plt.title(title, figure=fig)
    return fig


def plot_feature_importance(
    data_x: pd.DataFrame,
    data_y: pd.DataFrame.columns,
    feature_names: list,
    title: str = 'TestFeatureCorrelation',
):
    """Draw feature importance.

    Args:
        X_train (pd.DataFrame): Features in dataframe.
        Y_train (pd.DataFrame.columns): Ground truth of features in dataframe.
        feature_names (list): Each feature name(without 'y').
        title (str, optional): Diagram Title. Defaults to 'TestFeatureCorrelation'.
    """
    model = SGDClassifier()
    model.fit(data_x, data_y)
    feature_importance = model.coef_[0]

    fig = plt.figure(figsize=FIGSIZE)
    plt.barh(feature_names, feature_importance, color='red', figure=fig)
    plt.xticks(range(0, round(max(feature_importance)) + 10, 10), figure=fig) # only normalize positive maximum value to 10

    for i, v in enumerate(feature_importance):
        plt.text(v, i, str(round(v, ndigits=2)), color='blue', fontweight='bold', figure=fig)

    plt.title(title, figure=fig)
    return fig


def calculate_appearance_times(data: pd.DataFrame.columns):
    """Transform un-statistic pandas dataframe data to statistics list.
    Example: [1, 2, 4, 4, 3, 2] -> index: [1, 2, 3, 4], counts: [1, 2, 1, 2]

    Args:
        data (pd.DataFrame.columns): Column data of dataframe.

    Returns:
        index (list): Denote element in input data.
        count (list): Denote each counts of elements.
    """
    index = pd.value_counts(data).keys().tolist()
    counts = pd.value_counts(data).tolist()
    return index, counts


def visualize_features(df: pd.DataFrame, configs: dict):
    """Visualize statistics diagram of dataframe.

    Args:
        df (pd.DataFrame): Dataframe after preprocess step.
        configs (dict): Setting figures type and columns need to draw.

    Returns:
        Dict: return all figures in dict.
    """
    return_dict = {
        "boxplot": {},
        "distribution_histogram": {},
        "continuous_distribution_histogram": {},
        "feature_correlation": {},
        "feature_importance": {},
    }

    for plot_type, settings in configs.items():
        cols = settings['cols']

        if plot_type == 'boxplot':
            for col in cols:
                index, count = calculate_appearance_times(df[col])
                try:
                    box_plot = plot_boxplot(
                        index,
                        col,
                        title=f'{plot_type}_{col}'
                    )
                    return_dict[plot_type][col] = box_plot

                except Exception:
                    logger.info(traceback.format_exc())

        elif plot_type == 'distribution_histogram':
            for col in cols:
                index, count = calculate_appearance_times(df[col])
                try:
                    distribution_hist = plot_distribution_diagram(
                        count,
                        index,
                        title=f'{plot_type}_{col}'
                    )
                    return_dict[plot_type][col] = distribution_hist

                except Exception:
                    logger.info(traceback.format_exc())

        elif plot_type == 'continuous_distribution_histogram':
            for col in cols:
                try:
                    continuous_histogram = plot_continuous_distribution_diagram(
                        df[col],
                        title=f'{plot_type}_{col}'
                    )
                    return_dict[plot_type][col] = continuous_histogram

                except Exception:
                    logger.info(traceback.format_exc())

        elif plot_type == 'feature_correlation':
            try:
                feature_correlation = plot_feature_correlation(
                    df[cols],
                    title=f'{plot_type}'
                )
                return_dict[plot_type] = feature_correlation

            except Exception:
                logger.info(traceback.format_exc())

        elif plot_type == 'feature_importance':
            try:
                y_col = settings['y_col']
                feature_importance = plot_feature_importance(
                    df[cols],
                    df[y_col],
                    feature_names=cols,
                    title=f'{plot_type}'
                )
                return_dict[plot_type] = feature_importance

            except Exception:
                logger.info(traceback.format_exc())
        
        else:
            logger.info(f'Not Support {plot_type} Figure Type!')
    return return_dict


def save_result(figs: dict, base_path: str = '/checkpoints/experiment_name'):
    """Save final result to path.

    Args:
        all_figs (dict): Figures in dict.
        base_path (str, optional): Save path. Defaults to '/checkpoints/experiment_name'.
    """
    for plot_type, data in figs.items():
        if isinstance(data, dict):
            for col, fig in data.items():
                fig.savefig(f'{base_path}/{plot_type}_{col}.jpg')
        else:
            data.savefig(f'{base_path}/{plot_type}.jpg')
