"""
Various visualisation helper functions for plotting values of various metrics.
"""
import pathlib
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def metrics_boxplot(
    data: pd.DataFrame,
    x: str = "Metric",
    quantity_name: Optional[str] = None,
    label_rotation: int = 30,
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Plots the boxplot of the metrics logged in the `data` dataframe.
    The dataframe should contain a column "Metric" with the name of the metric and a column "Value" which contains the
    values of the metrics recorded.

    Args:
        data (pd.DataFrame): The data frame containing the logged metrics.
        x: The x-axis value of the boxplot.
        quantity_name (str): The name of the quantity of interest (if one was varied and the values of metrics were
                             measured at these different points).
        label_rotation: The rotation of the metric name labels when plotting.
        file_name (Optional[Union[str, pathlib.Path]], optional): Optional name of the file to save the plot to.
                                                                  Defaults to None.
    """
    n_metrics = len(set(data["Metric"]))
    fig, ax = plt.subplots(figsize=(max(1.0, 1.0 * n_metrics), 2), dpi=200)

    b = sns.boxplot(data=data, x=x, y="Value", ax=ax)
    ax.set(
        title="Metrics" if n_metrics > 1 else list(data["Metric"])[0],
        xlabel="Metric" if quantity_name is None else quantity_name,
        ylabel="Value",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=label_rotation)
    b.tick_params(labelsize=6)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show(bbox_inches="tight")


def metric_trend(
    data: pd.DataFrame,
    quantity_name: str,
    title: str = "",
    file_name: Optional[Union[str, pathlib.Path]] = None,
) -> None:
    """
    Plots the trends in metrics stored in the data frame `data` over different values of a quantity of interest.
    It assumes that the data frame contains a column names "Metric" with the name of the metrics of interest,
    a column names "Value" with the measurement of that metric, and a column named "x" which contains the value of
    the quantity at which the value of the metric was measured.

     Args:
         data (pd.DataFrame): The data frame containing the data as described above.
         quantity_name (str): The name of the quantity of interest (which was varied and the values of metrics were
                              measured at these different points).
         title (str): The title of the figure
         file_name (Optional[Union[str, pathlib.Path]], optional): Optional name of the file to save the plot to.
                                                                   Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(4, 2), dpi=200)

    sns.lineplot(data=data, x="x", y="Value", hue="Metric", legend="full", ci="sd", ax=ax)
    ax.set(xlabel=quantity_name)
    fig.suptitle(title, fontsize=11)
    ax.get_legend().remove()
    # legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
        # fig.savefig(file_name, bbox_extra_artists=(legend,), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
