from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_spoiler_positions_plot(
    dfs: List[pd.DataFrame], color: str, suffix: List[str]
) -> None:
    """
    Plot spoiler positions in the context

    :param dfs: list of dataframes with spoiler positions
    :param color: color of the plot
    :param suffix: list of suffixes for the plot titles
    :return: None
    """
    df_parts_all = []
    col = "spoiler parts [0 - beginning, 1 - end]"

    for df in dfs:
        spoiler_parts = []
        for index, row in df.iterrows():
            for positions in row["positions"]:
                pos = positions[0][0] + 1
                if pos == -1:
                    pos = len(row["context"])
                spoiler_parts.append(pos / len(row["context"]))

        df_parts_all.append(pd.DataFrame(spoiler_parts, columns=[col]))

    if len(dfs) == 1:
        sns.histplot(
            data=df_parts_all[0],
            x=col,
            kde=True,
            stat="density",
            color=color,
            fill=True,
        )
        if suffix is None:
            s = ""
        else:
            s = suffix[0]
        plt.title("Distribution of spoiler positions " + s)
        plt.show()
        return None

    fig, axs = plt.subplots(1, len(dfs), figsize=(15, 5))
    for i, ax in enumerate(axs):
        sns.histplot(
            data=df_parts_all[i],
            x=col,
            kde=True,
            stat="density",
            color=color,
            fill=True,
            ax=ax,
        )
        if suffix is None or i >= len(suffix):
            s = ""
        else:
            s = suffix[i]
        ax.set_title("Distribution of spoiler positions " + s)

    plt.tight_layout()
    plt.show()
