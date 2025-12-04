import kaleido
import numpy as np

import plotly.express as px
import pandas as pd
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def plot_boxplots_plotly(
    arrays: list[np.ndarray], labels: list[str] = None, title: str = "Boxplot of Similarity Scores"
):
    """
    Plots interactive boxplots for a list of NumPy arrays using Plotly.

    Args:
        arrays (list of np.ndarray): Each array contains similarity scores for a group.
        labels (list of str, optional): Labels for each group. If None, uses "Group 1", "Group 2", etc.
        title (str): Plot title.

    Returns:
        plotly.graph_objects.Figure: The interactive boxplot figure.
    """
    if labels is None:
        labels = [f"Group {i + 1}" for i in range(len(arrays))]
    elif len(labels) != len(arrays):
        raise ValueError("Length of labels must match number of arrays")

    # Create a long-form DataFrame
    df = pd.DataFrame(
        {
            "score": np.concatenate(arrays),
            "group": np.concatenate([[label] * len(arr) for label, arr in zip(labels, arrays)]),
        }
    )

    fig = px.box(
        df,
        x="group",
        y="score",
        color="group",
        title=title,
        points="outliers",  # or "all" to show all points
    )

    fig.update_layout(xaxis_title="Group", yaxis_title="Similarity Score", showlegend=False)

    return fig

if __name__ == "__main__":
    # from https://colab.research.google.com/drive/1wrdeLLlVItbotCS18qsRy9HVPqCWcXX3#scrollTo=ZntsTcWq3QQe
    same_video_similarity_matrices = np.load(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/data_analysis/text_similarity/same_video_similarity_values.npy"
    )

    mean_similarity_matrix_values_array = np.load(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/data_analysis/text_similarity/random_segments_mean_similarity_matrix_values_array.npy"
    )
    same_video_similarity_matrices.shape, mean_similarity_matrix_values_array.shape


    similarity_score_distributions_compared_fig = plot_boxplots_plotly(
        [same_video_similarity_matrices[:2000], mean_similarity_matrix_values_array[:5000]],
        ["Same Video", "Random Segments"],
        title=None,
    )
    similarity_score_distributions_compared_fig

    similarity_score_distributions_compared_fig.write_html("similarity_score_distributions_compared.html")
    similarity_score_distributions_compared_fig.write_image("similarity_score_distributions_compared.png")
    # pdf
    similarity_score_distributions_compared_fig.write_image("similarity_score_distributions_compared.pdf")
