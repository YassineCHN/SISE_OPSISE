import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

COLOR_SEQUENCE = [
    "#3498DB",
    "#E74C3C",
    "#2ECC71",
    "#F39C12",
    "#9B59B6",
    "#1ABC9C",
    "#E67E22",
    "#34495E",
    "#E91E63",
    "#00BCD4",
]


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str = "count",
    title: str = "",
    horizontal: bool = True,
    color_sequence: list = None,
) -> go.Figure:
    """Bar chart générique, horizontal par défaut."""
    seq = color_sequence or COLOR_SEQUENCE
    if horizontal:
        fig = px.bar(
            df, x=y, y=x, orientation="h", title=title,
            color_discrete_sequence=seq,
            labels={y: "Événements"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            df, x=x, y=y, title=title,
            color_discrete_sequence=seq,
            labels={y: "Événements"},
        )
    return fig


def pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str = "count",
    title: str = "",
    color_map: dict = None,
) -> go.Figure:
    """Donut chart avec libellés intégrés."""
    fig = px.pie(
        df, names=names, values=values, title=title,
        hole=0.4,
        color=names,
        color_discrete_map=color_map or {},
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def area_chart(
    df: pd.DataFrame,
    x: str,
    y: str = "count",
    title: str = "",
) -> go.Figure:
    """Area chart pour les timelines."""
    fig = px.area(
        df, x=x, y=y, title=title,
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={y: "Événements"},
    )
    return fig


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str = "count",
    color: str = None,
    title: str = "",
) -> go.Figure:
    """Line chart avec support couleur par catégorie."""
    return px.line(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={y: "Événements"},
    )


def heatmap(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Heatmap annotée depuis un DataFrame pivot (index = lignes)."""
    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale="Blues",
    )
    return fig
