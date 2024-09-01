import base64
import io
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as plx
import matplotlib.pyplot as plt
import pandas as pd
from dash.dependencies import Input, Output, State
from app import app
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import problexity as px


def layout():
    return dbc.Container(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Input Feature:"),
                            dcc.Dropdown(id='input-feature-dropdown', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Class Feature:"),
                            dcc.Dropdown(id='class-feature-dropdown', multi=False),
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Graph Type:"),
                            dcc.Dropdown(
                                id='graph-type-dropdown',
                                options=[
                                    {"label": "Scatter Plot", "value": "scatter"},
                                    {"label": "Bar Chart", "value": "bar"},
                                    {"label": "Pie Chart", "value": "pie"},
                                    {"label": "Histogram", "value": "histogram"},
                                ],
                                value="scatter",
                            ),
                        ],
                        width=6,
                    ),
                ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button("Generate Plot", id="generate-plot", n_clicks=0),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(id="output-graph"),
                        ],
                        width=12,
                    ),
                ],
            ),
        ],
        fluid=True,
    )


@app.callback(
    Output("input-feature-dropdown", "options"),
    Output("class-feature-dropdown", "options"),
    Input("stored-data2", "data"),
)
def update_dropdown_options(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    Output("output-graph", "figure"),
    Input("generate-plot", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown", "value"),
    State("class-feature-dropdown", "value"),
    State("graph-type-dropdown", "value"),
)
def update_graph(n_clicks, data, input_feature, class_feature, graph_type):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)
        
        if graph_type == "scatter":
            fig = plx.scatter(df, x=input_feature, color=class_feature, title=f'{input_feature} vs {class_feature}')
        elif graph_type == "bar":
            fig = plx.bar(df, x=input_feature, color=class_feature, title=f'{input_feature} vs {class_feature}')
        elif graph_type == "pie":
            fig = plx.pie(df, names=class_feature, title=f'Distribution of {class_feature}')
        elif graph_type == "histogram":
            
            fig = plx.histogram(df, x=input_feature, color=class_feature, title=f'Distribution of {input_feature}')
        return fig
    return {}