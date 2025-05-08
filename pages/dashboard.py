import base64
import io
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as plx
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
from app import app

def t(key, lang):
    return lang.get(key, key)

def layout(lang):
    return dbc.Container(
        [   
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(t("Select-Input-Feature", lang)),
                            dcc.Dropdown(id='input-feature-dropdown', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(t("Select-Class-Feature", lang)),
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
                            html.Label(t("Select-Graph-Type", lang)),
                            dcc.Dropdown(
                                id='graph-type-dropdown',
                                options=[
                                    {"label": t("Scatter-Plot", lang), "value": "scatter"},
                                    {"label": t("Pie-Chart", lang), "value": "pie"},
                                    {"label": t("Histogram", lang), "value": "histogram"},
                                    {"label": t("Box-Plot", lang), "value": "box"},
                                    {"label": t("Heatmap", lang), "value": "heatmap"},
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
                            html.Button(t("Generate-Plot", lang), id="generate-plot", n_clicks=0),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            html.Br(),
            html.Div(id = "error-message-plot"),
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