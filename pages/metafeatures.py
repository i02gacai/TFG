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
from pymfe.mfe import MFE

# Descripciones de las meta-características (se pueden extender según se desee)
descriptions = {
    "attr_conc.mean" : "Concentration of the attributes, which measures how many distinct values the attributes in the dataset have. Mean is the average, and Sd is the standard deviation of this concentration.",
    "attr_conc.sd" : "Concentration of the attributes, which measures how many distinct values the attributes in the dataset have. Mean is the average, and Sd is the standard deviation of this concentration.",
    "attr_ent.mean" : "Entropy of the attributes. Entropy measures the amount of uncertainty or randomness in the attribute values. Mean is the average.",
    "attr_ent.sd" : "Entropy of the attributes. Entropy measures the amount of uncertainty or randomness in the attribute values. Sd is the standard deviation.",
    "attr_to_inst" : "Ratio between the number of attributes and the number of instances in the dataset.",
    "can_cor.mean" : "Canonical correlation. It measures the linear relationship between two sets of variables (e.g., attributes and classes). Mean is the average, and sd is the standard deviation of the canonical correlation.",
    "can_cor.sd" : "Canonical correlation. It measures the linear relationship between two sets of variables (e.g., attributes and classes). Mean is the average, and sd is the standard deviation of the canonical correlation.",
    "cat_to_num" : "Ratio between the number of categorical attributes and the number of numerical attributes.",
    "class_conc.mean" : "Class concentration, similar to attr_conc, but applied to the classes. Mean is the average, and sd is the standard deviation of this concentration.",
    "class_conc.sd" : "Class concentration, similar to attr_conc, but applied to the classes. Mean is the average, and sd is the standard deviation of this concentration.",
    "class_ent" : "Class entropy, measuring the uncertainty in the class distribution.",
    "cor.mean" : "Attribute correlation, measuring the strength and direction of the linear relationship between pairs of attributes. Mean is the average, and sd is the standard deviation of the correlation.",
    "cor.sd" : "Attribute correlation, measuring the strength and direction of the linear relationship between pairs of attributes. Mean is the average, and sd is the standard deviation of the correlation.",
    "cov.mean" : "Covariance between attributes, measuring the joint variability of two random variables. Mean is the average, and sd is the standard deviation of the covariance.",
    "cov.sd" : "Covariance between attributes, measuring the joint variability of two random variables. Mean is the average, and sd is the standard deviation of the covariance.",
    "eigenvalues.mean" : "Eigenvalues of the covariance matrix, reflecting the amount of variance explained by each principal component. Mean is the average, and sd is the standard deviation of these eigenvalues.", 
    "eigenvalues.sd" : "Eigenvalues of the covariance matrix, reflecting the amount of variance explained by each principal component. Mean is the average, and sd is the standard deviation of these eigenvalues.",
    "eq_num_attr" : "Number of attributes that have the same number of distinct values.",
    "freq_class.mean" : "Class frequency, mean is the average, and sd is the standard deviation of the class frequencies in the dataset.",
    "freq_class.sd" : "Class frequency, mean is the average, and sd is the standard deviation of the class frequencies in the dataset.",
    "g_mean.mean" : "Geometric mean of the attribute values. Mean is the average, and sd is the standard deviation of the geometric mean.",
    "g_mean.sd" : "Geometric mean of the attribute values. Mean is the average, and sd is the standard deviation of the geometric mean.",
    "gravity" : "Centroid or center of gravity of the data, a point in space that represents the center of mass of all data points.",
    "h_mean.mean" : "Harmonic mean of the attribute values. Mean is the average, and sd is the standard deviation of the harmonic mean.",
    "h_mean.sd" : "Harmonic mean of the attribute values. Mean is the average, and sd is the standard deviation of the harmonic mean.",
    "inst_to_attr" : "Ratio between the number of instances and the number of attributes in the dataset.",
    "iq_range.mean" : "Interquartile range of the attribute values. Mean is the average, and sd is the standard deviation of the interquartile range.",
    "iq_range.sd" : "Interquartile range of the attribute values. Mean is the average, and sd is the standard deviation of the interquartile range.",
    "joint_ent.mean" : "Joint entropy of pairs of attributes, measuring the combined uncertainty. Mean is the average, and sd is the standard deviation of the joint entropy.",
    "joint_ent.sd" : "Joint entropy of pairs of attributes, measuring the combined uncertainty. Mean is the average, and sd is the standard deviation of the joint entropy.",
    "kurtosis.mean" : "Kurtosis of the attribute values, measuring the sharpness or flatness of the distribution. Mean is the average, and sd is the standard deviation of the kurtosis.",
    "kurtosis.sd" : "Kurtosis of the attribute values, measuring the sharpness or flatness of the distribution. Mean is the average, and sd is the standard deviation of the kurtosis.",
    "lh_trace" : "Trace of the variance ratio matrix (ratio of within-class variance to total variance). It measures the separability of classes in terms of variance.",
    "mad.mean" : "Mean absolute deviation of the attribute values. Mean is the average, and sd is the standard deviation of the mean absolute deviation.",
    "mad.sd" : "Mean absolute deviation of the attribute values. Mean is the average, and sd is the standard deviation of the mean absolute deviation.",
    "max.mean" : "Maximum value of the attributes. Mean is the average, and sd is the standard deviation of the maximum values.",
    "max.sd" : "Maximum value of the attributes. Mean is the average, and sd is the standard deviation of the maximum values.",
    "mean.mean" : "Mean of the attribute values. Mean is the average, and sd is the standard deviation of the mean of the values.",
    "mean.sd" : "Mean of the attribute values. Mean is the average, and sd is the standard deviation of the mean of the values.",
    "median.mean" : "Median of the attribute values. Mean is the average, and sd is the standard deviation of the medians.",
    "median.sd" : "Median of the attribute values. Mean is the average, and sd is the standard deviation of the medians.",
    "min.mean" : "Minimum value of the attributes. Mean is the average, and sd is the standard deviation of the minimum values.",
    "min.sd" : "Minimum value of the attributes. Mean is the average, and sd is the standard deviation of the minimum values.",
    "mut_inf.mean" : "Mutual information between pairs of attributes, measuring the dependency between them. Mean is the average, and sd is the standard deviation of the mutual information.",
    "mut_inf.sd" : "Mutual information between pairs of attributes, measuring the dependency between them. Mean is the average, and sd is the standard deviation of the mutual information.",
    "nr_attr": "Number of attributes in the dataset.",
    "nr_bin" : "Number of binary attributes (which have only two possible values).",
    "nr_cat" : "Number of categorical attributes in the dataset.",
    "nr_class" : "Number of classes in the dataset.",
    "nr_cor_attr" : "Number of correlated attributes, usually those with a correlation above a threshold.",
    "nr_disc" : "Number of discrete attributes, those that have a limited number of values.",
    "nr_inst" : "Number of instances in the dataset.",
    "nr_norm" : "Number of normalized (scaled) attributes.",
    "nr_num" : "Number of numerical attributes in the dataset.",
    "nr_outliers" : "Number of outliers in the dataset.",
    "ns_ratio" : "Ratio between the number of instances of the most common class and the least common class.",
    "num_to_cat" : "Ratio between the number of numerical and categorical attributes.",
    "p_trace" : "Trace of the correlation matrix. It reflects the sum of the variances explained by the principal components.",
    "range.mean" : "Range of the attribute values. Mean is the average, and sd is the standard deviation of the ranges.",
    "range.sd" : "Range of the attribute values. Mean is the average, and sd is the standard deviation of the ranges.",
    "roy_root" : "Roy's root, a statistic related to the maximum canonical correlation.",
    "sd.mean" : "Standard deviation of the attribute values. Mean is the average, and sd is the standard deviation of the standard deviation.",
    "sd.sd" : "Standard deviation of the attribute values. Mean is the average, and sd is the standard deviation of the standard deviation.",
    "sd_ratio" : "Ratio between the standard deviation within classes and the total standard deviation.",
    "skewness.mean" : "Skewness of the attribute values, measuring the degree of asymmetry in the distribution.",
    "skewness.sd" : "Skewness of the attribute values, measuring the degree of asymmetry in the distribution.",
    "sparsity.mean" : "Sparsity of the attributes, which measures the proportion of null or zero values in the data. Mean is the average, and sd is the standard deviation of the sparsity.",
    "sparsity.sd" : "Sparsity of the attributes, which measures the proportion of null or zero values in the data. Mean is the average, and sd is the standard deviation of the sparsity.",
    "t_mean.mean" : "Truncated mean of the attribute values. The truncated mean is calculated by excluding a percentage of the most extreme values (e.g., the highest and lowest values). Mean is the average, and sd is the standard deviation of the truncated mean.",
    "t_mean.sd" : "Truncated mean of the attribute values. The truncated mean is calculated by excluding a percentage of the most extreme values (e.g., the highest and lowest values). Mean is the average, and sd is the standard deviation of the truncated mean.",
    "var.mean" : "Variance of the attribute values. Variance measures the dispersion of the data around the mean. Mean is the average, and sd is the standard deviation of the variance.",
    "var.sd" : "Variance of the attribute values. Variance measures the dispersion of the data around the mean. Mean is the average, and sd is the standard deviation of the variance.",
    "w_lambda" : "Wilks' Lambda statistic, used in multivariate analysis to measure the separation between groups. Lower values indicate better separation between classes."
    }

# Layout for the dashboard page
def layout():
    return dbc.Container(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Input Feature:"),
                            dcc.Dropdown(id='input-feature-dropdown1', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Class Feature:"),
                            dcc.Dropdown(id='class-feature-dropdown1', multi=False),
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
                            html.Button("Calculate Meta-Features", id="calculate_meta_features", n_clicks=0),
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
                            html.Div(id="output-meta-features"),
                        ],
                        width=12,
                    ),
                ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button("Download CSV", id="download-csv", n_clicks=0),
                        ],
                        width=6,
                        className="text-center",
                    ),
                    dbc.Col(
                        [
                            html.Button("Download PDF", id="download-pdf", n_clicks=0),
                        ],
                        width=6,
                        className="text-center",
                    ),
                ],
            ),
            dcc.Download(id="download-component"),
            html.Div(id="output-pdf"),
        ]
    )

@app.callback(
    Output('download-component', 'data'),
    Input("download-csv", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown1", "value"),
    State("class-feature-dropdown1", "value"),
)
def update_download_link(n_clicks, data, input_feature, class_feature):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)
        
        # Ensure the input features are in the correct format
        X = df[input_feature].values
        y = df[class_feature].values

        # Reshape if X is a single feature
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Extraer las meta-características
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()
        
        # Crear un DataFrame con los resultados
        results_df = pd.DataFrame({'Feature': ft_names, 'Value': ft_values})
        
        # Crear el enlace de descarga
        return dcc.send_data_frame(results_df.to_csv, "metafeatures.csv")

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.print();
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('output-pdf', 'children'),
    Input('download-pdf', 'n_clicks')
)

@app.callback(
    Output("input-feature-dropdown1", "options"),
    Output("class-feature-dropdown1", "options"),
    Input("stored-data2", "data"),
)
def update_dropdown_options_(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    Output('output-meta-features', 'children'),
    Input("calculate_meta_features", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown1", "value"),
    State("class-feature-dropdown1", "value"),
)
def calculate_meta_features(n_clicks, data, input_feature, class_feature):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)
        
        # Ensure the input features are in the correct format
        X = df[input_feature].values
        y = df[class_feature].values

        # Reshape if X is a single feature
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Extraer las meta-características
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()
        
        # Crear un DataFrame con los resultados
        results_df = pd.DataFrame({'Feature': ft_names, 'Value': ft_values})
        
        # Crear filas de la tabla con tooltips
        table_rows = []
        for index, row in results_df.iterrows():
            feature_name = row['Feature']
            feature_value = row['Value']
            description = descriptions.get(feature_name, "Descripción no disponible.")
            
            table_rows.append(
                html.Tr([
                    html.Td(feature_name, id=f"feature-{index}"),
                    html.Td(feature_value)
                ])
            )
            table_rows.append(
                dbc.Tooltip(description, target=f"feature-{index}")
            )
        
        # Mostrar los resultados
        return html.Div([
            html.H5("Meta-Características Calculadas:"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Feature"), html.Th("Value")])),
                html.Tbody(table_rows)
            ], striped=True, bordered=True, hover=True)
        ])
    return html.Div()
