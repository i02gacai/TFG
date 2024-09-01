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

# Descriptions dictionary
descriptions = {
    "f1": "Maximum Fisher’s Discriminant Ratio: Measures how well the classes are separated relative to their internal variability. Higher values indicate that the classes are well-separated, making classification easier.",
    "f1v": "Directional-vector Maximum Fisher’s Discriminant Ratio: A variant of Fisher's Discriminant Ratio that considers the directions of the vectors. Lower values may indicate greater difficulty in class separation.",
    "f2": "Overlap of the Per-class Bounding Boxes: Measures the overlap between the bounding boxes of different classes. Higher values indicate more overlap, suggesting that classes are less well-separated and thus harder to classify correctly.",
    "f3": "Maximum Individual Feature Efficiency: Evaluates the efficiency of the best individual feature for separating classes. Values close to 1 indicate that at least one feature separates the classes very well.",
    "f4": "Collective Feature Efficiency: Similar to f3, but measures the combined efficiency of all features in separating classes. A value close to 1 indicates that the feature set is highly effective for class separation.",
    "l1": "Sum of the Error Distance by Linear Programming: Measures the difficulty of separating classes using a linear classifier via linear programming. Higher values indicate that finding a linear separation is more challenging.",
    "l2": "Error Rate of the Linear Classifier: The classification error rate of a simple linear classifier. Higher values indicate that the problem is more difficult from the perspective of a linear classifier.",
    "l3": "Non-linearity of the Linear Classifier: Measures the non-linearity of the problem concerning a linear classifier. Higher values indicate greater non-linearity and, therefore, more difficulty for linear classifiers.",
    "n1": "Fraction of Borderline Points: The proportion of instances that are near the decision boundary. A high value suggests that many instances are close to the boundary between classes, which can complicate classification.",
    "n2": "Ratio of Intra/Extra Class Nearest Neighbor Distance: The ratio of intra-class to extra-class nearest neighbor distances. A high value indicates that instances within the same class are closer to each other than to those of different classes, suggesting good class separation.",
    "n3": "Error Rate of the Nearest Neighbor Classifier: The error rate of a k-nearest neighbors (k-NN) classifier. A high value indicates that a k-NN classifier has difficulty distinguishing between classes.",
    "n4": "Non-linearity of the Nearest Neighbor Classifier: Measures the non-linearity of the problem with respect to a k-NN classifier. High values suggest that the problem does not fit well with proximity-based classification.",
    "t1": "Fraction of Hyperspheres Covering Data: The proportion of the data covered by the minimum hyperspheres necessary to cover all data instances. A high value may indicate that the data is well-concentrated.",
    "lsc": "Local Set Cardinality: Measures the average number of neighbors of the same class in the local environment of each instance. High values indicate a high local density of the same class, making classification easier.",
    "density": "Class Density: Measures the density of the class in the feature space. High values suggest that the instances within each class are densely packed.",
    "clsCoef": "Clustering Coefficient: The clustering coefficient measures the probability that two neighbors of an instance are also neighbors of each other. High values indicate a strong clustering structure, which can simplify classification.",
    "hubs": "Hubness: Measures the prevalence of hubs, which are data points that frequently appear as neighbors to other points. A high value can indicate that some points are central in the feature space, potentially affecting the problem's difficulty.",
    "t2": "Average Number of Discrete Domains: The average number of discrete domains into which the feature values are divided. Low values suggest that the attributes are more continuous.",
    "t3": "Normalized Entropy of the Discrete Domain: Measures the normalized entropy of the discrete feature domain. A low value indicates less uncertainty or variability in the discrete features.",
    "t4": "Mutual Information: Measures the amount of information shared between the features and the class labels. A low value indicates that the features are less informative with respect to the target class.",
    "c1": "Fraction of Uninformative Features: The proportion of features that do not provide relevant information for classification. A high value indicates that there are many redundant or irrelevant features.",
    "c2": "Feature Dependence: Measures the dependence between features. A high value suggests that the features are highly correlated with each other, which can affect the model's generalization ability."
}


def layout():
    return dbc.Container(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Input Feature:"),
                            dcc.Dropdown(id='input-feature-dropdown_', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Class Feature:"),
                            dcc.Dropdown(id='class-feature-dropdown_', multi=False),
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
                            html.Button("Calculate Measures", id="calculate-measures", n_clicks=0),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            html.Br(),
            html.Div(id='complexity-report'),
            html.Img(id="output-image", style={'height': '400px', 'width': '50%', 'margin-left': '25%'}, hidden=True),
        ],
        fluid=True,
    )


@app.callback(
    Output("input-feature-dropdown_", "options"),
    Output("class-feature-dropdown_", "options"),
    Input("stored-data2", "data"),
)
def update_dropdown_options(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    [Output("output-image", "src"),
     Output("output-image", "hidden"),
     Output("complexity-report", "children")],
    Input("calculate-measures", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown_", "value"),
    State("class-feature-dropdown_", "value"),
)
def calculate_complexity_measures(n_clicks, data, input_feature, class_feature):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)   
        # Initialize ComplexityCalculator with default parametrization
        cc = px.ComplexityCalculator()

        X , y = df[input_feature], df[class_feature]

        # Convert categorical columns to numerical representations
        for column in X.columns:
            if X[column].dtype == 'object':  # Check if column has string (object) values
                X[column] = pd.factorize(X[column])[0] 

        # Fit model with data
        cc.fit(X, y)
        
        cc._metrics()

        # Prepare figure
        fig = plt.figure(figsize=(7,7))

        # Generate plot describing the dataset
        cc.plot(fig, (1,1,1))

        # Save the figure in a buffer in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Get the report dictionary
        report = cc.report()


        # Convert the report dictionary into Dash HTML components with tooltips
        report_table = []
        report_table.append(html.Tr([html.Th("Metric"), html.Th("Value")]))
        report_table.append(html.Tr([html.Td("Number of Samples"), html.Td(report['n_samples'])]))
        report_table.append(html.Tr([html.Td("Number of Features"), html.Td(report['n_features'])]))
        report_table.append(html.Tr([html.Td("Score"), html.Td(report['score'])]))
        report_table.append(html.Tr([html.Td("Number of Classes"), html.Td(report['n_classes'])]))
        report_table.append(html.Tr([html.Td("Classes"), html.Td(report['classes'])]))
        report_table.append(html.Tr([html.Td("Prior Probability"), html.Td(str(report['prior_probability']))]))
        report_table.append(html.Tr([html.Th("Complexity Metrics", colSpan=2)]))

        for index, (key, value) in enumerate(report['complexities'].items()):
            metric_id = f"feature-{index}"  # Unique ID for each metric
            report_table.append(html.Tr([
                html.Td(key, id=metric_id),  # Assign unique ID to the metric cell
                html.Td(value)
            ]))
            # Add Tooltip
            report_table.append(
                dbc.Tooltip(
                    descriptions.get(key, "No description available."),
                    target=metric_id,
                    placement="top"
                )
            )

        report_table_html = html.Table(report_table)

        # Return the image, the hidden state, and the table
        return f'data:image/png;base64,{image_base64}', False, report_table_html
    
    return '', True, ''  # Return empty values by default
    