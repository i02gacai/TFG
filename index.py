import base64
import io
from dash import dcc, dash_table, html
import dash_bootstrap_components as dbc
import pandas as pd
import os
from dash.dependencies import Input, Output, State
from scipy.io import arff
from app import app
import pages

# The way to run the app in a local server.
server = app.server

app.layout = dbc.Container(
    [
        html.Div(
            [
                # A component that allows to get the current URL of the page.
                dcc.Location(id="url", refresh=False),
                # Creating the navbar of the app.
                dbc.Navbar(
                    children=[
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        # Creating an image with the logo of the app.
                                        html.Img(
                                            src=app.get_asset_url("logo.png"), height="80px"
                                        )
                                    ),
                                ],
                                className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                                align="center",
                            ),
                            href=app.get_relative_path("/"),
                        ),
                        # Creating the items of the navbar.
                        dbc.Row(
                            [
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            # The pages of the app.
                                            dbc.NavItem(dbc.NavLink("Home", href=app.get_relative_path("/"))),
                                            dbc.NavItem(dbc.NavLink("Dashboard", href=app.get_relative_path("/dash"))),
                                            dbc.NavItem(dbc.NavLink("Meta-Feature", href=app.get_relative_path("/metafeatures"))),
                                            dbc.NavItem(dbc.NavLink("Complexity", href=app.get_relative_path("/complexity"))),
                                        ],
                                        className="w-100",
                                        fill=True,
                                        horizontal='end'
                                    ),
                                    navbar=True,
                                    is_open=True,
                                ),
                            ],
                            className="flex-grow-1",
                        ),
                    ],
                ),
                # Components that allow to store data in the whole the app.
                dcc.Store(id='stored-data', data=None, storage_type='session'),
                dcc.Store(id='stored-data2', data=None, storage_type='session'),
                # BODY
                # A component that allows to upload a file.
                html.Div(
                    id='upload-div',
                    children=[
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            multiple=False,
                        ),
                    ]
                ),
                
                # A component that shows a message to the user about the data file.
                dbc.Alert(["The data to perform the functionalities has not been added, load a file to continue"],
                          id="alert-auto",
                          is_open=False,
                          dismissable=True,
                          fade=True,
                          ),

                html.Div( id = "card",
                    children=[
                        dbc.Card(
                            children=[
                                # A card with information about the web aplication.
                                dbc.CardHeader("Welcome!"),
                                dbc.CardBody(
                                    [
                                        dcc.Markdown(
                                            """
                                            This application is designed to make data analysis and visualization both efficient and accurate. With our tools, you can easily create customized charts, calculate metafeatures, and evaluate the complexity of your data in a simple and intuitive way. Our goal is to provide you with the necessary tools to deepen your analysis, enabling you to select the most appropriate evaluation techniques and algorithms for your needs.
                                            """,
                                            style={"margin": "0 10px"},
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],                
                ),


                html.Div(id="page-content"),
                html.Div(id="table-div"),  
                
            ]
        ),

    ]
)


@app.callback(Output('stored-data2', 'data'), [
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename')]
)
def update_data(contents, filename):
    """
    It takes the contents of the uploaded file and converts it to a dataframe.

    The dataframe is then converted to a dictionary to be returned.

    The dictionary is then used to update the data in the table.

    :param contents: the contents of the uploaded file
    :param filename: The name of the uploaded file
    :return: A list of dictionaries.
    """
    if contents:
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith(".csv"):
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0)
            elif filename.endswith(".xls") or filename.endswith(".xlsx"):
                # Assume that the user uploaded an Excel file
                df = pd.read_excel(io.BytesIO(decoded), header=0)
            elif filename.endswith(".arff"):
                # Assume that the user uploaded an ARFF file
                data, meta = arff.loadarff(io.StringIO(decoded.decode("utf-8")))
                df = pd.DataFrame(data)
            else:
                return None
        except Exception as e:
            print(e)
            return None

        # Ensure the DataFrame is JSON serializable
        data_records = df.to_dict('records')
        for record in data_records:
            for key, value in record.items():
                if isinstance(value, bytes):
                    record[key] = value.decode('utf-8')

        return data_records
    return None


@app.callback(Output('table-div', 'children'), Output('card', 'hidden'), [Input('stored-data2', 'data')])
def display_table(data):
    """
    Display the data in a table format.

    :param data: The data from the uploaded file
    :return: A DataTable component
    """
    if data:
        df = pd.DataFrame(data)
        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
        ), True
    return None, False


@app.callback(Output("page-content", "children"), Output('upload-div', 'hidden'), Output("alert-auto", "is_open"), Output('table-div', 'hidden'),
              [Input("url", "pathname"), Input('stored-data2', 'data')],
              prevent_initial_call=True
)
def display_page_content(pathname, data):
    """
    If the path is empty, return the home page. If the path is "dash", return the dashboard page. If the
    path is "train", return the models page. If the path is "predict", return the predict page.
    Otherwise, return a 404 page

    :param pathname: The pathname argument is the current location of the page
    :param data: The dataframe that is uploaded by the user
    :return: a list of dash components.
    """
    path = app.strip_relative_path(pathname)
    if not path:
        if data != None:
            return pages.home.layout(), False, False, False
        else:
            return pages.home.layout(), False, True, False
    elif path == "dash":
        if data != None:
            return pages.dashboard.layout(), True, False, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), "  No data loaded"]),
                    dbc.ModalFooter(dbc.Button([dcc.Link('Go back to home', href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False
    elif path == "metafeatures":
        if data != None:
            return pages.metafeatures.layout(), True, False, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), "  No data loaded"]),
                    dbc.ModalFooter(dbc.Button([dcc.Link('Go back to home', href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False
    elif path == "complexity":
        # Checking if exists a model in the server. If it does, it returns the predict page.
        if data != None:
            return pages.complexity.layout(), True, False, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), "  No data loaded"]),
                    dbc.ModalFooter(dbc.Button([dcc.Link('Go back to home', href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False
    else:
        return "404"


# A way to run the app in a local server.
if __name__ == "__main__":
    app.run_server(debug=True)


