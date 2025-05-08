import base64
import io
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import request, g
from scipy.io import arff
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pymfe.mfe import MFE
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as plx
import problexity as px
import pages
from app import app
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pymfe.mfe import MFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import dash_bootstrap_components as dbc
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 


# Configuración inicial de la aplicación y el servidor
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"])
server = app.server

initial_state = False

def load_language(lang_code):
    """
    Carga el archivo JSON correspondiente al idioma especificado.

    Parámetros:
    lang_code (str): El código del idioma (por ejemplo, 'es' para español, 'en' para inglés).

    Retorna:
    dict: El contenido del archivo JSON como un diccionario.

    Si el archivo con el idioma solicitado no se encuentra, se carga el archivo 'en.json' (inglés) por defecto.
    """
    try:
        with open(f'locales/{lang_code}.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # En caso de que el archivo de idioma no exista, cargamos inglés por defecto
        with open('locales/en.json', 'r', encoding='utf-8') as f:
            return json.load(f)

def t(key, lang):
    """
    Traduce una clave de texto a su valor correspondiente en el idioma proporcionado.

    Si la clave no se encuentra en el diccionario del idioma, se devuelve la clave misma.

    Args:
        key (str): La clave del texto a traducir.
        lang (dict): Un diccionario con las traducciones, donde las claves son las claves de texto
                     y los valores son las traducciones correspondientes.

    Returns:
        str: La traducción correspondiente a la clave en el idioma proporcionado, o la clave misma 
             si no se encuentra en el diccionario.
    """
    return lang.get(key, key)  # Si no encuentra la clave, devuelve la clave misma

def get_dataset_files():
    """
    Obtiene una lista de archivos en la carpeta 'dataset'.

    Esta función verifica si la carpeta 'dataset' existe en el directorio actual. Si existe, devuelve una lista
    de los archivos que contiene. Si no, devuelve una lista vacía.

    Returns:
        list: Una lista de los nombres de archivos en la carpeta 'dataset', o una lista vacía si la carpeta no existe.
    """
    dataset_folder = 'dataset'
    if os.path.exists(dataset_folder):  # Verifica si la carpeta 'dataset' existe
        return [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]
    return []  # Si no existe la carpeta, retorna una lista vacía


# Función para obtener el idioma por defecto
@app.server.before_request
def set_language():
    """
    Establece el idioma del usuario antes de procesar la solicitud.

    Esta función se ejecuta antes de cada solicitud HTTP para determinar el idioma preferido 
    por el usuario, basado en el encabezado 'Accept-Language'. Si no se encuentra, se asigna 'en' (inglés) como idioma por defecto.
    El idioma se almacena en `g.lang_code` y se carga mediante la función `load_language`.

    El código de idioma se extrae de los primeros dos caracteres de la cadena (por ejemplo, 'es' para español).
    """
    user_language = request.headers.get('Accept-Language', 'en').split(',')[0]  # Obtiene el idioma preferido del encabezado
    g.lang_code = user_language[:2]  # Obtiene el código del idioma, por ejemplo, 'es' para español
    g.lang = load_language(g.lang_code)  # Carga el archivo de idioma correspondiente


#Layout inicial con los componentes necesarios
app.layout = dbc.Container(
    [   
        # Un componente que permite obtener la URL actual de la página.
        dcc.Location(id="url", refresh=False),
        # Componentes que permiten almacenar datos en toda la app.
        dcc.Store(id="language", data=None, storage_type='session'),
        dcc.Store(id='stored-data', data=None, storage_type='session'),
        dcc.Store(id='stored-data2', data=None, storage_type='session'),

        dcc.Download(id="download-component-index"),

        dcc.Input(id='trigger', value='', type='text', style={'display': 'none'}),

        html.Div(id='output')
    ]
)

@app.callback(
    Output('output', 'children', allow_duplicate=True),
    Output('language', 'data', allow_duplicate=True),
    Input('trigger', 'value'),
    prevent_initial_call='initial_duplicate'
)
def initial_update_output(trigger):
    """
    Actualiza el contenido de la página inicial y establece el idioma por defecto.

    Esta función es llamada cuando se activa un cambio en el valor de 'trigger'. 
    Inicializa el estado de la página, creando una barra de navegación, un dropdown para seleccionar un dataset, 
    un área para cargar archivos y un mensaje de bienvenida.

    Args:
        trigger (str): Valor de activación que dispara la actualización del contenido.

    Returns:
        dash.no_update, dash.no_update: Retorna valores sin cambios si el estado ya está inicializado.
    """
    global initial_state
    if not initial_state:
        initial_state = True
        return html.Div(
            [   
                # Creando la barra de navegación de la app.
                dbc.Navbar(
                    children=[
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        # Creando una imagen con el logo de la app.
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
                        # Creando los elementos de la barra de navegación.
                        dbc.Row(
                            [
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            # Las páginas de la app.
                                            dbc.NavItem(dbc.NavLink(t("Home",g.lang), href=app.get_relative_path("/"))),
                                            dbc.NavItem(dbc.NavLink(t("Dashboard",g.lang), href=app.get_relative_path("/dash"))),
                                            dbc.NavItem(dbc.NavLink(t("Meta-Feature",g.lang), href=app.get_relative_path("/metafeatures"))),
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
                        dbc.Row(
                                [
                                    html.I(className="fas fa-language fa-fw mr-1"),
                                    dcc.Dropdown(
                                        id='language-dropdown',
                                        options=[
                                            {'label': 'English (en)', 'value': 'en'},
                                            {'label': 'Spanish (es)', 'value': 'es'},
                                            # Agregar más idiomas aquí si es necesario
                                        ],
                                        value=g.lang_code,  # Valor inicial basado en el idioma actual
                                        clearable=False,
                                        style = { 'width': '150px', 
                                            'margin-left': '5px',}
                                    ),
                                ],
                                id = 'language-row',
                                className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                                align="center",
                                style = {   'width': '200px',        # Ancho del dropdown
                                                    'height': '50px',
                                                    'backgroundColor': '#f9f9f9',
                                                    'margin-top': '20px',
                                                    'margin-left': '5px',}
                            ),
                    ],
                ),
                # Cuerpo de la página
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                id='dropdown-div',
                                children = [ 
                                    dcc.Dropdown(
                                    id='dataset-dropdown',
                                    options=[{'label': f, 'value': f} for f in get_dataset_files()],
                                    placeholder=t("Select-dataset",g.lang),
                                ),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                id='upload-div',
                                children=[
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div([t("Drag-drop",g.lang)]),
                                        style={
                                            "width": "100%",
                                            "height": "60px",
                                            "lineHeight": "60px",
                                            "border" : "#007bff",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "5px",
                                            "textAlign": "center",
                                            "margin": "10px",
                                        },
                                        multiple=False,
                                    ),
                                    html.Br(),
                                ]
                            ),
                            width=6,
                        )
                    ],
                    align="center",  # Alineación vertical de los componentes en el centro
                    className="mb-4",  # Agrega margen inferior para espaciado
                ),
                
                # Componente que muestra un mensaje al usuario sobre el archivo de datos.
                dbc.Alert([t("Msg-load-file",g.lang)],
                          id="alert-auto",
                          is_open=False,
                          dismissable=True,
                          fade=True,
                          ),

                html.Div( id = "card",
                    children=[
                        dbc.Card(
                            children=[
                                # Una tarjeta con información sobre la aplicación web.
                                dbc.CardHeader(t("Welcome", g.lang)),
                                dbc.CardBody(
                                    [
                                        dcc.Markdown(
                                            t("welcome-card",g.lang),
                                            style={"margin": "0 10px"},
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Br(),
                    ],                
                ),

                html.Div(id="page-content"),
                html.Div(id="error-message"),
                html.Div(id="table-div"),  
                
                dbc.Row(
                    [
                        dbc.Col(
                            [   
                                html.Br(),
                                html.Button(t("Download-CSV", g.lang), id="download-csv-index", n_clicks=0, hidden = True),
                            ],
                            width=12,
                            className="text-center",
                        ),
                    ],
                ),
                ]
        ), g.lang
    return dash.no_update, dash.no_update

@app.callback(
    Output('output', 'children', allow_duplicate=True),
    Output('language', 'data', allow_duplicate=True),
    Input('language-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(value):
    """
    Actualiza el contenido de la página y el idioma cuando se cambia la selección de idioma.

    Esta función es llamada cuando se selecciona un nuevo idioma en el dropdown. Actualiza la interfaz 
    de usuario para reflejar las traducciones correspondientes al idioma seleccionado, y también ajusta 
    la barra de navegación y otros elementos de la página en el idioma elegido.

    Args:
        value (str): El valor seleccionado del dropdown de idiomas.

    Returns:
        html.Div, str: La interfaz actualizada con el nuevo idioma y los datos de idioma correspondientes.
    """
    lang = load_language(value)  # Cargar el archivo de idioma basado en el valor seleccionado
    return html.Div(
        [   
            # Creando la barra de navegación de la app.
            dbc.Navbar(
                children=[
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(
                                    # Creando una imagen con el logo de la app.
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
                    # Creando los elementos de la barra de navegación.
                    dbc.Row(
                        [
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        # Las páginas de la app.
                                        dbc.NavItem(dbc.NavLink(t("Home", lang), href=app.get_relative_path("/"))),
                                        dbc.NavItem(dbc.NavLink(t("Dashboard", lang), href=app.get_relative_path("/dash"))),
                                        dbc.NavItem(dbc.NavLink(t("Meta-Feature", lang), href=app.get_relative_path("/metafeatures"))),
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
                    dbc.Row(
                            [
                                html.I(className="fas fa-language fa-fw mr-1"),
                                dcc.Dropdown(
                                    id='language-dropdown',
                                    options=[
                                        {'label': 'English (en)', 'value': 'en'},
                                        {'label': 'Spanish (es)', 'value': 'es'},
                                        # Agregar más idiomas aquí si es necesario
                                    ],
                                    value=value,  # Valor inicial basado en el idioma actual
                                    clearable=False,
                                    style = { 'width': '150px', 
                                        'margin-left': '5px',}
                                ),
                            ],
                            id='language-row',
                            className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                            align="center",
                            style = {   'width': '200px',        # Ancho del dropdown
                                                'height': '50px',
                                                'backgroundColor': '#f9f9f9',
                                                'margin-top': '20px',
                                                'margin-left': '5px',}
                        ),
                ],
            ),
            # Cuerpo de la página
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id='dropdown-div',
                            children = [ 
                                dcc.Dropdown(
                                id='dataset-dropdown',
                                options=[{'label': f, 'value': f} for f in get_dataset_files()],
                                placeholder=t("Select-dataset", lang),
                            ),
                            ]
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            id='upload-div',
                            children=[
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div([t("Drag-drop", lang)]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "border" : "#007bff",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Br(),
                            ]
                        ),
                        width=6,
                    )
                ],
                align="center",  # Alineación vertical de los componentes en el centro
                className="mb-4",  # Agrega margen inferior para espaciado
            ),
            # Componente que muestra un mensaje al usuario sobre el archivo de datos.
            dbc.Alert([t("Msg-load-file", g.lang)],
                        id="alert-auto",
                        is_open=False,
                        dismissable=True,
                        fade=True,
                        ),

            html.Div( id="card",
                children=[
                    dbc.Card(
                        children=[
                            # Una tarjeta con información sobre la aplicación web.
                            dbc.CardHeader(t("Welcome", lang)),
                            dbc.CardBody(
                                [
                                    dcc.Markdown(
                                        t("welcome-card", lang),
                                        style={"margin": "0 10px"},
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Br(),
                ],                
            ),

            html.Div(id="page-content"),
            html.Div(id="error-message"),
            html.Div(id="table-div"),  
            
            dbc.Row(
                [
                    dbc.Col(
                        [   
                            html.Br(),
                            html.Button(t("Download-CSV", lang), id="download-csv-index", n_clicks=0, hidden=True),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
        ]
    ), lang



# Callback updated to handle file selection from dropdown or file upload.
@app.callback(
    [Output('stored-data2', 'data'), Output('error-message', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('dataset-dropdown', 'value'),
     Input('language', 'data')]
)
def update_data(contents, filename, dataset_value, lang):
    """
    Toma el contenido del archivo cargado o el archivo seleccionado desde la carpeta 'datasets' y lo convierte en un dataframe.
    El dataframe luego se convierte en un diccionario que es retornado.
    """
    if contents:
        # Cuando el usuario sube un archivo
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith(".csv"):
                # Supone que el archivo cargado es CSV
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0)
            elif filename.endswith(".arff"):
                # Supone que el archivo cargado es ARFF
                data, meta = arff.loadarff(io.StringIO(decoded.decode("utf-8")))
                df = pd.DataFrame(data)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(decoded),header=0)
            else:
                return None, dbc.Alert(t("Unsoported-file", lang), color="danger")

        except Exception as e:
            print(e)
            return None, dbc.Alert(f"Error: {str(e)}", color="danger")

    elif dataset_value:
        # Cuando el usuario selecciona un archivo desde el dropdown
        dataset_path = os.path.join('dataset', dataset_value)
        try:
            if dataset_value.endswith(".csv"):
                # Supone que el archivo seleccionado es CSV
                df = pd.read_csv(dataset_path, header=0)
            elif dataset_value.endswith(".arff"):
                # Supone que el archivo seleccionado es ARFF
                data, meta = arff.loadarff(dataset_path)
                df = pd.DataFrame(data)
            elif dataset_value.endswith(".xlsx") or dataset_value.endswith(".xls"):
                df = pd.read_excel(dataset_path, header=0)
            else:
                return None, dbc.Alert(t("Unsoported-file", lang), color="danger")

        except Exception as e:
            print(e)
            return None, dbc.Alert(f"Error: {str(e)}", color="danger")
    else:
        return None, None

    # Asegurarse de que el DataFrame sea serializable en JSON
    data_records = df.to_dict('records')
    for record in data_records:
        for key, value in record.items():
            if isinstance(value, bytes):
                record[key] = value.decode('utf-8')

    return data_records, None



@app.callback(
    Output('table-div', 'children'),
    Output('card', 'hidden'),
    [Input('stored-data2', 'data')]
)
def display_table(data):
    """
    Muestra los datos en formato de tabla.

    :param data: Los datos del archivo cargado
    :return: Un componente DataTable
    """
    if data:
        df = pd.DataFrame(data)
        return dash_table.DataTable(
            id='editable-table',
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            editable=True,  # Hace la tabla editable
            row_deletable=True,
        ), True
    return None, False


@app.callback(
    [Output('stored-data', 'data'),
     Output("download-csv-index", "hidden", allow_duplicate= True)],
    Input('editable-table', 'data'),
    State('editable-table', 'data_previous'),
    prevent_initial_call='initial_duplicate'
)
def update_edited_data(rows, previous_rows):
    """
    Captura los datos editados de la tabla.

    :param rows: Datos actuales de la tabla
    :param previous_rows: Datos anteriores de la tabla
    :return: Datos editados para almacenar
    """
    if previous_rows is None:
        raise dash.exceptions.PreventUpdate
    return rows, False

@app.callback(
    Output('stored-data2', 'data', allow_duplicate=True),
    Input('stored-data', 'data'),
    prevent_initial_call='initial_duplicate'
)
def store_edited_data(edited_data):
    """
    Almacena los datos editados.

    :param edited_data: Datos editados de la tabla
    :return: Datos para almacenar
    """
    if edited_data is None:
        raise dash.exceptions.PreventUpdate
    return edited_data

# Callback para descargar los datos editados en un CSV
@app.callback(
    Output("download-component-index", "data"),
    Input("download-csv-index", "n_clicks"),
    Input('upload-data', 'filename'),
    Input('dataset-dropdown', 'value'),
    State("stored-data2", "data"),
)
def download_csv(n_clicks, filename, filename1, rows):
    if rows is None:
        return None
    if n_clicks > 0:
        # Convertir los datos editados en un DataFrame
        df_updated = pd.DataFrame(rows)
        
        # Convertir el DataFrame a CSV
        csv_string = df_updated.to_csv(index=False, encoding='utf-8')
        
        if filename is None:
            # Retornar un objeto que permita descargar el archivo CSV
            name = os.path.splitext(filename1)[0] + '.csv'
            return dcc.send_string(csv_string, name)
        else:
            name = os.path.splitext(filename)[0] + '.csv'
            return dcc.send_string(csv_string, name)


@app.callback(Output("page-content", "children"), Output('upload-div', 'hidden'), Output("alert-auto", "is_open"), Output('table-div', 'hidden'), Output('dropdown-div', 'hidden'), Output('download-csv-index', 'hidden', allow_duplicate= True), Output('language-row','hidden'),
              [Input("url", "pathname"), Input('stored-data2', 'data'), Input('language', 'data')],
              prevent_initial_call=True
)
def display_page_content(pathname, data, lang):
    """
    Si el path está vacío, retorna la página de inicio.
    Si el path es "dash", retorna la página del dashboard.
    Si el path es "train", retorna la página de modelos.
    Si el path es "predict", retorna la página de predicción.
    De lo contrario, retorna una página 404.

    :param pathname: El argumento pathname es la ubicación actual de la página
    :param data: El dataframe que es subido por el usuario
    :return: Una lista de componentes de Dash.
    """
    path = app.strip_relative_path(pathname)
    if not path:
        if data != None:
            return pages.home.layout(), False, False, False, False, False, False
        else:
            return pages.home.layout(), False, True, False, False, True, False
    elif path == "dash":
        if data != None:
            return pages.dashboard.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False, False
    elif path == "metafeatures":
        if data != None:
            return pages.metafeatures.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False, False
    elif path == "train":
        # Checking if exists a model in the server. If it does, it returns the predict page.
        if data != None:
            return pages.train.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False, False
    else:
        return "404"

#Dashboard Callbacks

@app.callback(
    Output("input-feature-dropdown", "options"),  # Parámetro de salida: opciones para el dropdown de características de entrada
    Output("class-feature-dropdown", "options"),  # Parámetro de salida: opciones para el dropdown de características de clase
    Input("stored-data2", "data"),  # Parámetro de entrada: datos almacenados (data) que provienen de la carga de un archivo
)
def update_dropdown_options(data):
    """
    Esta función actualiza las opciones de los dropdowns de características de entrada y clase
    con base en las columnas del dataframe proporcionado.

    Parámetros:
    - data: (list) Los datos cargados en la aplicación (usualmente un conjunto de datos en formato JSON).

    Retorna:
    - (list, list): Dos listas que contienen las opciones para los dropdowns, basadas en las columnas del dataframe.
      Cada opción es un diccionario con claves 'label' y 'value'.
    """
    if data:
        df = pd.DataFrame(data)  # Convierte los datos a un DataFrame de pandas
        # Crea las opciones para ambos dropdowns, donde 'label' y 'value' son las columnas del dataframe
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options  # Retorna las mismas opciones para ambos dropdowns
    return [], []  # Si no hay datos, retorna listas vacías para los dropdowns


@app.callback(
    [Output("output-graph", "figure"),  # Parámetro de salida: La figura del gráfico generado
     Output('error-message-plot', 'children')],  # Parámetro de salida: Mensaje de error si ocurre algún problema
    [Input("generate-plot", "n_clicks"),  # Parámetro de entrada: Número de clics en el botón para generar el gráfico
     Input("language", "data")],  # Parámetro de entrada: El idioma seleccionado por el usuario
    State("stored-data2", "data"),  # Estado: Los datos cargados por el usuario
    State("input-feature-dropdown", "value"),  # Estado: Característica seleccionada para el eje x o variable de entrada
    State("class-feature-dropdown", "value"),  # Estado: Característica seleccionada para la variable de clase (color)
    State("graph-type-dropdown", "value"),  # Estado: Tipo de gráfico seleccionado (ej. scatter, bar, etc.)
)
def update_graph(n_clicks, lang, data, input_feature, class_feature, graph_type):
    """
    Actualiza el gráfico según la selección de los parámetros y el tipo de gráfico. Si no se seleccionan todas las opciones,
    devuelve un mensaje de alerta.

    Parámetros:
    - n_clicks: (int) El número de clics en el botón para generar el gráfico
    - lang: (str) El idioma actual seleccionado por el usuario
    - data: (list) Los datos cargados en la aplicación (generalmente en formato JSON)
    - input_feature: (str) La característica seleccionada para el eje x o la variable de entrada
    - class_feature: (str) La característica seleccionada para la variable de clase
    - graph_type: (str) El tipo de gráfico seleccionado (scatter, bar, pie, etc.)

    Retorna:
    - (figure, error_message): 
        - figure: El gráfico generado, que se muestra en la aplicación.
        - error_message: Un mensaje de error en caso de que falte alguna opción o no se pueda generar el gráfico.
    """
    if n_clicks > 0:
        if not input_feature or not class_feature or not graph_type:
            # Si alguna de las opciones (input_feature, class_feature, graph_type) no se selecciona, retorna un mensaje de error
            return {}, dbc.Alert(t("Plot-Alert", lang), color="danger")
        
        if data:
            df = pd.DataFrame(data)  # Convierte los datos a un DataFrame de pandas

            # Manejo de diferentes tipos de gráficos
            if graph_type == "scatter":
                # Crea un gráfico de dispersión (scatter)
                fig = plx.scatter(df, x=input_feature[0], color=class_feature, title=f'{input_feature[0]} vs {class_feature}')
            elif graph_type == "bar":
                # Crea un gráfico de barras (bar)
                fig = plx.bar(df, x=input_feature[0], color=class_feature, title=f'{input_feature[0]} vs {class_feature}')
            elif graph_type == "pie":
                # Crea un gráfico de pastel (pie)
                fig = plx.pie(df, names=class_feature, title=f'Distribution of {class_feature}')
            elif graph_type == "histogram":
                # Crea un gráfico de histograma
                fig = plx.histogram(df, x=input_feature[0], color=class_feature, title=f'Distribution of {input_feature[0]}')
            elif graph_type == "line":
                # Crea un gráfico de líneas
                fig = plx.line(df, x=input_feature[0], y=input_feature[1], color=class_feature, title=f'{input_feature[0]} vs {input_feature[1]}')
            elif graph_type == "box":
                # Crea un gráfico de cajas (box plot)
                fig = plx.box(df, x=class_feature, y=input_feature[0], title=f'Box Plot of {input_feature[0]} by {class_feature}')
            elif graph_type == "heatmap":
                # Crea un gráfico de mapa de calor basado en la correlación de las características
                correlation = df.corr()  # Calcular la correlación
                fig = plx.imshow(correlation, title='Heatmap of Correlation')
            elif graph_type == "area":
                # Crea un gráfico de área
                fig = plx.area(df, x=input_feature[0], y=input_feature[1], title=f'Area Chart of {input_feature[0]} and {input_feature[1]}')

            return fig, html.Div()  # Si todo está correcto, devuelve el gráfico y no hay mensaje de error

    return {}, html.Div()  # Si no se ha hecho clic en el botón, retorna un gráfico vacío y sin mensaje de error


# Callback para descargar las meta-características en formato CSV
@app.callback(
    Output('download-component', 'data'),  # Parámetro de salida: Enlace de descarga para el archivo CSV
    Input("download-csv", "n_clicks"),  # Parámetro de entrada: Número de clics en el botón de descarga
    State("stored-data2", "data"),  # Estado: Datos cargados en la aplicación
    State("input-feature-dropdown1", "value"),  # Estado: Característica seleccionada para la entrada
    State("class-feature-dropdown1", "value"),  # Estado: Característica seleccionada para la clase
)
def update_download_link(n_clicks, data, input_feature, class_feature):
    """
    Este callback genera un enlace de descarga para las meta-características extraídas de los datos proporcionados.
    
    Parámetros:
    - n_clicks: (int) El número de clics en el botón de descarga.
    - data: (list) Los datos cargados en la aplicación (en formato JSON).
    - input_feature: (str) Característica seleccionada para la entrada.
    - class_feature: (str) Característica seleccionada para la clase.

    Retorna:
    - Un enlace para descargar un archivo CSV con las meta-características.
    """
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)
        X = df[input_feature].to_numpy()
        y = df[class_feature].to_numpy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        cc = px.ComplexityCalculator()
        cc.fit(X, y)
        cc._metrics()

        report = cc.report()

        # Creating the complexity data in a DataFrame
        complexity_data = {
            "Metric": [],
            "Value": []
        }

        # Fill complexity data for Linearity measures, Feature-based measures, etc.
        complexity_data["Metric"] += ["l1", "l2", "l3"]
        complexity_data["Value"] += [report['complexities']['l1'],
                                        report['complexities']['l2'],
                                        report['complexities']['l3']]

        complexity_data["Metric"] += ["f1", "f1v", "f2", "f3", "f4"]
        complexity_data["Value"] += [report['complexities']['f1'],
                                        report['complexities']['f1v'],
                                        report['complexities']['f2'],
                                        report['complexities']['f3'],
                                        report['complexities']['f4']]

        complexity_data["Metric"] += ["c1", "c2"]
        complexity_data["Value"] += [report['complexities']['c1'],
                                        report['complexities']['c2']]

        complexity_data["Metric"] += ["t2", "t3", "t4"]
        complexity_data["Value"] += [report['complexities']['t2'],
                                        report['complexities']['t3'],
                                        report['complexities']['t4']]

        complexity_data["Metric"] += ["hubs", "clsCoef", "density"]
        complexity_data["Value"] += [report['complexities']['hubs'],
                                        report['complexities']['clsCoef'],
                                        report['complexities']['density']]

        complexity_data["Metric"] += ["lsc", "t1", "n1", "n2", "n3", "n4"]
        complexity_data["Value"] += [report['complexities']['lsc'],
                                        report['complexities']['t1'],
                                        report['complexities']['n1'],
                                        report['complexities']['n2'],
                                        report['complexities']['n3'],
                                        report['complexities']['n4']]

        # Meta-feature extraction
        mfe_groups = ["clustering", "concept", "general", "statistical", "info-theory"]
        for group in mfe_groups:
            mfe = MFE(groups=[group])
            mfe.fit(X, y)
            ft_names, ft_values = mfe.extract()
            for name, value in zip(ft_names, ft_values):
                complexity_data["Metric"].append(f"{group}_{name}")
                complexity_data["Value"].append(value)

        complexity_df = pd.DataFrame(complexity_data)
        
        # Crear el enlace de descarga para el archivo CSV
        return dcc.send_data_frame(complexity_df.to_csv, "metafeatures.csv")
    return None

# Callback para actualizar las opciones de los dropdowns para las características de entrada y clase
@app.callback(
    Output("input-feature-dropdown1", "options"),  # Parámetro de salida: Opciones para el dropdown de características de entrada
    Output("class-feature-dropdown1", "options"),  # Parámetro de salida: Opciones para el dropdown de características de clase
    Input("stored-data2", "data"),  # Parámetro de entrada: Datos cargados en la aplicación
)
def update_dropdown_options_(data):
    """
    Actualiza las opciones de los dropdowns con las columnas de los datos cargados.
    
    Parámetros:
    - data: (list) Los datos cargados en la aplicación.
    
    Retorna:
    - options: Opciones actualizadas para los dropdowns de entrada y clase.
    """
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    [Output("output-image", "src"),  # Salida: La imagen de complejidad generada en formato base64.
     Output("output-image", "hidden"),  # Salida: Si se oculta o no la imagen generada.
     Output("download-csv", "hidden"),  # Salida: Si se oculta o no la opción de descargar el CSV.
     Output("output-meta-features", "children"),  # Salida: El reporte con las metacaracterísticas extraídas.
     Output("error-message1", "children")],  # Salida: Mensaje de error si ocurre alguna excepción.
    Input("calculate_meta_features", "n_clicks"),  # Entrada: Número de clics en el botón "calculate_meta_features".
    State('language', 'data'),  # Estado: El idioma seleccionado por el usuario.
    State("stored-data2", "data"),  # Estado: Los datos cargados en la aplicación.
    State("input-feature-dropdown1", "value"),  # Estado: Característica seleccionada para la entrada.
    State("class-feature-dropdown1", "value"),  # Estado: Característica seleccionada para la clase.
)
def calculate_complexity_and_meta_features(n_clicks, lang, data, input_feature, class_feature):
    """
    Este callback calcula la complejidad del conjunto de datos y extrae las metacaracterísticas.
    Muestra un gráfico y un reporte de las metacaracterísticas calculadas.

    Parámetros:
    - n_clicks: (int) Número de clics en el botón de cálculo de metacaracterísticas.
    - lang: (str) El idioma seleccionado por el usuario.
    - data: (list) Los datos cargados en la aplicación.
    - input_feature: (str) La característica seleccionada para la entrada.
    - class_feature: (str) La característica seleccionada para la clase.

    Retorna:
    - La imagen de complejidad en formato base64.
    - Un reporte con las metacaracterísticas extraídas.
    - Mensaje de error si ocurre algún problema durante el proceso.
    """
    
    error_message = None

    # Verifica si el botón ha sido presionado
    if n_clicks > 0:
        # Verifica si se han seleccionado las características de entrada y clase
        if not input_feature or not class_feature:
            # Si no se seleccionaron las características, muestra un mensaje de error
            error_message = dbc.Alert(t("Plot-Alert", lang), color="danger")
            return '', True, True, '', error_message

        if data:  # Si los datos están disponibles
            try:
                # Convierte los datos en un DataFrame de pandas
                df = pd.DataFrame(data)
                X = df[input_feature].to_numpy()  # Extrae las características de entrada
                y = df[class_feature].to_numpy()  # Extrae las características de clase

                if X.ndim == 1:  # Si X tiene solo una dimensión, remodela a 2D
                    X = X.reshape(-1, 1)

                # Calcula las complejidades con el objeto ComplexityCalculator
                cc = px.ComplexityCalculator()
                cc.fit(X, y)  # Ajusta el calculador de complejidad a los datos
                cc._metrics()  # Calcula las métricas de complejidad

                # Crea la figura para el gráfico de complejidad
                fig = plt.figure(figsize=(7, 7))
                cc.plot(fig, (1, 1, 1))
                buffer = io.BytesIO()  # Crea un buffer en memoria para almacenar la imagen
                plt.savefig(buffer, format='png')  # Guarda la figura en el buffer en formato PNG
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Convierte la imagen a base64
                plt.close(fig)  # Cierra la figura de matplotlib

                # Genera el reporte de complejidad
                report = cc.report()

                # Función para crear tablas de métricas y valores
                def create_table(title, metrics):
                    table_header = [html.Thead(html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))]))]
                    table_rows = [html.Tr([html.Td(key, title=t(key, lang)), html.Td(value)]) for key, value in metrics]
                    table_body = [html.Tbody(table_rows)]
                    return html.Div([
                        html.H3(t(title, lang)),  # Título de la tabla
                        html.Hr(),
                        dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True),  # Crea la tabla
                        html.Br()
                    ])

                # Crea tablas con las métricas de complejidad
                tables = [
                    create_table("Linearity-measures", [("l1", report['complexities']['l1']),
                                                         ("l2", report['complexities']['l2']),
                                                         ("l3", report['complexities']['l3'])]),
                    create_table("Feature-based-measures", [("f1", report['complexities']['f1']),
                                                             ("f1v", report['complexities']['f1v']),
                                                             ("f2", report['complexities']['f2']),
                                                             ("f3", report['complexities']['f3']),
                                                             ("f4", report['complexities']['f4'])]),
                    create_table("Class-imbalance-measures", [("c1", report['complexities']['c1']),
                                                              ("c2", report['complexities']['c2'])]),
                    create_table("Dimensionality-measures", [("t2", report['complexities']['t2']),
                                                              ("t3", report['complexities']['t3']),
                                                              ("t4", report['complexities']['t4'])]),
                    create_table("Network-measures", [("hubs", report['complexities']['hubs']),
                                                       ("clsCoef", report['complexities']['clsCoef']),
                                                       ("density", report['complexities']['density'])]),
                    create_table("Neighborhood-measures", [("lsc", report['complexities']['lsc']),
                                                           ("t1", report['complexities']['t1']),
                                                           ("n1", report['complexities']['n1']),
                                                           ("n2", report['complexities']['n2']),
                                                           ("n3", report['complexities']['n3']),
                                                           ("n4", report['complexities']['n4'])])
                ]

                # Define grupos de metacaracterísticas para el cálculo adicional
                mfe_groups = ["clustering", "concept", "general", "statistical", "info-theory"]
                mfe_tables = []
                for group in mfe_groups:
                    mfe = MFE(groups=[group])  # Crea un extractor de metacaracterísticas para el grupo
                    mfe.fit(X, y)  # Ajusta los datos al extractor de metacaracterísticas
                    ft_names, ft_values = mfe.extract()  # Extrae las metacaracterísticas
                    mfe_tables.append(create_table(f"{t('Meta-Features-of-the-group', lang)}: {group.capitalize()}",
                                                   zip(ft_names, ft_values)))  # Crea una tabla con las metacaracterísticas

                final_report = html.Div(tables + mfe_tables)  # Combina todas las tablas en un solo reporte

                # Retorna la imagen en formato base64, oculta la imagen, oculta el CSV y muestra el reporte final
                return f'data:image/png;base64,{image_base64}', False, False, final_report, None

            except Exception as e:
                # Si ocurre un error, muestra un mensaje de error
                error_message = dbc.Alert(f"{t('Error', lang)}: {str(e)}", color="danger")
                return '', True, True, '', error_message

    # Si no se ha presionado el botón, oculta todos los elementos de salida
    return '', True, True, '', None


# A way to run the app in a local server.
if __name__ == "__main__":
    app.run_server(debug=True)

