from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from utils.helpers import *
from utils.figures import *
from app import *
from data_reader import *

# Setting the font size, color, and background color for the LED displays.
FONTSIZE = 20
FONTCOLOR = '#F5FFFA'
BGCOLOR ='#3445DB'

def layout():
    return [
    ]