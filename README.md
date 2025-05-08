## Data Analysis and Complexity Calculation Project
This project is an application designed to facilitate data analysis and the calculation of various complexity measures in datasets. Users can upload files in CSV or ARFF format, visualize the data in a table, generate charts, and compute a variety of statistical and complexity measures.

## Main Features
File Upload and Validation

    The system allows users to upload a file in CSV or ARFF format.
    The system validates that the uploaded file is valid and displays an error message if it is not.

Data Visualization

    The system displays a table with the uploaded data, allowing for clear and organized visualization.

Chart Generation

    The system supports the generation of bar charts, pie charts, histograms, and scatter plots for better data visualization.
    Users can select the data attributes they wish to include in the charts.

Complexity Measures Calculation

The system supports the calculation of various complexity measures, categorized as follows:

    Class Imbalance Measures:
        Calculation of entropy and imbalance ratio.

    Correlation Measures:
        Calculation of maximum and mean correlation.

    Dimensionality Measures:
        Calculation of the average number of features per dimension.

    Feature-Based Measures:
        Calculation of F1, F1v, F2, F3, and F4.

    Geometric Measures:
        Calculation of the nonlinearity of a linear regressor.

    Neighborhood Measures:
        Calculation of n1, n2, n3, n4, t1, and lsc.

    Network Measures:
        Calculation of density and clustering coefficient.

    Smoothness Measures:
        Calculation of the output distribution measure.

    Meta-features:
        The system allows for the calculation of meta-features.

Results Saving and Visualization

    The system allows saving the calculated complexity measures to an output file for further analysis.
    Users can view the results of the complexity measures directly in the application's interface.

## Running The Script

In order to access the application, open up Terminal and make sure to change the working directory to exactly where you saved this folder:

* Create a virtual environment and activate it
* Run __pip install -r requirements.txt__
* Run __python3 index.py__ on Terminal
* Your app will be accesible on any web browswer at the address: 127.0.0.1:8050

## Built With
* [Dash](https://dash.plot.ly)



