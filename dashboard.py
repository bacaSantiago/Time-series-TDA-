import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

"""_EDA_
This section contains the exploratory data analysis functions for the dashboard.
"""



"""_Cluster_
This section contains the clustering functions for the dashboard.
"""


"""_TDA_
This section contains the topological data analysis functions for the dashboard.
"""


"""_Dash_
This section contains the functions for the Dash web application.
"""

eda_tab = dbc.Tab(
    label="Exploratory Data Analysis",
    children=[
        html.H2("Exploratory Data Analysis", className="text-center mt-4"),
        html.P(
            "Here you can place your summary statistics, summary tables, "
            "time-series plots of prices, rating distributions, etc.",
            className="text-center mb-4"
        ),
        # Example row of two placeholder graphs
        dbc.Row([
            dbc.Col(dcc.Graph(id="price-trend-graph", figure={}), width=6),
            dbc.Col(dcc.Graph(id="rating-distribution", figure={}), width=6),
        ], className="mb-5"),
    ],
)

clustering_tab = dbc.Tab(
    label="Clustering",
    children=[
        html.H2("Clustering Analysis", className="text-center mt-4"),
        html.P(
            "K-Means, hierarchical or time-series clustering results go here.",
            className="text-center mb-4"
        ),
        dcc.Graph(id="clustering-graph", figure={}),
    ],
)

tda_tab = dbc.Tab(
    label="Topological Data Analysis",
    children=[
        html.H2("Topological Data Analysis", className="text-center mt-4"),
        html.P(
            "Persistent homology diagrams, Mapper graph visualizations, "
            "and topological summaries.",
            className="text-center mb-4"
        ),
        dcc.Graph(id="tda-persistence", figure={}),
        dcc.Graph(id="tda-mapper", figure={}),
    ],
)

extra_tab = dbc.Tab(
    label="Extra",
    children=[
        html.H2("Additional Analyses / Notes", className="text-center mt-4"),
        html.P(
            "Use this space for anomaly detection, anomaly tables, "
            "or links to raw data download.",
            className="text-center mb-4"
        ),
    ],
)



# Create the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    title="Topological Analysis of Airbnb Price Dynamics in Cartagena"
)
server = app.server

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Topological Analysis of Airbnb Price Dynamics in Cartagena",
            color="dark",
            dark=True,
            fluid=True,
        ),
        dbc.Tabs([eda_tab, clustering_tab, tda_tab, extra_tab], className="mt-3"),
        html.Footer(
            dbc.Container(
                [
                    html.Hr(),
                    html.P(
                        "© 2025 Cartagena Airbnb TDA Project | Santiago Baca", 
                        className="text-center text-muted",
                        style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}
                    ),
                ],
                style={"marginTop": "20px"}
            ),
            style={"backgroundColor": "#222", "color": "white", "padding": "20px"},
        ),
    ],
    fluid=True,
    style={"padding": "20px"}
)

if __name__ == "__main__":
    app.run(debug=True)