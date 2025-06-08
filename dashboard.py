import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.express as px
import re

# Load the datasets from SQLite database
_conn = sqlite3.connect("airbnb_cartagena.sqlite")
df_attr = pd.read_sql_query("SELECT * FROM Attributes", _conn, dtype={"ID": str})
df_ts = pd.read_sql_query("SELECT * FROM TimeSeriesRaw", _conn, dtype={"ID": str})
df_ts_interp = pd.read_sql_query("SELECT * FROM TimeSeriesInterpolated", _conn, dtype={"ID": str})
_conn.close()

red = "#7e0d24"  # dark red color for plots
dates = [col for col in df_ts.columns if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", col)]



"""_EDA_
-----------------------------------------------------------------------------------------------------------
This section contains the exploratory data analysis functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""

# 1.1 Distribution of Base Fees
def base_fee_distribution():
    """
    Histogram (with rug) of the Base fee across all listings,
    shown as a density, with an overlaid KDE curve.
    """
    # Histogram
    x = df_attr["Base fee"].dropna()
    fig = px.histogram(
        x=x,
        nbins=50,
        histnorm="probability density",       
        marginal="box",
        title="Distribution of Base Fees",
        template="plotly_dark",
        color_discrete_sequence=[red]
    )
    # Kernel Density Estimation (KDE)
    kde = stats.gaussian_kde(x)
    x_range = np.linspace(x.min(), x.max(), 200)
    y_kde = kde(x_range)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_kde,
            mode="lines",
            name="KDE",
            line=dict(color="#ffffff", width=2)
        )
    )
    fig.update_layout(
        xaxis_title="Base Fee (USD)",
        yaxis_title="Density",
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# 1.2 Cleaning Fee vs. Base Fee
def cleaning_vs_base_fee():
    """
    Scatter of Cleaning fee vs. Base fee, with an OLS regression line styled and labeled.
    """
    fig = px.scatter(
        df_attr,
        x="Base fee",
        y="Cleaning fee",
        marginal_x="violin",
        marginal_y="violin",
        trendline="ols",
        trendline_color_override="white",
        title="Cleaning Fee vs. Base Fee",
        labels={"Base fee": "Base Fee (USD)", "Cleaning fee": "Cleaning Fee (USD)"},
        template="plotly_dark",
        color_discrete_sequence=[red],
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))  
    fig.update_layout(
        xaxis_title="Base Fee (USD)",
        yaxis_title="Cleaning Fee (USD)"
    )
    return fig

# 1.3 Ratings Summary
def ratings_summary():
    """
    Boxplots of each rating category to compare medians and variability.
    """
    rating_cols = [c for c in df_attr.columns if c.endswith("_rating")]
    df_long = df_attr[rating_cols].copy().melt(var_name="Rating", value_name="Score")
    fig = px.box(
        df_long,
        x="Rating",
        y="Score",
        notched=True,
        points="suspectedoutliers",
        title="Distribution of Rating Scores by Category",
        labels={"Score": "Rating Score", "Rating": "Category"},
        template="plotly_dark",
        color_discrete_sequence=[red]
    )
    fig.update_layout(xaxis_tickangle=-15)
    return fig

# 1.4 Property Type Counts
def property_type_counts():
    """
    Pie chart showing count of listings by property type.
    """
    counts = df_attr["Property type"].value_counts().reset_index()
    counts.columns = ["Property type", "Count"]
    labels = df_attr["Property type"].unique()
    fig = px.pie(
        counts,
        names="Property type",
        values="Count",
        title="Listing Counts by Property Type",
        hole=0.3,
        color ="Property type",
        color_discrete_map={
            labels[3]: "#5e0a18",
            labels[1]: red,
            labels[0]: "#9e1f34",
            labels[2]: "#b02a3f"
        },
        template="plotly_dark"
    ) 
    return fig

# 1.5 Amenity Presence Lollipop
def amenity_presence_lollipop():
    """
    Lollipop chart of percentage of listings that offer each amenity.
    """
    pct = (df_attr.iloc[:, 21:].mean() * 100).round(1).sort_values()
    fig = go.Figure([
        go.Scatter(
            x=pct.values,
            y=pct.index,
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='none'
        ),
        go.Scatter(
            x=pct.values,
            y=pct.index,
            mode='markers',
            marker=dict(
                size=8,
                color=pct.values,
                colorscale='amp',
                cmin=-20,
                cmax=120,
                showscale=False
            ),
            showlegend=False,
            name='%',
        )
    ])
    fig.update_layout(
        template='plotly_dark',
        title="Amenity Presence Across Listings (%)",
        xaxis_title="% of Listings",
        yaxis_title="Amenity",
        yaxis=dict(autorange='reversed'),
        height=800,
        margin=dict(l=200, r=40, t=80, b=40)
    )
    return fig

# 1.6 Price vs. Number of Bedrooms/Baths
def price_vs_size_boxplots():
    """
    Boxplots of Base fee grouped by Bedrooms and Baths count.
    """
    # Create a long format DataFrame for boxplot
    df_bed = df_attr[["Base fee", "Bedrooms"]].copy().rename(columns={"Bedrooms": "Count"})
    df_bed["Type"] = "Bedrooms"
    df_bath = df_attr[["Base fee", "Baths"]].copy().rename(columns={"Baths": "Count"})
    df_bath["Type"] = "Baths"
    df_long = pd.concat([df_bed, df_bath], axis=0)
    df_long = df_long[df_long["Count"] != 4]
    df_long["Count"] = df_long["Count"].astype(str)
    
    # Create figure
    fig = px.box(
        df_long,
        x="Count",
        y="Base fee",
        points="suspectedoutliers",
        color="Type",
        title="Base Fee by Number of Bedrooms and Baths",
        labels={"Count": "Count", "Base fee": "Base Fee (Nightly)", "Type": "Dimension"},
        color_discrete_map={
            "Bedrooms": red,
            "Baths": "#cb1e3b"
        },
        template="plotly_dark"
    )
    fig.update_layout(boxmode="group")
    return fig

# 1.7 Price vs. Person Capacity and Beds
def price_vs_capacity_beds_boxplots():
    """
    Boxplots of Base fee grouped by Person capacity and Beds count.
    """
    # Create a long format DataFrame for boxplot
    df_cap = (
        df_attr[["Base fee", "Person capacity"]]
        .rename(columns={"Person capacity": "Count"})
        .assign(Type="Capacity")
    )
    df_beds = (
        df_attr[["Base fee", "Beds"]]
        .rename(columns={"Beds": "Count"})
        .assign(Type="Beds")
    )
    df_long = pd.concat([df_cap, df_beds], axis=0)
    df_long["Count"] = df_long["Count"].astype(str)

    # Create figure
    fig = px.box(
        df_long,
        x="Count",
        y="Base fee",
        color="Type",
        points="suspectedoutliers",
        title="Base Fee by Person Capacity and Beds",
        labels={
            "Count": "Count",
            "Base fee": "Base Fee (Nightly)",
            "Type": "Dimension"
        },
        color_discrete_map={
            "Capacity": red,
            "Beds": "#cb1e3b",
        },
        category_orders={
            "Count": sorted(df_long["Count"].unique(), key=lambda x: float(x))
        },
        template="plotly_dark"
    )
    fig.update_layout(boxmode="group")
    return fig

# 2.1 Price Availability Timeline
def price_availability_timeline():
    """
    For each calendar date, show percent of listings with a valid price (i.e. != NaN).
    """
    # Calculate the percentage of listings with a valid price for each date
    avail = df_ts[dates].apply(lambda col: col.notna().sum() / len(df_ts) * 100).reset_index()
    avail.columns = ["Date", "% Available"]
    avail["Date"] = pd.to_datetime(avail["Date"], dayfirst=True, format="%d/%m/%Y")
    avail = avail.sort_values("Date")
    
    # Create figure
    fig = px.line(
        avail,
        x="Date",
        y="% Available",
        title="Daily % of Listings with a Valid Price",
        template="plotly_dark",
        color_discrete_sequence=[red]
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(range=[0, 100])
    return fig

"""
# 2.2 Missing-Data Gap Distribution
def gap_distribution_histogram():
    #Histogram of max_nan_gap across listings to show how many have big contiguous missing spans.
    fig = px.histogram(
        df_ts,
        x="max_nan_gap",
        nbins=30,
        title="Distribution of Maximum Consecutive Missing-Price Gaps",
        labels={"max_nan_gap": "Max Consecutive Missing Days", "count": "Number of Listings"},
        template="plotly_dark"
    )
    return fig
"""

# 2.2 Missing-Data Gap Distribution
def top_gap_table():
    """
    Returns a Dash DataTable showing the 10 listings
    with the largest consecutive missing-price gaps.
    """
    # build the top-10 DataFrame
    df_top10 = (
        df_ts
        .loc[:, ["ID", "Name", "max_nan_gap"]]
        .sort_values("max_nan_gap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    
    # build the Dash table
    return dash_table.DataTable(
        id="top-gap-table",
        columns=[
            {"name": "ID",          "id": "ID"},
            {"name": "Listing Name","id": "Name"},
            {"name": "Max Gap (days)","id": "max_nan_gap"},
        ],
        data=df_top10.to_dict("records"),  # type: ignore
        style_header={
            "backgroundColor": "#111111",
            "fontWeight": "bold",
            "color": "#ffffffc4",
            "fontSize": "12px",
            "textAlign": "center"
        },
        style_cell={
            "backgroundColor": "#111111",
            "color": "#ffffffc4",
            "textAlign": "left",
            "fontSize": "12px",
            "padding": "5px"
        },
        style_table={"overflowX": "auto"},
    )

# 2.3 Sample Price Trends (Raw)
def sample_price_trends():
    """
    Draw 6 small-multiples showing the raw daily price series
    (including NaN gaps) for the listings with the most missing entries.
    """
    # Pick the 6 listings with the most missing values
    df = df_ts.copy()
    df["missing_total"] = df[dates].isna().sum(axis=1)
    top3_missing = df.nlargest(3, "missing_total")["ID"].tolist()
    clean3_interp = (
        df_ts_interp
        .assign(**{"missing_interp": df_ts_interp[dates].isna().sum(axis=1)})
        .query("missing_interp == 0").head(3)["ID"].tolist()
    )
    sample_ids = top3_missing + clean3_interp

    # Melt listings into long format
    df_long = (
        df.loc[df["ID"].isin(sample_ids), ["ID", "Name"] + dates]
          .melt(id_vars=["ID", "Name"], var_name="Date", value_name="Price")
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, format="%d/%m/%Y")
    df_long = df_long.sort_values(["ID", "Date"])

    # Create figure
    fig = px.line(
        df_long,
        x="Date",
        y="Price",
        facet_col="ID",
        facet_col_wrap=2,
        color="ID",
        line_group="ID",
        category_orders={"ID": sample_ids},
        title=f"Top {len(sample_ids)} Listings by Total Missing Values",
        labels={"Price": "Price (USD/night)", "Date": "Date"},
        template="plotly_dark",
        color_discrete_sequence=[red]
    )
    fig.update_layout(
        showlegend=False,
        height=300 * 3,
    )
    fig.for_each_annotation(lambda a: a.update(text=f"ID: {a.text.split('=')[1]}"))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)

    return fig

# 3.1 Compare Raw vs. Interpolated for Sample
def compare_raw_vs_interpolated():
    """
    Overlay raw (dashed) vs. interpolated (solid) daily price series
    for the 6 listings with the most missing entries —
    interpolated in red, raw in white (drawn on top).
    """
    # Pick the 6 listings with the most missing values
    df = df_ts.copy()
    df["missing_total"] = df[dates].isna().sum(axis=1)
    top3_missing = df.nlargest(3, "missing_total")["ID"].tolist()
    clean3_interp = (
        df_ts_interp
        .assign(missing_interp=lambda d: d[dates].isna().sum(axis=1))
        .query("missing_interp == 0").head(3)["ID"].tolist()
    )
    sample_ids = top3_missing + clean3_interp

    # Melt listings into long format
    raw = (
        df_ts
        .loc[df_ts["ID"].isin(sample_ids), ["ID","Name"] + dates]
        .melt(id_vars=["ID","Name"], var_name="Date", value_name="RawPrice")
    )
    raw["RawPrice"] = raw["RawPrice"].astype(float)
    interp = (
        df_ts_interp
        .loc[df_ts_interp["ID"].isin(sample_ids), ["ID","Name"] + dates]
        .melt(id_vars=["ID","Name"], var_name="Date", value_name="InterpPrice")
    )
    interp["InterpPrice"] = interp["InterpPrice"].astype(float)

    # Merge raw and interpolated dataframes
    df_cmp = pd.merge(raw, interp, on=["ID","Name","Date"])
    df_cmp["Date"] = pd.to_datetime(df_cmp["Date"], dayfirst=True, format="%d/%m/%Y")
    df_long = df_cmp.melt(
        id_vars=["ID","Name","Date"],
        value_vars=["RawPrice","InterpPrice"],
        var_name="Source",
        value_name="Price"
    ).replace({"Source": {"RawPrice": "Raw", "InterpPrice": "Interpolated"}})

    # Create the plot
    fig = px.line(
        df_long,
        x="Date", y="Price",
        color="Source",
        category_orders={"Source": ["Interpolated", "Raw"], "ID": sample_ids},
        line_dash="Source",
        line_dash_map={"Raw": "dash", "Interpolated": "solid"},
        color_discrete_map={"Interpolated": red, "Raw": "white"},
        facet_col="ID", facet_col_wrap=2,
        title="Raw vs. Interpolated Price Series",
        labels={"Price":"Price (USD/night)", "Date":"Date"},
        template="plotly_dark"
    )
    fig.update_traces(
        selector=dict(name="Raw"),
        line=dict(width=2),
        opacity=0.8
    )
    fig.update_layout(
        showlegend=True,
        height=300 * 3,
        legend_title_text="Series"
    )
    fig.for_each_annotation(lambda a: a.update(text=f"ID: {a.text.split('=')[1]}"))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)

    return fig

# 3.2 Weekly/Monthly Average Price Trends
def average_price_trend(freq: str = "W"):
    """
    Compute and plot the average daily price across all listings,
    aggregated by week ('W') or month ('M').
    """
    # Melt the DataFrame to long format
    df_long = (
        df_ts_interp
        .melt(id_vars=["ID"], value_vars=dates,
              var_name="Date", value_name="Price")
    )
    df_long["Price"] = df_long["Price"].astype(float)
    df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, format="%d/%m/%Y")

    # Group by date and calculate the mean price
    daily = df_long.groupby("Date")["Price"].mean().reset_index()
    daily = daily.set_index("Date")
    if freq.upper() == "W":
        agg = daily.resample("W-MON")["Price"].mean().reset_index().sort_values("Date")
        title = "Weekly Average Price Trend"
    elif freq.upper() == "M":
        agg = daily.resample("M")["Price"].mean().reset_index().sort_values("Date")
        title = "Monthly Average Price Trend"
    else:
        raise ValueError("freq must be 'W' or 'M'")

    # Create the line plot
    fig = px.line(
        agg,
        x="Date",
        y="Price",
        title=title,
        labels={"Price": "Avg. Price (USD/night)", "Date": "Date"},
        template="plotly_dark",
        color_discrete_sequence=[red]
    )
    return fig

# 3.3 Price Volatility Heatmap
def price_volatility_heatmap():
    """
    Build a heatmap of (price - mean_price)/mean_price for each listing/date.
    X = dates, Y = listing IDs.
    """
    # Calculate deviation from mean price
    mat = df_ts_interp.set_index("ID")[dates].astype(float)
    row_means = mat.mean(axis=1)
    dev = (mat.sub(row_means, axis=0)).div(row_means, axis=0)
    dev = dev[~dev.isna().any(axis=1)]
    dev = dev.loc[ dev.abs().max(axis=1).sort_values(ascending=True).index]
    date_index = pd.to_datetime(dates, dayfirst=True, format="%d/%m/%Y")
    ids = dev.index.astype(str)
    
    # Create figure
    fig = go.Figure(
        go.Heatmap(
            z=dev.values,
            x=date_index,
            y=ids,
            colorscale='oxy',
            zmid=0,
            colorbar=dict(title="Volatility"),
            hovertemplate="Listing: %{y}<br>Date: %{x|%Y-%m-%d}<br>Dev: %{z:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        title="Price Volatility (Deviation from Listing Mean)",
        xaxis_title="Date",
        yaxis_title="Listing ID",
        template="plotly_dark",
        height=700
    )
    return fig




"""_Cluster_
-----------------------------------------------------------------------------------------------------------
This section contains the clustering functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""


"""_TDA_
-----------------------------------------------------------------------------------------------------------
This section contains the topological data analysis functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""


"""_Dash_
-----------------------------------------------------------------------------------------------------------
This section contains the functions for the Dash web application.
-----------------------------------------------------------------------------------------------------------
"""

eda_tab = dbc.Tab(
    label="Exploratory Data Analysis",
    children=[
        html.H2("Exploratory Data Analysis", className="text-center mt-4"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="base-fee-dist", figure=base_fee_distribution()),
                width=6
            ),
            dbc.Col(
                dcc.Graph(id="cleaning-vs-base", figure=cleaning_vs_base_fee()),
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="ratings-summary", figure=ratings_summary()), 
                width=6
            ),
            dbc.Col(
                dcc.Graph(id="property-type-counts", figure=property_type_counts()), 
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="amenity-heatmap", figure=amenity_presence_lollipop()), 
                width=6
            ),
            dbc.Col([
                dcc.Graph(id="price-vs-size", figure=price_vs_size_boxplots()), 
                dcc.Graph(id="price-vs-capacity-beds", figure=price_vs_capacity_beds_boxplots()),
            ], width=6),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="price-availability", figure=price_availability_timeline()), 
                width=6
            ),
            dbc.Col(
                html.Div([
                    html.H5("Top 10 Listings by Missing-Price Gap",
                            style={"color": "white", "marginBottom": "10px"}),
                    top_gap_table()
                ]),
                width=6,
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="sample-price-trends", figure=sample_price_trends()), 
                width=5
            ),
            dbc.Col(
                dcc.Graph(id="compare-raw-interp", figure=compare_raw_vs_interpolated()), 
                width=7
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="weekly-price-trend", figure=average_price_trend("W")),
                width=6
            ),
            dbc.Col(
                dcc.Graph(id="volatility-heatmap", figure=price_volatility_heatmap()), 
                width=6
            ),
        ], className="mb-5"),
    ]
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
            color="#222",
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
            style={"backgroundColor": "#111111", "color": "white", "padding": "20px"},
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#111111", "padding": "20px"}
)

if __name__ == "__main__":
    app.run(debug=True)