import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Load the datasets from SQLite database
_conn = sqlite3.connect("airbnb_cartagena.sqlite")
df_attr = pd.read_sql_query("SELECT * FROM Attributes", _conn)
df_ts = pd.read_sql_query("SELECT * FROM TimeSeriesRaw", _conn)
df_ts_interp = pd.read_sql_query("SELECT * FROM TimeSeriesInterpolated", _conn)
_conn.close()


"""_EDA_
-----------------------------------------------------------------------------------------------------------
This section contains the exploratory data analysis functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""

# 1.1 Distribution of Base Fees
def base_fee_distribution():
    """
    Histogram (with rug) of the Base fee across all listings.
    """
    fig = px.histogram(
        df_attr,
        x="Base fee",
        nbins=50,
        marginal="rug",
        title="Distribution of Base Fees",
        labels={"Base fee": "Base Fee (USD)"},
        template="plotly_dark"
    )
    fig.update_layout(
        xaxis_title="Base Fee (USD)",
        yaxis_title="Count of Listings"
    )
    return fig

# 1.2 Cleaning Fee vs. Base Fee
def cleaning_vs_base_fee():
    """
    Scatter of Cleaning fee vs. Base fee, with an OLS regression line.
    """
    fig = px.scatter(
        df_attr,
        x="Base fee",
        y="Cleaning fee",
        trendline="ols",
        title="Cleaning Fee vs. Base Fee",
        labels={"Base fee": "Base Fee (USD)", "Cleaning fee": "Cleaning Fee (USD)"},
        template="plotly_dark"
    )
    # optionally tweak the trendline styling
    fig.update_traces(marker=dict(size=6, opacity=0.7))
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
    # select only the rating columns
    rating_cols = [
        "accuracy_rating",
        "checking_rating",
        "cleanliness_rating",
        "communication_rating",
        "location_rating",
        "value_rating",
        "satisfaction_rating"
    ]
    # melt into long form
    df_long = (
        df_attr[rating_cols]
        .rename(columns={
            "satisfaction_rating": "guestSatisfaction"
        })
        .melt(var_name="Rating", value_name="Score")
    )
    fig = px.box(
        df_long,
        x="Rating",
        y="Score",
        title="Distribution of Rating Scores by Category",
        labels={"Score": "Rating Score", "Rating": "Category"},
        template="plotly_dark"
    )
    fig.update_layout(xaxis_tickangle=-15)
    return fig

# 1.4 Property Type Counts
def property_type_counts():
    """
    Bar chart showing count of listings by property type.
    """
    counts = df_attr["Property type"].value_counts().reset_index()
    counts.columns = ["Property type", "Count"]
    fig = px.bar(
        counts,
        x="Property type",
        y="Count",
        title="Listing Counts by Property Type",
        labels={"Count": "Number of Listings"},
        template="plotly_dark"
    )
    fig.update_layout(xaxis_tickangle=-15)
    return fig

# 1.5 Amenity Presence Heatmap
def amenity_presence_heatmap():
    """
    Heatmap of percentage of listings that offer each key amenity.
    """
    # list here your amenity columns in the order you want them shown
    amenity_cols = [
        "City skyline view", "Lake view", "Beach view", "Sea view",
        "Jacuzzi", "Shared pool", "Shared gym",
        "Outdoor furniture", "Sun loungers",
        "Washer", "Dryer", "Room-darkening shades",
        "AC", "Heating",
        "Wifi", "Dedicated workspace",
        "Crib", "Window guards", "Babysitter recommendations", "Outdoor playground",
        "Kitchen", "Microwave", "Dishes and silverware", "Dining table",
        "Fire extinguisher", "First aid kit", "Security cameras",
        "Luggage dropoff", "Self check-in", "Keypad"
    ]
    # compute percentage (0–100)
    pct = (df_attr[amenity_cols].mean() * 100).round(1).reset_index()
    pct.columns = ["Amenity", "Percent"]
    
    # pivot to heatmap shape (single row)
    heatmap_df = pct.pivot(index=["Amenity"], columns=[], values="Percent")
    
    fig = px.imshow(
        heatmap_df,
        labels=dict(x="", y="Amenity", color="Percent of Listings"),
        x=["% of Listings"],
        y=heatmap_df.index,
        text_auto=True,
        aspect="auto",
        title="Amenity Presence Across Listings (%)",
        template="plotly_dark",
        color_continuous_scale="Viridis"
    )
    return fig

# 1.6 Price vs. Number of Bedrooms/Baths
def price_vs_size_boxplots():
    """
    Boxplots of Base fee grouped by Bedrooms and Baths count.
    """
    # melt into long form for two separate boxplots
    df_bed = df_attr[["Base fee", "Bedrooms"]].copy()
    df_bed["Type"] = "Bedrooms"
    df_bed = df_bed.rename(columns={"Bedrooms": "Count"})
    
    df_bath = df_attr[["Base fee", "Baths"]].copy()
    df_bath["Type"] = "Baths"
    df_bath = df_bath.rename(columns={"Baths": "Count"})
    
    df_long = pd.concat([df_bed, df_bath], axis=0)
    
    fig = px.box(
        df_long,
        x="Count",
        y="Base fee",
        color="Type",
        title="Base Fee by Number of Bedrooms and Baths",
        labels={"Count": "Count", "Base fee": "Base Fee (Nightly)", "Type": "Dimension"},
        template="plotly_dark"
    )
    fig.update_layout(boxmode="group")
    return fig

# 2.1 Price Availability Timeline
def price_availability_timeline():
    """
    For each calendar date, show percent of listings with a valid price (i.e. != -1 or NaN).
    """
    # count total listings
    total = len(df_ts)
    # compute per-date availability
    avail = (
        df_ts[non_empty_dates]
        .apply(lambda col: col.replace(-1, np.nan).notna().sum() / total * 100)
        .reset_index()
    )
    avail.columns = ["Date", "PctAvailable"]
    # parse dates
    avail["Date"] = pd.to_datetime(avail["Date"], dayfirst=True, format="%d/%m/%Y")
    avail = avail.sort_values("Date")
    
    fig = px.line(
        avail,
        x="Date",
        y="PctAvailable",
        title="Daily % of Listings with a Valid Price",
        labels={"PctAvailable": "% Available", "Date": "Date"},
        template="plotly_dark"
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(range=[0, 100])
    return fig

# 2.2 Missing-Data Gap Distribution
def gap_distribution_histogram():
    """
    Histogram of max_nan_gap across listings to show how many have big contiguous missing spans.
    """
    fig = px.histogram(
        df_ts,
        x="max_nan_gap",
        nbins=30,
        title="Distribution of Maximum Consecutive Missing-Price Gaps",
        labels={"max_nan_gap": "Max Consecutive Missing Days", "count": "Number of Listings"},
        template="plotly_dark"
    )
    return fig

# 2.3 Sample Price Trends (Raw)
def sample_price_trends(n_samples: int = 9, random_state: int = 42):
    """
    Draw n_samples small-multiples showing the raw daily price series
    (including -1/NaN gaps) for a handful of representative listings.
    """
    # Pick a few listings at random (or you could replace this with top-by-reviews etc.)
    sample_ids = (
        df_ts
        .sample(n=min(n_samples, len(df_ts)), random_state=random_state)["ID"]
        .tolist()
    )
    
    # Melt only those listings into long format
    df_long = (
        df_ts
        .loc[df_ts["ID"].isin(sample_ids), ["ID", "Name"] + non_empty_dates]
        .melt(
            id_vars=["ID", "Name"],
            value_vars=non_empty_dates,
            var_name="Date",
            value_name="Price"
        )
    )
    
    # Coerce -1 to NaN so gaps show up blank
    df_long["Price"] = df_long["Price"].replace(-1, np.nan).astype(float)
    
    # Parse and sort dates
    df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, format="%d/%m/%Y")
    df_long = df_long.sort_values(["ID", "Date"])
    
    # Plot small multiples
    fig = px.line(
        df_long,
        x="Date",
        y="Price",
        facet_col="ID",
        facet_col_wrap=3,
        color="ID",
        line_group="ID",
        title=f"Sample of {len(sample_ids)} Raw Price Time-Series",
        labels={"Price": "Price (USD/night)", "Date": "Date"},
        template="plotly_dark",
    )
    
    # Cosmetic: drop facet row titles, tighten spacing
    fig.update_layout(showlegend=False, height=300 * ((len(sample_ids) // 3) + 1))
    fig.for_each_annotation(lambda a: a.update(text=f"ID {a.text.split('=')[1]}"))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    
    return fig

# 3.1 Compare Raw vs. Interpolated for Sample
def compare_raw_vs_interpolated(n_samples: int = 6, random_state: int = 42):
    """
    Overlay raw (dashed) vs. interpolated (solid) daily price series
    for a small set of listings to visually validate the fill.
    """
    # pick sample IDs
    sample_ids = (
        df_ts
        .sample(n=min(n_samples, len(df_ts)), random_state=random_state)["ID"]
        .tolist()
    )

    # melt raw
    raw_long = (
        df_ts
        .loc[df_ts["ID"].isin(sample_ids), ["ID", "Name"] + non_empty_dates]
        .melt(id_vars=["ID", "Name"], value_vars=non_empty_dates,
              var_name="Date", value_name="RawPrice")
    )
    raw_long["RawPrice"] = raw_long["RawPrice"].replace(-1, np.nan).astype(float)

    # melt interpolated
    interp_long = (
        df_ts_interp
        .loc[df_ts_interp["ID"].isin(sample_ids), ["ID", "Name"] + non_empty_dates]
        .melt(id_vars=["ID", "Name"], value_vars=non_empty_dates,
              var_name="Date", value_name="InterpPrice")
    )
    interp_long["InterpPrice"] = interp_long["InterpPrice"].astype(float)

    # merge
    df_cmp = pd.merge(raw_long, interp_long, on=["ID", "Name", "Date"])
    df_cmp["Date"] = pd.to_datetime(df_cmp["Date"], dayfirst=True, format="%d/%m/%Y")
    df_cmp = df_cmp.sort_values(["ID", "Date"])

    # plot
    fig = px.line(
        df_cmp,
        x="Date",
        y="InterpPrice",
        color="ID",
        line_group="ID",
        facet_col="ID",
        facet_col_wrap=3,
        title="Raw vs. Interpolated Price (solid=interp, dashed=raw)",
        labels={"InterpPrice": "Price (USD/night)", "Date": "Date"},
        template="plotly_dark"
    )
    # add raw as dashed
    for sid in sample_ids:
        sub = df_cmp[df_cmp["ID"] == sid]
        fig.add_scatter(
            x=sub["Date"],
            y=sub["RawPrice"],
            mode="lines",
            line=dict(dash="dash"),
            name=f"{sid} raw",
            showlegend=False,
            row=(sample_ids.index(sid) // 3) + 1,
            col=(sample_ids.index(sid) % 3) + 1
        )

    fig.update_layout(showlegend=False,
                      height=300 * ((len(sample_ids) // 3) + 1))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    return fig

# 3.2 Weekly/Monthly Average Price Trends
def average_price_trend(freq: str = "W"):
    """
    Compute and plot the average daily price across all listings,
    aggregated by week ('W') or month ('M').
    """
    # melt interpolated table
    df_long = (
        df_ts_interp
        .melt(id_vars=["ID"], value_vars=non_empty_dates,
              var_name="Date", value_name="Price")
    )
    df_long["Price"] = df_long["Price"].astype(float)
    df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, format="%d/%m/%Y")

    # compute daily mean
    daily = df_long.groupby("Date")["Price"].mean().reset_index()

    # resample
    daily = daily.set_index("Date")
    if freq.upper() == "W":
        agg = daily.resample("W-MON")["Price"].mean().reset_index().sort_values("Date")
        title = "Weekly Average Price Trend"
    elif freq.upper() == "M":
        agg = daily.resample("M")["Price"].mean().reset_index().sort_values("Date")
        title = "Monthly Average Price Trend"
    else:
        raise ValueError("freq must be 'W' or 'M'")

    # plot
    fig = px.line(
        agg,
        x="Date",
        y="Price",
        title=title,
        labels={"Price": "Avg. Price (USD/night)", "Date": "Date"},
        template="plotly_dark"
    )
    return fig

# 3.3 Price Volatility Heatmap
def price_volatility_heatmap():
    """
    Build a heatmap of (price − mean_price)/mean_price for each listing/date.
    X = dates, Y = listing IDs.
    """
    # compute raw and interp price matrix
    mat = df_ts_interp.set_index("ID")[non_empty_dates].astype(float)
    # compute each row’s mean
    row_means = mat.mean(axis=1)
    # compute deviation matrix
    dev = (mat.sub(row_means, axis=0)).div(row_means, axis=0)
    # reorder index for readability (optional)
    dev = dev.sort_index()
    # prepare axes
    dates = pd.to_datetime(non_empty_dates, dayfirst=True, format="%d/%m/%Y")
    ids = dev.index.astype(str)
    # build heatmap
    fig = go.Figure(
        go.Heatmap(
            z=dev.values,
            x=dates,
            y=ids,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Deviation"),
            hovertemplate="Listing: %{y}<br>Date: %{x|%Y-%m-%d}<br>Dev: %{z:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        title="Price Volatility (Deviation from Listing Mean)",
        xaxis_title="Date",
        yaxis_title="Listing ID",
        template="plotly_dark",
        height=600
    )
    return fig

# 3.4 Distribution of Interpolation Changes
def interpolation_adjustment_histogram():
    """
    Histogram of (interpolated − raw) for only those cells that were imputed.
    """
    # raw price with NaNs
    raw = df_ts.set_index("ID")[non_empty_dates].replace(-1, np.nan).astype(float)
    interp = df_ts_interp.set_index("ID")[non_empty_dates].astype(float)
    # mask of imputed points: where raw is NaN but interp is not
    mask = raw.isna() & interp.notna()
    # compute diff where mask
    diffs = (interp - raw)[mask].stack().reset_index(name="Diff")
    # histogram
    fig = px.histogram(
        diffs,
        x="Diff",
        nbins=50,
        title="Distribution of Interpolation Adjustments (interp − raw)",
        labels={"Diff": "Price Difference (USD)"},
        template="plotly_dark"
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
            # Base fee distribution
            dbc.Col(
                dcc.Graph(id="base-fee-dist", figure=base_fee_distribution()),
                width=6
            ),
            # Cleaning vs Base fee
            dbc.Col(
                dcc.Graph(id="cleaning-vs-base", figure=cleaning_vs_base_fee()),
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            # Ratings summary
            dbc.Col(
                dcc.Graph(id="ratings-summary", figure=ratings_summary()), 
                width=6
            ),
            # Property type counts
            dbc.Col(
                dcc.Graph(id="property-type-counts", figure=property_type_counts()), 
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            # Amenity presence heatmap
            dbc.Col(
                dcc.Graph(id="amenity-heatmap", figure=amenity_presence_heatmap()), 
                width=6
            ),
            # Price vs Size boxplots
            dbc.Col(
                dcc.Graph(id="price-vs-size", figure=price_vs_size_boxplots()), 
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            # Price availability timeline
            dbc.Col(
                dcc.Graph(id="price-availability", figure=price_availability_timeline()), 
                width=6
            ),
            # Gap distribution histogram
            dbc.Col(
                dcc.Graph(id="gap-distribution",  figure=gap_distribution_histogram()),
                width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            # Sample price trends
            dbc.Col(
                dcc.Graph(id="sample-price-trends", figure=sample_price_trends()), 
                width=12
            ),
        ], className="mb-5"),
        dbc.Row([
            # Compare raw vs interpolated
            dbc.Col(
                dcc.Graph(id="compare-raw-interp", figure=compare_raw_vs_interpolated()), 
                width=12
            ),
        ], className="mb-5"),
        dbc.Row([
            # Weekly average price trend
            dbc.Col(
                dcc.Graph(id="weekly-price-trend", figure=average_price_trend("W")),
                width=12
            ),
        ], className="mb-5"),
        dbc.Row([
            # Price volatility heatmap
            dbc.Col(
                dcc.Graph(id="volatility-heatmap", figure=price_volatility_heatmap()), 
                width=12
            ),
        ], className="mb-5"),
        dbc.Row([
            # Interpolation adjustment histogram
            dbc.Col(
                dcc.Graph(id="interp-adjustment-hist", figure=interpolation_adjustment_histogram()), 
                width=12
            ),
        ], className="mb-5")
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