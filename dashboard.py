import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import plotly.express as px
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import pdist, squareform
import umap
import gudhi as gd
import plotly.io as pio

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
    pct = (df_attr.iloc[:, 21:].mean() * 100).round(1).sort_values(ascending=False)
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
        height=870,
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
        labels={"Count": "Count", "Base fee": "Base Fee (Nightly)", "Type": "Feature"},
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
            "Type": "Feature"
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
        color_discrete_sequence=[red],
    )
    fig.update_traces(line=dict(width=3), fill="tozeroy")
    fig.update_yaxes(range=[0, 100])
    return fig

# 2.2 Missing-Data Gap Table
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
            "color": "#dddddd",
            "fontSize": "12px",
            "textAlign": "center"
        },
        style_cell={
            "backgroundColor": "#111111",
            "color": "#dddddd",
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
        facet_col_spacing=0.1,
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
        category_orders={"Source": ["Raw", "Interpolated"], "ID": sample_ids},
        line_dash="Source",
        line_dash_map={"Raw": "solid", "Interpolated": "dash"},
        color_discrete_map={"Raw": red, "Interpolated": "#dddddd", },
        facet_col="ID", facet_col_wrap=2,
        facet_col_spacing=0.1,
        title="Raw vs. Interpolated Price Series",
        labels={"Price":"Price (USD/night)", "Date":"Date"},
        template="plotly_dark"
    )
    dashed_traces = [t for t in fig.data if t.line.dash == "dash"] 
    solid_traces  = [t for t in fig.data if t.line.dash == "solid"]
    fig.data = tuple(dashed_traces + solid_traces)
    fig.update_traces(
        selector=dict(name="Raw"),
        line=dict(width=2),
    )
    fig.update_traces(
        selector=dict(name="Interpolated"),
        line=dict(width=2, dash="5px,2px")
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


# 3.4 Map of Base Fee Distribution
def map_base_fee_density():
    """
    Density-map of listing locations, weighted by Base fee.
    """
    fig = px.density_map(
        df_attr,
        lat="latitude",
        lon="longitude",
        z="Base fee",
        radius=20,
        hover_name="Name",
        hover_data=["Base fee", "Property type"],
        color_continuous_scale="YlOrRd",
        map_style="carto-darkmatter",
        zoom=11,
        title="Cartagena Base Fee Density"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        paper_bgcolor="#111111",   
        plot_bgcolor="#111111",   
        font=dict(color="white") 
    )
    return fig



"""_Cluster_
-----------------------------------------------------------------------------------------------------------
This section contains the clustering functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""

# 4.1.1 PCA + K-Means embedding scatter
def attribute_pca_kmeans(n_clusters=4):
    """
    2D PCA embedding of your listing attributes, colored by K-Means cluster.
    """
    # Build feature matrix
    feat = df_attr.select_dtypes(include=["number", "bool"]).copy()
    feat = feat.astype({c: "int" for c in feat.select_dtypes("bool").columns})
    feat.drop(columns=["latitude", "longitude"], errors='ignore', inplace=True)
    X = StandardScaler().fit_transform(feat)
    
    # PCA and K-Means
    emb = PCA(n_components=2, random_state=0).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=69).fit(emb)
    labels = km.labels_.astype(str)
    
    # Create figure
    plot_df = pd.DataFrame({
        "PC1": emb[:,0],
        "PC2": emb[:,1],
        "cluster": labels,
        "Name": df_attr["Name"]
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="cluster",
        hover_name="Name",
        title=f"PCA (2D) + K-Means (k={n_clusters}) on Listing Attributes",
        template="plotly_dark",
        color_discrete_sequence=["#d79c9c", red, "#c71a37", "#ff657f"] 
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig

# 4.1.2 3D PCA + K-Means embedding scatter
def attribute_pca_kmeans_3d(n_clusters=4):
    """
    3D PCA embedding of your listing attributes, colored by K-Means cluster.
    """
    # Build feature matrix
    feat = df_attr.select_dtypes(include=["number", "bool"]).copy()
    feat = feat.astype({c: "int" for c in feat.select_dtypes("bool").columns})
    feat.drop(columns=["latitude", "longitude"], errors='ignore', inplace=True)
    X = StandardScaler().fit_transform(feat)

    # PCA and K-Means
    emb = PCA(n_components=3, random_state=0).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(emb)
    labels = km.labels_.astype(str)

    # Create figure
    plot_df = pd.DataFrame({
        "PC1": emb[:, 0],
        "PC2": emb[:, 1],
        "PC3": emb[:, 2],
        "cluster": labels,
        "Name": df_attr["Name"]
    })
    fig = px.scatter_3d(
        plot_df,
        x="PC1", y="PC2", z="PC3",
        color="cluster",
        hover_name="Name",
        title=f"3D PCA + K-Means (k={n_clusters}) on Listing Attributes",
        template="plotly_dark",
        color_discrete_sequence=["#d79c9c", red, "#c71a37", "#ff657f"]
    )
    fig.update_traces(marker=dict(size=6.8, opacity=0.8))
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        )
    )
    return fig

# 4.2.1 Dendrogram of Time-Series
def timeseries_dendrogram():
    """
    Agglomerative clustering dendrogram (Euclidean) of each listing's 
    interpolated price series, after dropping any series with NaNs.
    """
    # Build the matrix (listings × days)
    mat = df_ts_interp.set_index("ID")[dates].astype(float).values
    ids = df_ts_interp["ID"].astype(str).tolist()

    # Create the figure
    fig = ff.create_dendrogram(
        mat,
        orientation="left",
        labels=ids,
        colorscale=px.colors.sequential.amp[::-1]
    )
    fig.update_layout(
        template="plotly_dark",
        width=800,
        height=1350,
        title="Hierarchical Clustering Dendrogram of Price Time Series",
        xaxis_title="Distance",
        yaxis_title="Listing ID"
    )
    return fig

# 4.2.2 Cluster-Centroid Time Series
def timeseries_cluster_centroids(n_clusters=4):
    """
    For each cluster (Agglomerative, Euclidean), plot the average daily price curve.
    """
    # Build the matrix (listings × days) and perform clustering
    mat = df_ts_interp.set_index("ID")[dates].astype(float).values
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward").fit_predict(mat)
    
    # Create a DataFrame with the cluster labels
    df_long = (
        df_ts_interp.assign(cluster=labels)
        .melt(
            id_vars=["ID", "cluster"],
            value_vars=dates,
            var_name="Date",
            value_name="Price"
        )
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"], dayfirst=True, format="%d/%m/%Y")
    avg = (
        df_long
        .groupby(["cluster", "Date"])["Price"]
        .mean()
        .reset_index()
    )

    # Plot the average price time series for each cluster
    avg["cluster"] = avg["cluster"].astype(str)
    fig = px.line(
        avg,
        x="Date", y="Price",
        color="cluster",
        title=f"Cluster Centroid Time Series (k={n_clusters})",
        labels={"cluster":"Cluster", "Price":"Avg. Price (USD/night)"},
        template="plotly_dark",
        color_discrete_sequence=["#d79c9c", red, "#c71a37", "#ff657f"] 
    )
    fig.update_traces(line=dict(width=2))
    return fig

# 4.3.0 Compute Volatility Features
def compute_volatility_features(spike_thresh=0.1):
    """
    For each listing (row), compute:
      - std_dev:  standard deviation of (price - mean)/mean
      - max_dev:  maximum absolute deviation
      - spike_freq: fraction of days with |dev| > spike_thresh
    Returns a DataFrame indexed by ID.
    """
    # Build deviation matrix
    mat = df_ts_interp.set_index("ID")[dates].astype(float)
    row_means = mat.mean(axis=1)
    dev = mat.sub(row_means, axis=0).div(row_means, axis=0)

    # Extract features
    std_dev    = dev.std(axis=1)
    max_dev    = dev.abs().max(axis=1)
    spike_freq = (dev.abs() > spike_thresh).sum(axis=1) / dev.shape[1]
    
    feats = pd.DataFrame({
        "std_dev":    std_dev,
        "max_dev":    max_dev,
        "spike_freq": spike_freq
    }).round(3)
    return feats

# 4.3.1 K-Means Clustering on Volatility Features
def cluster_volatility(feats, k=4):
    """
    Fit KMeans on the features DataFrame and return
    a new DataFrame with a 'cluster' column (as string).
    """
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(feats)
    df = feats.copy()
    df["cluster"] = labels.astype(str)
    return df

# 4.3.2 Scatter-Matrix of Features Colored by Cluster
def volatility_scatter_matrix(df_vol, k=4):
    """
    Pair-plot of (std_dev, max_dev, spike_freq) colored by KMeans cluster.
    """
    df = df_vol.reset_index().rename(columns={"index": "ID"})
    fig = px.scatter_matrix(
        df,
        dimensions=["std_dev", "max_dev", "spike_freq"],
        color="cluster",
        title=f"Volatility Features Scatter-Matrix (k={k})",
        labels={
            "std_dev": "Std Dev",
            "max_dev": "Max Dev",
            "spike_freq": "Spike Freq",
            "cluster": "Cluster"
        },
        template="plotly_dark",
        color_discrete_sequence=["#d79c9c", red, "#c71a37", "#ff657f"] 
    )
    fig.update_traces(diagonal_visible=False)
    return fig

# 4.3.3 Boxplots of Each Feature by Cluster
def volatility_boxplots(df_vol, k=4):
    """
    Boxplots of each volatility metric, grouped by cluster.
    """
    df_long = (
        df_vol
        .reset_index()
        .melt(id_vars=["ID", "cluster"],
              value_vars=["std_dev", "max_dev", "spike_freq"],
              var_name="Metric",
              value_name="Value")
    )
    fig = px.box(
        df_long,
        x="Metric",
        y="Value",
        color="cluster",
        title=f"Volatility Metrics by Cluster (k={k})",
        labels={"Value": "Value", "Metric": "Volatility Metric", "cluster": "Cluster"},
        template="plotly_dark",
        color_discrete_sequence=["#d79c9c", red, "#c71a37", "#ff657f"]
    )
    fig.update_layout(boxmode="group")
    return fig

# 4.4.0 Extract the binary amenity matrix and cluster
def cluster_by_amenities(k=4):
    """
    Fit KMeans on the amenity matrix, returning the amenity matrix
    and a Series of cluster labels (as strings).
    """
    to_keep = [
        "Keypad", "Lock on door", "Smoke detector", "Security cameras", "AC", "Heating", 
        "Patio or balcony", "Stove", "Elevator", "Refrigerator", "Kitchen", "Wifi", 
        "TV", "Jacuzzi", "Carport", "Hot water", 
    ]
    am = df_attr[to_keep].copy().astype(int).set_index(df_attr["ID"])
    km = KMeans(n_clusters=k, random_state=0, init="k-means++")
    labels = pd.Series(km.fit_predict(am), index=am.index).astype(str)
    return am, labels

# 4.4.1 Stacked‐Bar of Amenity Proportions by Cluster
def amenity_cluster_bars(am, labels):
    """
    For each cluster, compute % of listings that have each amenity, then
    draw a stacked bar chart (amenities on the legend).
    """
    df = am.copy()
    df["cluster"] = labels
    prop = df.groupby("cluster").mean().reset_index().melt(
        id_vars="cluster", var_name="amenity", value_name="pct"
    ) 
    hot_colors = px.colors.sample_colorscale("amp_r", np.linspace(0, 0.7, prop["amenity"].nunique()))
    #random.shuffle(hot_colors)
    fig = px.bar(
        prop,
        x="cluster",
        y="pct",
        color="amenity",
        title="Amenity % by Cluster",
        labels={"cluster":"Cluster", "pct":"% Listings", "amenity":"Amenity"},
        template="plotly_dark",
        color_discrete_sequence=hot_colors
        
    )
    fig.update_layout(barmode="stack", xaxis_title="Cluster")
    return fig


"""_TDA_
-----------------------------------------------------------------------------------------------------------
This section contains the topological data analysis functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""

# 5.1 Build UMAP and Distance Matrix
def build_umap_and_distances():
    # Drop some columns that are not needed
    df_temp = df_attr.copy()
    """
    df_temp=df_temp[[
    'Name', 'Host', 'Base fee', 'Cleaning fee', 'URL', 'ID', 'latitude',
    'longitude', 'Property type', 'Person capacity', 'accuracy_rating',
    'checking_rating', 'cleanliness_rating', 'communication_rating',
    'location_rating', 'value_rating', 'satisfaction_rating', 'Reviews',
    'Bedrooms', 'Beds', 'Baths', 'City skyline view', 'Beach view',
    'Sea/Lake view', 'Hot water', 'Jacuzzi', 'Shared pool', 'Shared gym',
    'Patio or balcony', 'Outdoor furniture', 'Outdoor playground',
    'Elevator', 'Carport', 'Dedicated workspace', 'AC', 'Heating', 'TV',
    'Cable TV', 'Wifi', 'Laundry service', 'Kitchen', 'Dining table',
    'Microwave', 'Dishes and silverware', 'Refrigerator', 'Stove', 'Keypad',
    'Washer', 'Pets allowed', 'Crib', 'Security cameras', 'Lock on door']]
    to_keep = [
        "Keypad", "Lock on door", "Smoke detector", "Security cameras", "AC", "Heating", 
        "Patio or balcony", "Stove", "Elevator", "Refrigerator", "Kitchen", "Wifi", 
        "TV", "Jacuzzi", "Carport", "Hot water", 
    ]
    df_temp = df_temp[["ID", "Base fee"] + to_keep]
    """
    df_temp = df_temp.iloc[:, :21]
    

    # Melt time series data to long format
    df_prices = (
        df_ts_interp.copy()
        .melt(id_vars="ID", value_vars=dates, var_name="Date", value_name="Value")
        .assign(Date=lambda d: pd.to_datetime(d["Date"], dayfirst=True))
    )

    # Summarize per‐listing log‐price mean, std, and trend
    def summarize(group):
        #y = np.log1p(group["Value"].replace(0, np.nan))  # Avoid log(0)
        y = np.log1p(group["Value"])
        #y = group["Value"]
        days = (group["Date"] - group["Date"].min()).dt.days.values.reshape(-1,1)
        lr = LinearRegression().fit(days, y) if len(np.unique(days))>1 else None
        return pd.Series({
            "price_mean": y.mean(),
            "price_std":  y.std(),
            "price_trend": lr.coef_[0] if lr else 0.0
        })
    df_price_summary = (
        df_prices
        .groupby("ID", group_keys=False)
        .apply(summarize)
    )
    df_merged = df_attr.merge(df_price_summary, on="ID")

    # Filter out near‐constant / low‐variance features
    selector = VarianceThreshold(threshold=0.1)
    X = selector.fit_transform(df_merged.select_dtypes("number"))
    to_keep = df_merged.select_dtypes("number").columns[selector.get_support()]

    # Drop highly correlated (>0.9)
    df_reduced = pd.DataFrame(X, columns=to_keep)
    corr = df_reduced.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)) # type: ignore
    to_drop = [c for c in upper.columns if (upper[c] > 0.9).any()]
    df_space = df_reduced.drop(columns=to_drop)

    # Scale + UMAP embedding
    X_scaled = StandardScaler().fit_transform(df_space)
    umap_proj = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=69).fit_transform(X_scaled)

    # Build distance matrix and UMAP space DataFrame
    df_space = df_merged.loc[df_space.index, ['ID','Base fee']].reset_index(drop=True)
    df_space[['UMAP1','UMAP2','UMAP3']] = umap_proj
    dist_matrix = squareform(pdist(df_space.drop(columns=["ID", 'Base fee']).values, metric="euclidean"))
    df_dist = pd.DataFrame(dist_matrix, index=df_space['ID'], columns=df_space['ID'])
    
    return df_space, df_dist, dist_matrix, df_prices

# 5.2 UMAP Space Scatter Plot
def umap_space(df_space):
    """
    3D plot of the UMAP embedded space.
    """
    fig = px.scatter_3d(
        df_space,
        x="UMAP1", y="UMAP2", z="UMAP3",
        color='Base fee',
        hover_name="ID",
        color_continuous_scale="amp",
        opacity=0.85,
        title="Embedded UMAP Space",
        template="plotly_dark"
    )
    return fig

# 5.3 Vietoris-Rips Complex on UMAP Space
def vietoris_rips_3d(df_space, simplex_tree):
    """
    3D visualization of the Vietoris-Rips complex on the UMAP embedding.
    """
    coords = df_space[['UMAP1', 'UMAP2', 'UMAP3']].values
    ids = df_space['ID'].astype(str).tolist()

    # Build all 1-simplex edges
    edge_traces = []
    for simplex, _ in simplex_tree.get_skeleton(1):
        if len(simplex) == 2:
            i, j = simplex
            x0, y0, z0 = coords[i]
            x1, y1, z1 = coords[j]
            edge_traces.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(color='white', width=1),
                hoverinfo='none',
                showlegend=False
            ))

    # Point cloud trace
    point_trace = go.Scatter3d(
        x=df_space['UMAP1'],
        y=df_space['UMAP2'],
        z=df_space['UMAP3'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_space['Base fee'],
            opacity=0.85,
            colorscale='amp',
            cmin=40,
            cmax=120,
        ),
        text=ids,
        name='Points'
    )

    # Create the figure
    fig = go.Figure(data=[point_trace] + edge_traces)
    fig.update_layout(
        title='Vietoris-Rips Complex on UMAP Embedding',
        template='plotly_dark',
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3'
        ),
    )
    return fig

# 5.4 Projections of the Vietoris-Rips Complex
def rips_projections(df_space, simplex_tree):
    """
    Show XY, XZ and YZ projections of the Vietoris-Rips complex on the UMAP embedding.
    """
    coords = df_space[["UMAP1", "UMAP2", "UMAP3"]].values
    ids = df_space["ID"].astype(str).tolist()

    # Prepare subplot grid and scatter plot parameters
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("UMAP1 vs UMAP2", "UMAP1 vs UMAP3", "UMAP2 vs UMAP3"),
        horizontal_spacing=0.05
    )
    scatter_kwargs = dict(
        mode="markers",
        xaxis=None, yaxis=None,
        marker=dict(size=6, color=df_space["Base fee"], showscale=True, 
                    colorbar=dict(title="Base fee",  outlinewidth=0, outlinecolor="rgba(0,0,0,0)"), 
                    colorscale="amp", cmin=40, cmax=125),
        hovertext=ids,
        hoverinfo="text",
        showlegend=False 
    )

    # The three projection pairs
    dims = [(0,1), (0,2), (1,2)]
    for col, (i, j) in enumerate(dims, start=1):
        # Add the edges for the current projection
        for simplex, _ in simplex_tree.get_skeleton(1):
            if len(simplex) == 2:
                a, b = simplex
                fig.add_trace(
                    go.Scatter(
                        x=[coords[a,i], coords[b,i], None],
                        y=[coords[a,j], coords[b,j], None],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        hoverinfo="none",
                        showlegend=False
                    ),
                    row=1, col=col
                )
        # Add the scatter points for the current projection
        fig.add_trace(
            go.Scatter(
                x=coords[:,i],
                y=coords[:,j],
                **scatter_kwargs
            ),
            row=1, col=col
        )
    fig.update_layout(
        title="Vietoris-Rips Complex Projections",
        template="plotly_dark",
        height=500, width=1200,
    )
    for idx in range(1, 4):
        fig.update_xaxes(matches=None, row=1, col=idx)
        fig.update_yaxes(matches=None, row=1, col=idx)

    return fig

# 5.5 Persistence Diagram and Barcode
def persistence_and_barcode(dist_matrix, max_edge=1.0):
    """
    Compute Vietoris-Rips persistence and plot the persistence diagram and barcode.
    """
    # Build Rips complex and compute persistence up to dimension 2
    rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge) #type: ignore
    st = rips.create_simplex_tree(max_dimension=4)
    st.compute_persistence()
    pairs = st.persistence()

    # Group (birth, death) by homology dimension
    dims = {}
    for dim, (b, d) in pairs:
        if d == float('inf'):
            d = max_edge
        dims.setdefault(dim, []).append((b, d))

    colors = {0: red, 1: "#ff657f", 2: "#fac9c9"}
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Persistence Diagram", "Barcode"),
        horizontal_spacing=0.1
    )

    # Persistence diagram
    for dim, bd in dims.items():
        births, deaths = zip(*bd)
        fig.add_trace(
            go.Scatter(
                x=births, y=deaths, mode="markers",
                marker=dict(color=colors.get(dim, "gray"), size=9, opacity=0.85),
                name=f"H{dim}", legendgroup=str(dim)
            ),
            row=1, col=1
        )
    # Diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, max_edge], y=[0, max_edge], mode="lines",
            line=dict(color="white", dash="dash"), showlegend=False
        ),
        row=1, col=1
    )

    # Barcode
    y = 0
    for dim, bd in dims.items():
        for b, d in bd:
            fig.add_trace(
                go.Scatter(
                    x=[b, d], y=[y, y], mode="lines",
                    line=dict(color=colors.get(dim, "gray"), width=4),
                    showlegend=False
                ),
                row=1, col=2
            )
            y += 1

    fig.update_xaxes(title_text="Birth", row=1, col=1)
    fig.update_yaxes(title_text="Death", row=1, col=1)
    fig.update_xaxes(title_text="Filtration", row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_layout(
        title="Persistent Homology: Diagram & Barcode",
        template="plotly_dark",
        width=1000, height=800,
    )
    return fig

# 5.6 Betti Number Evolution
def betti_evolution(distance_matrix, edge_lengths=np.linspace(0.1, 1.1, 100), max_dim=2):
    """
    Compute and plot Betti numbers (H₀, H₁, … H_max_dim) 
    for a Vietoris-Rips complex as the max-edge threshold varies.
    """
    # For each threshold, build Rips complex and read off its Betti numbers
    records = []
    for eps in edge_lengths:
        st = (
            gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=eps) #type: ignore
              .create_simplex_tree(max_dimension=max_dim + 1)
        )
        st.compute_persistence()
        bettis = st.betti_numbers()
        # pad with zeros if some dimensions are missing
        bettis += [0] * (max_dim + 1 - len(bettis))
        records.append((eps, *bettis[: max_dim + 1]))
    cols = ["epsilon"] + [f"H{d}" for d in range(max_dim + 1)]
    df = pd.DataFrame(records, columns=cols)
    df_long = df.melt(id_vars="epsilon", var_name="Homology dim", value_name="Count")

    # Create the line plot
    fig = px.line(
        df_long,
        x="epsilon",
        y="Count",
        color="Homology dim",
        title="Evolution of Betti Numbers vs. Rips Scale",
        labels={"epsilon": "Max edge length", "Count": "Betti count"},
        color_discrete_map={"H0": red, "H1": "#ff657f", "H2": "#d79c9c"},
        template="plotly_dark"
    )
    fig.update_layout(
        font=dict(color="white"),
        legend_title_text="homology group",
        height=350
    )
    return fig

# 5.7 Sliding-Window Persistence Diagrams
def sliding_window_persistence_multi(listing_ids, embedding_dim=3, max_edge_length=1.4):
    """
    Compute sliding-window persistence diagrams for up to 9 listing IDs
    """
    n = len(listing_ids)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=listing_ids,
        horizontal_spacing=0.05, vertical_spacing=0.1
    )
    color_map = {0: red, 1: "#ff657f", 2: "#fac9c9"}
    seen_dims = set()
    for idx, lid in enumerate(listing_ids):
        # Calculate row and column for the subplot
        r = idx//cols + 1
        c = idx%cols + 1    
        raw = df_ts_interp.loc[df_ts_interp["ID"] == lid, dates] \
                .values.flatten().astype(float)
        if len(raw) < embedding_dim:
            continue
        
        # Build the sliding window persistence diagram
        series = StandardScaler().fit_transform(raw.reshape(-1,1)).flatten()
        N = len(series) - embedding_dim + 1
        cloud = np.stack([series[i:i+embedding_dim] for i in range(N)])
        rips = gd.RipsComplex(points=cloud, max_edge_length=max_edge_length) # type: ignore
        st = rips.create_simplex_tree(max_dimension=3)
        st.compute_persistence()

        # Collect births/deaths by dimension
        dims = {}
        for d,(b,e) in st.persistence():
            e = e if e!=float("inf") else np.nanmax(series)
            dims.setdefault(d,[]).append((b,e))

        # Create figure
        for d,pts in dims.items():
            births, deaths = zip(*pts)
            show = (d not in seen_dims)
            fig.add_trace(
                go.Scatter(
                    x=births, y=deaths, mode="markers",
                    marker=dict(color=color_map.get(d,"gray"), size=6),
                    name=f"H{d}", legendgroup=f"dim{d}",
                    showlegend=show
                ),
                row=r, col=c
            )
            if show:
                seen_dims.add(d)
        mn, mx = 0, np.nanmax(series)
        fig.add_trace(
            go.Scatter(
                x=[mn,mx], y=[mn,mx], mode="lines",
                line=dict(color="white", dash="dash"),
                showlegend=False
            ),
            row=r, col=c
        )
        fig.update_xaxes(title_text="Birth", row=r, col=c)
        fig.update_yaxes(title_text="Death", row=r, col=c)

    fig.update_layout(
        title=f"Sliding‐Window Persistence (embed={embedding_dim}, edge≤{max_edge_length})",
        template="plotly_dark",
        height=300*rows, width=300*cols
    )
    return fig

# 5.8 Bipersistence Heatmap
def bipersistence_heatmap(df_summary, n_steps=40):
    """
    Approximate a 2-parameter persistence surface β₀(p, v)
    """
    # compute β₀(p, v) as the count of listings with price_mean ≤ p and price_std ≤ v
    P = df_summary["price_mean"].values
    V = df_summary["price_std"].values
    p_grid = np.linspace(P.min(), P.max(), n_steps)
    v_grid = np.linspace(V.min(), V.max(), n_steps)
    comp_counts = np.zeros((len(v_grid), len(p_grid)), dtype=int)
    for i, v_thr in enumerate(v_grid):
        for j, p_thr in enumerate(p_grid):
            mask = (P <= p_thr) & (V <= v_thr)
            comp_counts[i, j] = mask.sum()

    # Create a heatmap
    df = pd.DataFrame(
        comp_counts, 
        index=np.round(v_grid,3), 
        columns=np.round(p_grid,3)
    )
    fig = px.imshow(
        df,
        labels=dict(x=f"price_mean ≤ p", y=f"price_std ≤ v", color="β₀ count"),
        x=df.columns, 
        y=df.index,
        title=f"Approximate β₀(p,v) — Price vs Volatility",
        aspect="auto",
        template="plotly_dark",
        color_continuous_scale="amp_r"
    )
    fig.update_xaxes(side="bottom")
    return fig



"""_Classification_
-----------------------------------------------------------------------------------------------------------
This section contains the classification functions for the dashboard.
-----------------------------------------------------------------------------------------------------------
"""
# 6.1 Load Precomputed Precision–Recall Curves
def price_tier_pr_curves(fig):
    """
    Load the precomputed Precision–Recall curve JSON and return the corresponding figure.
    """
    fig.update_layout(
        title="Precision–Recall Curves by Class",
        template="plotly_dark",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=420
    )
    return fig




"""_Dash_
-----------------------------------------------------------------------------------------------------------
This section contains the functions for the Dash web application.
-----------------------------------------------------------------------------------------------------------
"""

eda_tab = dbc.Tab(
    label="Exploratory Data Analysis",
    children=[
        html.H4(
            "Exploratory Data Analysis (EDA)",
            style={"text-align": "center", "margin-top": "30px", "color": "white", 
                   "font-weight": "bold", "font-size": "28px"}
        ),
        html.P(
            """
            In this section we load our three normalized tables (Attributes, TimeSeriesRaw, TimeSeriesInterpolated) 
            from our SQLite store (populated via our Airbnb web-scraper). We extracted structured data from Airbnb 
            room pages—including pricing, coordinates, amenities, and ratings for listings in Cartagena. We then 
            explore nightly base fees, cleaning charges, ratings, property types, amenities and size effects, as 
            well as the completeness and interpolation of our time-series price data.
            """,
            style={"text-align": "center", "margin-bottom": "40px", "color": "#bbbbbb",
                "max-width": "1300px", "margin-left": "auto", "margin-right": "auto"}
        ),
        html.Hr(style={"margin-bottom": "30px"}),

        dbc.Row([
            dbc.Col(
                dcc.Graph(id="base-fee-dist", figure=base_fee_distribution()), width=6
            ),
            dbc.Col(
                dcc.Graph(id="cleaning-vs-base", figure=cleaning_vs_base_fee()), width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="ratings-summary", figure=ratings_summary()), width=6
            ),
            dbc.Col(
                dcc.Graph(id="property-type-counts", figure=property_type_counts()), width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col( 
                dcc.Graph(id="amenity-lollipop", figure=amenity_presence_lollipop()), width=6
            ),
            dbc.Col([
                dcc.Graph(id="price-vs-size", figure=price_vs_size_boxplots()), 
                dcc.Graph(id="price-vs-capacity-beds", figure=price_vs_capacity_beds_boxplots()),
            ], width=6),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="price-availability", figure=price_availability_timeline()), width=6
            ),
            dbc.Col(
                html.Div([
                    html.H5("Top 10 Listings by Missing-Price Gap",
                            style={"color": "white", "marginBottom": "20px"}),
                    top_gap_table()
                ]), width=6,
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="sample-price-trends", figure=sample_price_trends()), width=5
            ),
            dbc.Col(
                dcc.Graph(id="compare-raw-interp", figure=compare_raw_vs_interpolated()), width=7
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="weekly-price-trend", figure=average_price_trend("W")), width=5
            ),
            dbc.Col(
                dcc.Graph(id="volatility-heatmap", figure=price_volatility_heatmap()), width=7
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="map-base-fee", figure=map_base_fee_density()), width=11
            ),
        ], className="mb-5 justify-content-center"),
    ]
)


df_ts_interp = df_ts_interp.dropna(subset=dates, how="any").reset_index(drop=True)
feats = compute_volatility_features()
df_vol = cluster_volatility(feats)
am, am_labels = cluster_by_amenities()

clustering_tab = dbc.Tab(
    label="Clustering",
    children=[
        html.H4(
            "Clustering",
            style={"text-align": "center", "margin-top": "30px", "color": "white", 
                   "font-weight": "bold", "font-size": "28px"}
        ),
        html.P(
            """
            In this section we perform clustering on our Airbnb listings using various features.
            We explore K-Means clustering on listing attributes, PCA embeddings,
            hierarchical clustering of time-series price data, and clustering based on volatility features.
            We also analyze amenity presence and clustering, and visualize the results using scatter plots,
            dendrograms, and bar charts.
            """,
            style={"text-align": "center", "margin-bottom": "40px", "color": "#bbbbbb",
                "max-width": "1300px", "margin-left": "auto", "margin-right": "auto"}
        ),
        html.Hr(style={"margin-bottom": "30px"}),


        dbc.Row([
            dbc.Col(
                dcc.Graph(id="attribute-pca-kmeans", figure=attribute_pca_kmeans()), width=6
            ),
            dbc.Col(
                dcc.Graph(id="attribute-pca-kmeans-3d", figure=attribute_pca_kmeans_3d()), width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="timeseries-dendogram", figure=timeseries_dendrogram()), width=6
            ),
            dbc.Col([
                dcc.Graph(id="timeseries-cluster-centroids", figure=timeseries_cluster_centroids()), 
                dcc.Graph(id="volatility-scatter-matrix", figure=volatility_scatter_matrix(df_vol)), 
                dcc.Graph(id="volatilily-boxplots", figure=volatility_boxplots(df_vol)),
            ], width=6),
        ], className="mb-5"), 
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="amenity_cluster", figure=amenity_cluster_bars(am, am_labels)), width=10
            ),
        ], className="mb-5 justify-content-center")
    ]
)


df_space, df_dist, dist_matrix, df_prices = build_umap_and_distances()
rips_complex = gd.RipsComplex(distance_matrix=df_dist.values, max_edge_length=0.6) #type: ignore
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
main_ids = [
    "865417719613815681","1364376551860961934","1316158964673335895",
    "1315344706047362032","1282890977555839180","1038158992459515932",
    "846281101302571252","897405006229683800","1315344706047362032"
]
price_summary = df_prices.groupby("ID").apply(
    lambda g: pd.Series({
        "price_mean": g["Value"].mean(),
        "price_std":  g["Value"].std()
    })
)
dfa = price_summary.reset_index().merge(compute_volatility_features(), left_on="ID", right_index=True)

tda_tab = dbc.Tab(
    label="Topological Data Analysis",
    children=[
        html.H4(
            "Topological Data Analysis (TDA)",
            style={"text-align": "center", "margin-top": "30px", "color": "white", 
                   "font-weight": "bold", "font-size": "28px"}
        ),
        html.P(
            """
            In this section we perform topological data analysis (TDA) on the UMAP embedding of our Airbnb listings.
            We visualize the Vietoris-Rips complex, compute persistent homology, and analyze Betti numbers.
            We explore the UMAP space, projections of the Vietoris-Rips complex, and the evolution of Betti numbers.
            We also visualize the persistence diagram and barcode to understand the topological features of the data.
            """,
            style={"text-align": "center", "margin-bottom": "40px", "color": "#bbbbbb",
                "max-width": "1300px", "margin-left": "auto", "margin-right": "auto"}
        ),
        html.Hr(style={"margin-bottom": "30px"}),


        dbc.Row([
            dbc.Col(
                dcc.Graph(id="umap-space", figure=umap_space(df_space)), width=6
            ),
            dbc.Col(
                dcc.Graph(id="vietoris-rips-3d", figure=vietoris_rips_3d(df_space, simplex_tree)), width=6
            ),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="rips-projections", figure=rips_projections(df_space, simplex_tree)), width=10
            ),
        ], className="mb-5 justify-content-center"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="persistence-and-barcode", figure=persistence_and_barcode(dist_matrix)), width=8
            ),
            dbc.Col([
                dcc.Graph(id="betti-evolution", figure=betti_evolution(dist_matrix)), 
                dcc.Graph(id="bipersistence-heatmap", figure=bipersistence_heatmap(dfa)),
            ], width=4),
        ], className="mb-5"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="persistence-multi", figure=sliding_window_persistence_multi(main_ids)), width=7
            ),
        ], className="mb-5 justify-content-center"),
        dbc.Row(
            [
            dbc.Col(
                html.Div(
                    html.Img(
                        src="/assets/rips_complex_animation.gif",
                        style={"width": "85%", "height": "auto"}
                    ), style={"textAlign": "center"}   
                ),width=10, className="mx-auto" 
            ),
        ], className="mb-5")
    ]
)


m1_metrics = pd.read_csv("models/model1/price_tier_metrics.csv")
m1_cm = pd.read_csv("models/model1/price_tier_confusion_matrix.csv", index_col=0)
m1_pr = pio.read_json("models/model1/price_tier_pr_curves.json")

classification_tab = dbc.Tab(
    label="Classification",
    children=[
        html.H4(
            "Classification",
            style={"text-align": "center", "margin-top": "30px", "color": "white", 
                   "font-weight": "bold", "font-size": "28px"}
        ),
        html.P(
            """
            In this section we perform classification on our Airbnb listings using various features.
            """,
            style={"text-align": "center", "margin-bottom": "40px", "color": "#bbbbbb",
                "max-width": "1300px", "margin-left": "auto", "margin-right": "auto"}
        ),
        html.Hr(style={"margin-bottom": "30px"}),
        
        # Header for Model 1
        html.H5("XGBoost Classifier on Static, Temporal, Topological & Amenity Features", 
                style={'text-align': 'center', 'margin-top': '20px', 'color': 'white',
                    'margin-bottom': '10px'}
        ),
        html.P(
            "This XGBoost model classifies Airbnb listings into Low, Mid, or High price tiers using numerical attributes, "
            "interpolated time series, UMAP+TDA summaries, and binary amenity indicators. The model was optimized through "
            "GridSearchCV with 5-fold cross-validation.",
            style={"text-align": "center", "color": "#bbbbbb", "max-width": "1000px",
                "margin-left": "auto", "margin-right": "auto", "margin-bottom": "50px"}
        ),

        # Row for Metrics, Confusion Matrix, and Precision-Recall Curve
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("Metrics Summary", style={'text-align': 'center', 'color': 'white', 'fontsize': '16px'}),
                    dash_table.DataTable(
                        id='m1-metrics-table',
                        columns=[{'name': col, 'id': col} for col in m1_metrics.columns],
                        data=m1_metrics.to_dict('records'), # type: ignore
                        style_cell={'textAlign': 'center', 'backgroundColor': '#111111', 'color': 'dddddd'},
                        style_header={'backgroundColor': '#222222', 'fontWeight': 'bold', 'color': '#dddddd'},
                        style_as_list_view=True
                    )
                ], style={"margin-bottom": "30px"}),
                html.Div([
                    html.H6("Confusion Matrix", style={'text-align': 'center', 'color': 'white', 'fontsize': '25px'}),
                    dash_table.DataTable(
                        id='m1-confusion-matrix',
                        columns=[{'name': f'Pred {cls}', 'id': cls} for cls in m1_cm.columns],
                        data=m1_cm.reset_index().rename(columns={"index": "True"}).to_dict('records'), # type: ignore
                        style_cell={'textAlign': 'center', 'backgroundColor': '#111', 'color': 'white'},
                        style_header={'backgroundColor': '#222', 'fontWeight': 'bold', 'color': 'white'},
                        style_as_list_view=True
                    )
                ]), 
            ], width=5), 
            dbc.Col(dcc.Graph(
                id='m1-pr-curve',
                figure=price_tier_pr_curves(m1_pr),
                style={'height': '420px'}
            ), width=5)
        ], className="mb-5 justify-content-center"),
    ]
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
        dbc.Tabs([eda_tab, clustering_tab, tda_tab, classification_tab], className="mt-3"),
        html.Footer(
            dbc.Container(
                [
                    html.Hr(),
                    html.P(
                        "© 2025 Cartagena Airbnb TDA Project | Santiago Baca & Edson Álvarez", 
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