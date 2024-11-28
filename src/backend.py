import os
import datetime
# import functools # partial not working well with folium
from typing import Callable, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

import folium
import geojson


_quantiles: list[int] = [0, 50, 75, 90, 95] 
_timeseries_figsize: tuple[int, int] = (30, 5)

def round_to_order_of_magnitude(a: float) -> int:
    if np.isclose(a, 0):
        return a
    n: int = int(np.floor(np.log10(a)))
    return int(np.round(a, 1-n))

def compute_rounded_quantiles(
    df_series: pd.Series, qs: list[int] = _quantiles
) -> list[int]:
    return [
        round_to_order_of_magnitude(
            np.percentile(df_series.values, q)
        ) for q in qs
    ]
    

# configuring paths
data_path: str = "./data/"
fnm_regions: str = "regions.csv"
regions_geo_filename: str = "map.geojson"
new_df_fnm: str = "regions_counted_dump.csv"
# file containing predictions
pred_filename: str = "pred_june.npz"

# coordinates of the starting point (Empire State Building)
_lat_ESB: float = 40.748817
_lon_ESB: float = -73.985428
_region_colname: str = "region"
_counts_colname: str = "trip_counts"

# time constants
_sd: int = 24
_sec_in_hour: int = 3600
_end_history1: str = "2016.05.31 23:00"
_end_history2 = "2016.06.30 17:00"
_left_border_date: datetime.date = datetime.date(month=5, year=2016, day=1)
_right_border_date: datetime.date = datetime.date(month=6, year=2016, day=1)

# column names of the resulting dictionary
pts1_key: str = "plotTimeSeries"
pts2_key: str = "plotTimeSeries2"
plotmap_key: str = "plotMap"
filtered_regions_key: str = "filtered_regions"
ind_regions_key: str = "ind_regions"
deltaT_key: str = "deltaT"
df_datetimes_key: str = "datetimes_from_df"

_timeseries_plot_kwargs: dict[str, Any] = {
    "color": "darkred",
    "grid": 0.5,
    "ylabel": "Counts",
    "xlabel": "Time (hour)",
}
_prediction_color: str = "goldenrod"

# some custom types
PolygonPoint = list[float, float]
ResultsDict = dict[str, Any]


def get_polygon_points_list(df: pd.DataFrame) -> list[PolygonPoint]:
    return [
        [df["west"], df["south"]],
        [df["east"], df["south"]],
        [df["east"], df["north"]],
        [df["west"], df["north"]],
        [df["west"], df["south"]],
    ]

def get_geojson_feature_from_polygon(
    df: pd.DataFrame, id_colname: str = _region_colname
):
    polygon: list[PolygonPoint] = get_polygon_points_list(df)
    return geojson.Feature(
        geometry=geojson.Polygon(
            [polygon]
        ),
        id=int(df[id_colname])
    )

def region2geojson(
    reg_df: pd.DataFrame,
    save_fnm: str = "map.geojson"
) -> None:
    '''
    Converts dataframe to (geo)json suitable for drawing with folium
    '''
    features: list[geojson.Feature] = []
    reg_df.apply(
        lambda X: features.append(
            get_geojson_feature_from_polygon(X)
        ), axis=1
    )
    with open(save_fnm, 'w') as fd:
        geojson.dump(
            geojson.FeatureCollection(features), fd, sort_keys=True
        )

def get_suffix(string: str, separating_char: str = '_') -> str:
    '''
    revert -> split by separator -> take the first part
    (the tail of the original) -> revert again
    '''
    return string[::-1].split(separating_char)[0][::-1]

def plot_map(
    i: int,
    regions_geojson_path: str,
    df_data: pd.DataFrame,
    ind_regions: npt.NDArray[np.int32],
    #longitude_points: npt.NDArray[np.float64],
    #latitude_points: npt.NDArray[np.float64],
    min_longitude: int,
    max_longitude: int,
    min_latitude: int,
    max_latitude: int,
    default_starting_lon: float = _lon_ESB,
    default_starting_lat: float = _lat_ESB,
    region_colname: str = _region_colname,
    counts_colname: str = _counts_colname,
) -> folium.Map:
    '''
    ''' 
    map_ny: folium.Map = folium.Map(
        tiles="cartodb voyager",
        zoom_start=11,
        min_zoom=10,
        location=[default_starting_lat, default_starting_lon],
        min_lon=min_longitude,
        max_lon=max_longitude,
        min_lat=min_latitude,
        max_lat=max_latitude,
        max_bounds=True,
        control_scale=True,
    )
    map_ny.options['maxBounds'][0][1] = min_longitude
    map_ny.options['maxBounds'][1][1] = max_longitude
    map_ny.options['maxZoom'] = 12
    #map_ny.fit_bounds([(min_longitude, max_longitude), (min_latitude, max_latitude)])
    data: pd.DataFrame = df_data.iloc[i:i+1, ind_regions].T
    data.reset_index(inplace=True)
    data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: get_suffix(x))
    #data.iloc[:, 0] = data.iloc[:, 0].map(float)
    data.columns = [region_colname, counts_colname]

    threshold_scale: list[int] = compute_rounded_quantiles(data[counts_colname])
    threshold_scale.append(int(round(data.trip_counts.max())))

    folium.Choropleth(
        geo_data=regions_geojson_path,
        name="choropleth",
        data=data,
        columns=data.columns,
        key_on="feature.id",
        fill_color="Reds",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Number of trips",
        threshold_scale=threshold_scale,
        reset=True,
    ).add_to(map_ny)

    return map_ny

def plot_time_series(
    ind: int,
    df: pd.DataFrame,
    ind_regions: npt.NDArray[np.int32],
    figsize: tuple[int, int] = _timeseries_figsize,
) -> None:
    fig: plt.figure
    ax: plt.Axes
    ind_region: int = ind_regions[ind]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df.iloc[:, ind_region].plot(
        ax=ax, **_timeseries_plot_kwargs
    )
    plt.show()

def plot_time_series2(
    ind: int,
    lag: int,
    show_original: bool,
    show_prediction: bool,
    df: pd.DataFrame,
    predictions: npt.NDArray[np.float64],
    ind_regions: npt.NDArray[np.int32],
    delta_T: int,
    num_of_predictions_pj: int,
    label_original: str = "real data",
    label_prediction: str = "predicted",
    figsize: tuple[int, int] = _timeseries_figsize,
) -> None:
    if not (show_prediction or show_original):
        plt.clf()
        return
    ind_region: int = ind_regions[ind]
    fig: plt.figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    margin: int = delta_T - lag - 1
    timeseries_plot_kwargs: dict[str, Any] = _timeseries_plot_kwargs.copy()
    min_y: int = 0
    max_y: int = max(
        df.iloc[:, ind_region].max(),
        predictions[ind, :, :].max()
    )
    if show_original:
        if lag < delta_T-1: # margin > 0
            df.iloc[
                -num_of_predictions_pj-margin : -margin, ind_region
            ].plot(ax=ax, label=label_original, **timeseries_plot_kwargs)
        else:
            df.iloc[
                -num_of_predictions_pj : , ind_region
            ].plot(ax=ax, label=label_original, **timeseries_plot_kwargs)
    if show_prediction:
        timeseries_plot_kwargs["color"] = _prediction_color
        if lag < delta_T-1:
            tmp_df = pd.DataFrame(
                index=df.index[-num_of_predictions_pj-margin : -margin],
                data=predictions[ind, :, lag]
            )
            tmp_df.iloc[:, 0].plot(
                ax=ax, label=label_prediction, **timeseries_plot_kwargs
            )
        else:
            tmp_df = pd.DataFrame(
                index=df.index[-num_of_predictions_pj - margin:],
                data=predictions[ind, :, lag]
            )
            tmp_df.iloc[:, 0].plot(
                ax=ax, label=label_prediction, **timeseries_plot_kwargs
            )
    ax.set_ylim((min_y, max_y))
    ax.legend()
    plt.show()
    if show_prediction:
        del tmp_df

def load_counts_data(
    data_path: str,
    filename: str,
    time_colname: str = "tpep_pickup_datetime",
    time_column_dtype: str = "datetime64[ns]",
) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_path, filename), )
    df = df.set_index(
        pd.DatetimeIndex(
            df[time_colname],
            dtype=time_column_dtype,
            freq=None,
        )
    )
    df = df.asfreq("h")
    df = df[df.columns[1:]].astype("i")
    df = df.fillna(0)
    return df
    
        
def init(
    longitude_colindex: int = 1,
    latitude_colindex: int = 3,
    n_lat_points: int = 50,
    left_border_date: datetime.date = _left_border_date,
    right_border_date: datetime.date = _right_border_date,
    min_avg_count_per_day: int = 5,
    avg_trip_counts_colname: str = "mean_trip_counts",
) -> ResultsDict:
    regions: pd.DataFrame = pd.read_csv(
        os.path.join(data_path, fnm_regions), sep=";"
    )
    #region;west;east;south;north
    n_regions_records: int = len(regions)
    lon_points = regions.loc[::n_lat_points, "west"].values
    lon_points = np.append(lon_points, [regions.loc[n_regions_records-1, "east"]])
    lat_points = regions.loc[:n_lat_points, "south"].values
    lat_points = np.append(lat_points, [regions.loc[n_regions_records-1, "north"]])

    # https://gis.stackexchange.com/questions/220997/pandas-to-geojson-multiples-points-features-with-python
    df: pd.DataFrame = load_counts_data(data_path, new_df_fnm)
    regions[avg_trip_counts_colname] = np.mean(
        df[
            (df.index >= np.datetime64(left_border_date)) &
            (df.index < np.datetime64(right_border_date))
        ].values, axis=0
    )
    ind_regions: npt.NDArray[np.int32] = np.where(
        regions[avg_trip_counts_colname].values >= min_avg_count_per_day
    )[0]
    Nzones: int = len(ind_regions)

    filtered_regions = regions.iloc[ind_regions]
    regions_geo_path: str = os.path.join(data_path, regions_geo_filename)
    region2geojson(filtered_regions, regions_geo_path)
    threshold_scale: list[int] = compute_rounded_quantiles(
        filtered_regions[avg_trip_counts_colname]
    )
    plot_map_func: Callable[int, folium.Map] = lambda i: (
        plot_map(
            i,
            regions_geojson_path=regions_geo_path,
            df_data=df,
            ind_regions=ind_regions,
            min_longitude=lon_points[0],
            max_longitude=lon_points[-1],
            min_latitude=lat_points[0],
            max_latitude=lat_points[-1],
        )
    )
    map_ny: folium.Map = plot_map_func(i=0)


    df_pred = np.load(os.path.join(data_path, pred_filename))
    pred_june = df_pred["pred_june"]
    num_of_zones: int
    num_of_predictions_pj: int
    delta_T: int
    num_of_zones, num_of_predictions_pj, delta_T = pred_june.shape
    assert num_of_zones == len(ind_regions)

    end_history1: datetime.datetime = datetime.datetime.strptime(
        _end_history1, "%Y.%m.%d %H:%M"
    )
    end_history2: datetime.datetime = datetime.datetime.strptime(
        _end_history2, "%Y.%m.%d %H:%M"
    )

    num_of_predictions_td: datetime.timedelta = end_history2 - end_history1
    num_of_predictions: int = (
        num_of_predictions_td.seconds/_sec_in_hour
        + _sd*num_of_predictions_td.days
        + 1
    )
    assert num_of_predictions_pj == num_of_predictions

    plot_time_series_func: Callable[int, None] = lambda ind: (
        plot_time_series(
            ind,
            df=df,
            ind_regions=ind_regions,
        )
    )
    plot_time_series2_func: Callable[
        [int, int, bool, bool], None
    ] = lambda ind, lag, show_original, show_prediction: (
        plot_time_series2(
            ind=ind,
            lag=lag,
            show_original=show_original,
            show_prediction=show_prediction,
            df=df,
            predictions=pred_june,
            ind_regions=ind_regions,
            delta_T=delta_T,
            num_of_predictions_pj=num_of_predictions_pj,
        )
    )

        
    results: ResultsDict = {
        pts1_key: plot_time_series_func,
        pts2_key: plot_time_series2_func,
        plotmap_key: plot_map_func,
        filtered_regions_key: filtered_regions,
        ind_regions_key: ind_regions,
        deltaT_key: delta_T,
        df_datetimes_key: df.index,
        #"df": df,
    }
    return results
    






