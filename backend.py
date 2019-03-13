import folium
from branca.utilities import split_six
from ipywidgets import interact
import ipywidgets as widgets

import pandas as pd
import geojson
import datetime
import numpy as np

import matplotlib.pyplot as plt

# global variables (bad practice...)
regions_geo = 'map.geojson'
new_df_fnm = 'regions_counted_dump.csv'
latESB, lonESB = 40.748817, -73.985428
pred_filename = 'pred_june.npz'
Sd = 24
secInHour = 3600
_endHistory1 = '2016.05.31 23:00'
_endHistory2 = '2016.06.30 17:00'


def region2geojson(reg_df, save_fnm='map.geojson'):
    features = []
    reg_df.apply(
        lambda X: features.append(
            geojson.Feature(
                geometry=geojson.Polygon(
                    [
                        [
                            [X['west'], X['south']],
                            [X['east'], X['south']],
                            [X['east'], X['north']],
                            [X['west'], X['north']],
                            [X['west'], X['south']]
                        ]
                    ]
                ), 
                id=X['region']
            )
        ),
        axis=1
    )
    with open(save_fnm, 'w') as f:
        geojson.dump(geojson.FeatureCollection(features), f, sort_keys=True)
        
def main():
    dirname = './'
    fnm_regions = 'regions.csv'
    regions = pd.read_csv(dirname+fnm_regions, sep=';')

    lonPoints = regions.iloc[::50, 1].values
    lonPoints = np.append(lonPoints, regions.iloc[-1, 2])
    latPoints = regions.iloc[:50, 3].values
    latPoints = np.append(latPoints, regions.iloc[-1, 4])

    # https://gis.stackexchange.com/questions/220997/pandas-to-geojson-multiples-points-features-with-python

    

    df = pd.read_csv(dirname+new_df_fnm, )
    df = df.set_index(
        pd.DatetimeIndex(
            df['tpep_pickup_datetime'],
            dtype='datetime64[ns]',
            freq=None
        )
    )
    df = df.asfreq('H')
    df = df[df.columns[1:]]
    df = df.astype('i')
    df = df.fillna(0)

    left_border_date = np.datetime64(datetime.date(month=5, year=2016, day=1))
    right_border_date = np.datetime64(datetime.date(month=6, year=2016, day=1))

    regions['mean_trip_counts'] = np.mean(
        df[
            (df.index >= left_border_date) &
            (df.index < right_border_date)
        ].values,
        axis=0
    )

    min_mean_count_per_day = 5

    ind_regions = np.where(regions['mean_trip_counts'].values >= min_mean_count_per_day)[0]
    Nzones = len(ind_regions)

    filtered_regions = regions.iloc[ind_regions]
    region2geojson(filtered_regions, regions_geo)

    threshold_scale = split_six(filtered_regions['mean_trip_counts'])



    def plotMap(i):
        mapNY3 = folium.Map(
            zoom_start=11,
            location=[latESB, lonESB], min_lon=lonPoints[0], max_lon=lonPoints[-1],
            min_lat=latPoints[0], max_lat=latPoints[-1]
        )
        #mapNY3.fit_bounds([(latPoints[0], lonPoints[0]), (latPoints[-1], lonPoints[-1])])
        data = df.iloc[i:i+1, ind_regions].T

        data.reset_index(inplace=True)

        data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: x[::-1].split('_')[0][::-1])
        data.iloc[:, 0] = data.iloc[:, 0].map(float)
        data.columns = ['region', 'trip_counts']


        threshold_scale = split_six(data['trip_counts'])
        threshold_scale.append(data.trip_counts.max())


        folium.Choropleth(
            geo_data=regions_geo,
            name='choropleth',
            data=data,
            columns=['region', 'trip_counts'],
            key_on='feature.id',
            fill_color='Reds',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Number of trips',
            threshold_scale=threshold_scale,
            reset=True
        ).add_to(mapNY3)


        return mapNY3 # интерактивная карта

    mapNY3 = plotMap(0)


    numOfZones = len(ind_regions)


    df_pred = np.load(dirname+pred_filename)
    pred_june = df_pred['pred_june']
    numOfZones, numOfPredictions_pj, deltaT = pred_june.shape

    assert numOfZones == len(ind_regions)



    endHistory1 = datetime.datetime.strptime(_endHistory1, '%Y.%m.%d %H:%M')
    endHistory2 = datetime.datetime.strptime(_endHistory2, '%Y.%m.%d %H:%M')

    numOfPredictions = endHistory2 - endHistory1
    numOfPredictions = numOfPredictions.seconds / secInHour + Sd*numOfPredictions.days + 1

    assert numOfPredictions_pj == numOfPredictions

    def plotTimeSeries(ind_region):
        fig, ax = plt.subplots(1, 1, figsize=(30, 5))
        df.iloc[:, ind_regions[ind_region]].plot(ax=ax)
        plt.show()

    def plotTimeSeries2(ind_region, lag, show_original, show_prediction):
        fig, ax = plt.subplots(1, 1, figsize=(30, 5))
        margin = deltaT-lag-1
        if show_original:
            if lag < deltaT-1: # margin > 0
                df.iloc[-numOfPredictions_pj-margin:-margin, ind_regions[ind_region]].plot(ax=ax, label='real data')
            else:
                df.iloc[-numOfPredictions_pj:, ind_regions[ind_region]].plot(ax=ax, label='real data')
        if show_prediction:
            if lag < deltaT-1:
                tmp_df = pd.DataFrame(
                    index=df.index[-numOfPredictions_pj-margin:-margin],
                    data=pred_june[ind_region, :, lag]
                )
                tmp_df.iloc[:, 0].plot(ax=ax, label='predicted')
            else:
                tmp_df = pd.DataFrame(
                    index=df.index[-numOfPredictions_pj-margin:],
                    data=pred_june[ind_region, :, lag]
                )
                tmp_df.iloc[:, 0].plot(ax=ax, label='predicted')
        ax.legend()
        plt.show()
        
    stuff = {}
    stuff['plotTimeSeries'] = plotTimeSeries
    stuff['plotTimeSeries2'] = plotTimeSeries2
    stuff['plotMap'] = plotMap
    
    stuff['filtered_regions'] = filtered_regions
    stuff['ind_regions'] = ind_regions
    stuff['deltaT'] = deltaT
    
    stuff['datetimes_from_df'] = df.index
    
    return stuff
    






