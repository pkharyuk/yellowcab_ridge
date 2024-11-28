from collections.abc import Iterator
from typing import Any

#from ipywidgets import interact
import ipywidgets as widgets

import backend


def get_reversed_enumerated_dict(
    data: Iterator[Any]
) -> dict[Any, int]:
    return dict(
        (datapoint, i) for i, datapoint in enumerate(data)
    )

def plot_dataset(results: backend.ResultsDict) -> None:
    widgets.interact(
        results[backend.plotmap_key],
        i=widgets.SelectionSlider(
            options=get_reversed_enumerated_dict(
                results[backend.df_datetimes_key]
            ),
            value=0,
            description="Datetime: ",
            orientation="horizontal",
            layout={"width": "800px"},
        )
    );

def plot_timeseries(results: backend.ResultsDict) -> None:
    widgets.interact(
        results[backend.pts1_key],
        ind=widgets.Dropdown(
            options=get_reversed_enumerated_dict(
                results[
                    backend.filtered_regions_key
                ][backend._region_colname].values
            ),
            value=0,
            description="Region index:",
        )
    );

def plot_paired_timeseries(results: backend.ResultsDict):
    widgets.interact(
        results[backend.pts2_key],
        ind=widgets.Dropdown(
            options=get_reversed_enumerated_dict(
                results[
                    backend.filtered_regions_key
                ][backend._region_colname].values
            ),
            value=0,
            description="Region index:",
        ),
        lag=widgets.Dropdown(
            options=dict(
                (i+1, i) for i in range(results[backend.deltaT_key])
            ),
            value=0,
            description="Time lag:",
        ),
        show_original=widgets.Checkbox(
            value=True,
            description="Show real data",
        ),
        show_prediction=widgets.Checkbox(
            value=True,
            description="Show predicted",
        ),
    );
    
    
    
