import math

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def __get_percentile(data, p: float):
    # calculate quartiles as written in the plotly documentation (however it works better when self-implemented):
    # https://plotly.com/python/box-plots/#choosing-the-algorithm-for-computing-quartiles
    data.sort()
    n = len(data)
    x = n * p + 0.5
    x1, x2 = math.floor(x), math.ceil(x)
    if x1 == x2:
        return data[x1 - 1]
    y1, y2 = data[x1 - 1], data[x2 - 1]  # account for zero-indexing
    return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)


def plot_attack_area_analysis_selfmade_quantiles(atk_types_data: dict, x_values: [int], x_label: str,
                                                 figure_name: str = None, real_world_data: dict = None,
                                                 is_starlink: bool = False, is_interObsDist: bool = False):
    # atk_types_data = {atk_type: [[x1], [x2], ...] }
    if figure_name is None:
        figure_name = f"Areas (5th percentile, Q1, median, Q3, 95th percentile) of different attacker types"
    starlink_attacker_color = {'3a (1rec)': px.colors.qualitative.Plotly[6],
                               '3b (3rec)': px.colors.qualitative.Plotly[2]}
    fig = go.Figure()
    attacker_types = list(atk_types_data.keys())
    for temp_type in attacker_types:
        temp_data = atk_types_data[temp_type]
        my_x_axis = []
        for x_element_index in range(len(temp_data)):
            x_element_list = temp_data[x_element_index]
            x_element_len = len(x_element_list)
            new_x_part = [x_values[x_element_index]] * x_element_len
            my_x_axis = my_x_axis + new_x_part
        array_data = np.array([])
        low_fences = []
        upper_fences = []
        q1_list = []
        q2_list = []
        q3_list = []
        for i in range(len(temp_data)):
            temp_arr = np.array(temp_data[i])
            array_data = np.concatenate((array_data, temp_arr))
            low_fences.append(__get_percentile(temp_arr, 0.05))
            q1_list.append(__get_percentile(temp_arr, 0.25))
            q2_list.append(__get_percentile(temp_arr, 0.5))
            q3_list.append(__get_percentile(temp_arr, 0.75))
            upper_fences.append(__get_percentile(temp_arr, 0.95))
        if is_starlink:
            new_trace = go.Box(name=temp_type, lowerfence=low_fences, q1=q1_list, median=q2_list, q3=q3_list,
                               upperfence=upper_fences, marker_color=starlink_attacker_color[temp_type])
        else:
            if is_interObsDist and temp_type in ["weak noisy (1/3rec)", "weak noisy1 (1/6rec)", "weak noisy1 (6rec)"]:
                new_trace = go.Box(name=temp_type, lowerfence=low_fences, q1=q1_list, median=q2_list, q3=q3_list,
                                   upperfence=upper_fences, visible='legendonly')
            else:
                new_trace = go.Box(name=temp_type, lowerfence=low_fences, q1=q1_list, median=q2_list, q3=q3_list,
                                   upperfence=upper_fences)
        fig.add_trace(new_trace)

    if real_world_data != None:
        time_durations = real_world_data['times']
        estimations = real_world_data['estimations']
        fig.add_trace(go.Scatter(x=time_durations, y=estimations,
                                 mode='markers', marker=dict(color='green', size=10),
                                 ))

    # build custom x-axis
    x_tick_vals = []
    for i in range(len(x_values)):
        x_tick_vals.append(i)
    if is_starlink:
        my_yaxis_range = [-1.8, 3.5]  # for figure 15 (starlink)
    else:
        if is_interObsDist:
            my_yaxis_range = [1.4, 4.9]  # for figure 13 (inter observer distance + observer amount)
        else:
            my_yaxis_range = [-2.2, 6.2]  # for figure 10, 11, 12
    # add custom layout
    fig.update_layout(
        title_text=figure_name,
        width=850,  # 850 (fpr figures 10, 11, 13, 15), 950 (for figure 12)
        height=500,  # 500 for iridium, 400 for starlink
        xaxis_title=x_label,
        yaxis_title="region of interest [km²]",
        yaxis_range=my_yaxis_range,
        boxmode='group',  # group together boxes of the different traces for each value of x
        xaxis=dict(
            tickmode='array',
            tickvals=x_tick_vals,
            ticktext=x_values
        ),
        font=dict(
            size=16,
        )
    )
    fig.update_yaxes(type="log")
    fig.show()
