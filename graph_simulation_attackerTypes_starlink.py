import numpy as np
import os
import csv

import plot_measurement_row

def read_output_csv(filename: str):
    filename = filename
    if os.path.exists(filename):
        all_rows = []
        with open(filename, 'r') as csvinput:
            reader = csv.reader(csvinput)
            for row in reader:
                if row[0][0] == '[':
                    row[0] = row[0][1:]
                if row[-1][-1] == ']':
                    row[-1] = row[-1][:-1]
                all_rows.append(row)
        csvinput.close()
        all_colums = np.array(all_rows, dtype=float).T
        return all_colums
    else:
        print(f"ERROR: read_output_csv: file ({filename}) does not exist!")


def combine_rec_outputs_to_plot(x_axis: [float], x_label: str, figure_name: str, output_folder: str,
                                output_mapping: [(int, int, str)], atk_types: [str]):
    if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
        print(f"ERROR: Ms_RN.combine_output_to_plot: given folder ({output_folder}) doesn't exist or is not a folder!")
        exit(1)
    if output_folder[-1] is not os.sep:
        output_folder = f"{output_folder}{os.sep}"
    # prepare the data-structure to fill in all the data-points
    temp_data_list = []
    for j in range(len(atk_types)):
        type_list = []
        for k in range(len(x_axis)):
            type_list.append([])
        temp_data_list.append(type_list)
    # get each file and fill the data-structure
    for entry in output_mapping:
        temp_x_index, temp_atk_type, temp_file = entry
        print(f"open: {temp_file}")
        some_data = read_output_csv(output_folder + temp_file)
        for i in range(len(some_data)):
            loaded_column = some_data[i, :]
            temp_data_list[temp_atk_type + i][temp_x_index] = loaded_column
    # create the plot
    print_dict = {}  # {atk_type: [[x1], [x2], ...] }
    for i in range(len(atk_types)):
        print_dict[atk_types[i]] = temp_data_list[i]
    plot_measurement_row.plot_attack_area_analysis_selfmade_quantiles(print_dict, x_axis, x_label=x_label, figure_name=figure_name, is_starlink=True)


if __name__ == '__main__':
    fig_name = f"Starlink Areas (5th percentile, Q1, median, Q3, 95th percentile) of different rec-setups over different sniffing intervals"

    attack_types = [
        "3a (1rec)", "3b (3rec)",
    ]
    x_values = ['1 min', '3 min', '10 min', '30 min']
    x_text = f"observation duration"
    folder_name = "simulation_data/starlink/"

    mapping = [
        (0, 0, "40kmFibo_cont_60sec_3eves_weakEvents.csv"),
        (1, 0, "40kmFibo_cont_180sec_3eves_weakEvents.csv"),
        (2, 0, "40kmFibo_cont_600sec_3eves_weakEvents.csv"),
        (3, 0, "40kmFibo_cont_1800sec_3eves_weakEvents.csv"),
    ]
    combine_rec_outputs_to_plot(x_axis=x_values, x_label=x_text, figure_name=fig_name, output_folder=folder_name,
                                output_mapping=mapping, atk_types=attack_types)
