import os
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt





def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


if __name__ == "__main__":

    filename = str(sys.argv[1])
    if len(sys.argv)>2:
        if sys.argv[2] == "--plot":
            is_plot = True
        else:
            is_plot = False

    xs, ys = load_points_from_file(filename)
    view_data_segments(xs,ys)
    print("hello")
