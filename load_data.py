import pathlib

import numpy as onp
import jax.numpy as jnp

import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path("data/kyushu_university")


def drop_nan_rows(d):
    nan_rows = onp.isnan(d).any(axis=1)
    return d[onp.logical_not(nan_rows)]


if __name__ == "__main__":
    for item in DATA_DIR.iterdir():
        if item.suffix == ".csv":
            try:
                data = onp.genfromtxt(item, delimiter=",", skip_header=1)
                print(item, data.shape, drop_nan_rows(data).shape)
                headers = onp.genfromtxt(item, delimiter=",", max_rows=1, dtype=str)
                plt.plot(drop_nan_rows(data), label=headers)
                plt.legend()
                plt.show()
                # print(headers)
            except Exception as e:
                print("Failed to load", item, "because", e)
