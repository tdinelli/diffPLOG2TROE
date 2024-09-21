import csv
import jax.numpy as jnp


def load_data_matrix(data_path) -> jnp.ndarray:
    """Useful function that load a csv into a JAX array matrix"""
    with open(data_path, newline="", encoding="utf-8-sig") as file:
        csv_reader = csv.reader(file, delimiter=';')
        print(csv_reader)
        tmp = []
        for row in csv_reader:
            row_data = [float(x) for x in row]
            tmp.append(row_data)
    file.close()

    return jnp.asarray(tmp, dtype=jnp.float64)
