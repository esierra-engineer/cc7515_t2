import pandas as pd
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida
os.makedirs("plots", exist_ok=True)

# Leer CSV
df = pd.read_csv("CUDA_resultados.csv")
df = df[df["time"] != "ERROR"]
df["time"] = pd.to_numeric(df["time"], errors="coerce")

# Separar CPU y GPU
df_cpu = df[df["method"] == "CPU"]
df_gpu = df[df["method"] != "CPU"]

# Obtener valores únicos
steps_list = sorted(df["steps"].unique())
methods = sorted(df_gpu["method"].unique())
local_sizes = sorted(df_gpu["localSize"].dropna().unique())
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']

# --- Mejores casos ---
agg_gpu = df_gpu.groupby(["n", "steps", "method"])["time"].min().reset_index()

for steps in steps_list:
    plt.figure(figsize=(10, 6))
    subset_cpu = df_cpu[df_cpu["steps"] == steps]
    plt.plot(subset_cpu["n"], subset_cpu["time"], label="CPU", marker='x', color="black")

    for i, method in enumerate(methods):
        subset = agg_gpu[(agg_gpu["steps"] == steps) & (agg_gpu["method"] == method)]
        if not subset.empty:
            plt.plot(subset["n"], subset["time"], label=method, marker='o', color=colors[i % len(colors)])

    plt.title(f"Tiempos de ejecución (mejor caso por método) - steps = {steps}")
    plt.xlabel("Número de cuerpos (n)")
    plt.ylabel("Tiempo (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/times_best_steps_{steps}.png")
    plt.close()

    # --- Speedup  ---
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        subset_gpu = agg_gpu[(agg_gpu["steps"] == steps) & (agg_gpu["method"] == method)]
        merged_simple = pd.merge(subset_cpu, subset_gpu, on="n", suffixes=("_cpu", "_gpu"))
        if not merged_simple.empty:
            merged_simple["speedup"] = merged_simple["time_cpu"] / merged_simple["time_gpu"]
            plt.plot(merged_simple["n"], merged_simple["speedup"], label=method, marker='s', color=colors[i % len(colors)])
    plt.title(f"Speedup (CPU / mejor caso GPU) - steps = {steps}")
    plt.xlabel("Número de cuerpos (n)")
    plt.ylabel("Speedup")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/speedup_combined_steps_{steps}.png")
    plt.close()

# --- SUBPLOTS separados por método ---
for steps in steps_list:
    for i, method in enumerate(methods):
        plt.figure(figsize=(10, 6))
        for ls in local_sizes:
            subset = df_gpu[(df_gpu["steps"] == steps) & (df_gpu["method"] == method) & (df_gpu["localSize"] == ls)]
            if not subset.empty:
                label = f"ls={ls}"
                plt.plot(subset["n"], subset["time"], label=label, marker='o')

        plt.title(f"{method} - steps = {steps}")
        plt.xlabel("Número de cuerpos (n)")
        plt.ylabel("Tiempo (s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"plots/subplot_times_steps_{steps}_{method}.png"
        plt.savefig(filename)
        plt.close()

print("Gráficos generados en carpeta 'plots'.")