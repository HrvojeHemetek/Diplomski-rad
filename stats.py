import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def grouped_bar_chart():
    df = pd.read_csv("results.csv")

    df["config"] = "n=" + df["n"].astype(str) + ", t=" + df["time"].astype(str) + ", " + df["model"]

    df_opt = df[df["optimize"] == True].reset_index(drop=True)
    df_no_opt = df[df["optimize"] == False].reset_index(drop=True)

    labels = df_no_opt["config"].tolist()

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - bar_width / 2, df_no_opt["average_error"], bar_width, label="Bez optimizacije", color="#4C72B0")
    bars2 = ax.bar(x + bar_width / 2, df_opt["average_error"], bar_width, label="S optimizacijom", color="#DD8452")

    ax.set_ylabel("Prosječna greška")
    ax.set_title("Usporedba prosječne greške po konfiguraciji za ε = 0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def grouped_bar_chart_success_rate():
    df = pd.read_csv("results.csv")

    df["config"] = "n=" + df["n"].astype(str) + ", t=" + df["time"].astype(str) + ", " + df["model"]

    df_opt = df[df["optimize"] == True].reset_index(drop=True)
    df_no_opt = df[df["optimize"] == False].reset_index(drop=True)

    labels = df_no_opt["config"].tolist()

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - bar_width / 2, df_no_opt["success_rate"], bar_width, label="Bez optimizacije", color="#4C72B0")
    bars2 = ax.bar(x + bar_width / 2, df_opt["success_rate"], bar_width, label="S optimizacijom", color="#DD8452")

    ax.axhline(y=100 - 100/np.e, color='r', linestyle='--')
    ax.set_ylabel("Stopa uspješnosti (%)")
    ax.set_title("Usporedba stope uspješnosti po konfiguraciji za ε = 0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    grouped_bar_chart()
    grouped_bar_chart_success_rate()