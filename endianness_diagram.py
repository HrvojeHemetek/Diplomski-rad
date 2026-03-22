import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_circuit(ax, qubit_labels, gate_qubit, title, state_label, decimal_val):
    n_qubits = len(qubit_labels)
    wire_y = list(range(n_qubits - 1, -1, -1))
    x_start, x_end = 0.5, 4.5
    gate_x = 2.5

    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.8, n_qubits - 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    for i, (label, y) in enumerate(zip(qubit_labels, wire_y)):
        ax.plot([x_start, x_end], [y, y], "k-", linewidth=1.2)
        ax.text(x_start - 0.15, y, f"|{label}⟩", ha="right", va="center", fontsize=13)

        if i == gate_qubit:
            box_size = 0.3
            rect = mpatches.FancyBboxPatch(
                (gate_x - box_size, y - box_size),
                2 * box_size, 2 * box_size,
                boxstyle="round,pad=0.05",
                facecolor="#5B9BD5", edgecolor="black", linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(gate_x, y, "X", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

        if i == gate_qubit:
            ax.text(x_end + 0.15, y, "|1⟩", ha="left", va="center", fontsize=13)
        else:
            ax.text(x_end + 0.15, y, "|0⟩", ha="left", va="center", fontsize=13)

    ax.text(2.5, -0.6, f"Stanje: {state_label} = |{decimal_val}⟩",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F2F2F2", edgecolor="#999999"))


def show_endianness_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Prikaz poretka kvantnih bitova u kvantnim logičkim krugovima", fontsize=16, fontweight="bold", y=0.99)

    draw_circuit(
        ax1,
        qubit_labels=["q₀", "q₁", "q₂"],
        gate_qubit=0,
        title="Big-endian (q₀ = MSB)",
        state_label="|q₀q₁q₂⟩ = |100⟩",
        decimal_val="4"
    )

    draw_circuit(
        ax2,
        qubit_labels=["q₀", "q₁", "q₂"],
        gate_qubit=0,
        title="Little-endian (q₀ = LSB)",
        state_label="|q₂q₁q₀⟩ = |001⟩",
        decimal_val="1"
    )

    plt.tight_layout()
    plt.savefig("endianness_diagram.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    show_endianness_diagram()
