import matplotlib.pyplot as plt

def make_line_plot(df, x, y, title=""):
    """Gráfica de línea simple con matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        ax.plot(df[x], df[y])
    except Exception as e:
        ax.text(0.5, 0.5, f"Error al graficar: {e}", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return fig
