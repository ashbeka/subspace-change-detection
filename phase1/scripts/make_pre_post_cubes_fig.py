from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


# Script lives in phase1/scripts; PHASE1_ROOT should be phase1/.
PHASE1_ROOT = Path(__file__).resolve().parents[1]


def draw_cube(ax, x, y, width, height, depth, face_color, edge_color, label):
    """Draw a simple pseudo-3D cube."""
    # Front face
    front = Rectangle((x, y), width, height, facecolor=face_color, edgecolor=edge_color)
    ax.add_patch(front)

    # Top face (slanted)
    top = Polygon(
        [
            (x, y + height),
            (x + depth, y + height + depth * 0.6),
            (x + width + depth, y + height + depth * 0.6),
            (x + width, y + height),
        ],
        closed=True,
        facecolor=face_color,
        edgecolor=edge_color,
    )
    ax.add_patch(top)

    # Side face
    side = Polygon(
        [
            (x + width, y),
            (x + width + depth, y + depth * 0.6),
            (x + width + depth, y + height + depth * 0.6),
            (x + width, y + height),
        ],
        closed=True,
        facecolor=face_color,
        edgecolor=edge_color,
    )
    ax.add_patch(side)

    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )


def main():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Draw two cubes: Pre and Post
    draw_cube(
        ax,
        x=0.5,
        y=0.5,
        width=1.5,
        height=1.5,
        depth=0.4,
        face_color="#4C72B0",  # blue-ish
        edge_color="black",
        label="Pre image\n(13 bands)",
    )

    draw_cube(
        ax,
        x=3.0,
        y=0.5,
        width=1.5,
        height=1.5,
        depth=0.4,
        face_color="#55A868",  # green-ish
        edge_color="black",
        label="Post image\n(13 bands)",
    )

    ax.text(
        0.5 + 1.5 / 2,
        0.2,
        "Pre-event Sentinel-2 stack",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(
        3.0 + 1.5 / 2,
        0.2,
        "Post-event Sentinel-2 stack",
        ha="center",
        va="center",
        fontsize=9,
    )

    out_path = PHASE1_ROOT / "docs" / "figs" / "phase1_pre_post_cubes.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
