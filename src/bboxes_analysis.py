import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent

def plot_track_stability_3D(annotation_file, N, output_dir="./plots/track_stability"):
    """
    Plot the first N track IDs' bounding boxes over time in 3D.
    Each track shows (frame, x1, y1) and (frame, x2, y2).
    
    Args:
        annotation_file (str): Path to MOT-style .txt file.
        N (int): Number of unique IDs to plot.
        output_dir (str): Directory to save the plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract video name from file
    video_name = os.path.splitext(os.path.basename(annotation_file))[0]

    # Load annotations
    cols = ["frame", "id", "x", "y", "w", "h", "score", "a1", "a2", "a3"]
    df = pd.read_csv(annotation_file, header=None, names=cols)

    # Keep only first N unique IDs
    unique_ids_all = sorted(df["id"].unique())
    if len(unique_ids_all) < N:
        print(f"⚠️ Only {len(unique_ids_all)} tracks available, plotting all of them.")
        unique_ids = unique_ids_all
    else:
        unique_ids = unique_ids_all[:N]

    df = df[df["id"].isin(unique_ids)]

    # Setup 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    for idx, tid in enumerate(unique_ids):
        track = df[df["id"] == tid]

        frames = track["frame"].values
        x1 = track["x"].values
        y1 = track["y"].values
        x2 = (track["x"] + track["w"]).values
        y2 = (track["y"] + track["h"]).values

        color = cmap(idx % 10)

        # Solid line = top-left (legend shows this)
        ax.plot(frames, x1, y1, color=color, linestyle="-", label=f"ID {tid}")
        # Dashed line = bottom-right (hidden from legend)
        ax.plot(frames, x2, y2, color=color, linestyle="--", label="_nolegend_")

        # Tunnel polygons
        verts = []
        for f, xx1, yy1, xx2, yy2 in zip(frames, x1, y1, x2, y2):
            rect = [(f, xx1, yy1), (f, xx2, yy1), (f, xx2, yy2), (f, xx1, yy2)]
            verts.append(rect)

        poly = Poly3DCollection(verts, alpha=0.01, facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)

    ax.set_xlabel("Frame")
    ax.set_ylabel("X coordinate")
    ax.set_zlabel("Y coordinate")
    ax.set_title(f"Track Stability (First {len(unique_ids)} IDs) – {video_name}")
    ax.legend(loc="upper right", fontsize=9, frameon=True, title="Track IDs")

    out_path = os.path.join(output_dir, f"{video_name}_ids={N}.png")
    plt.savefig(out_path, dpi=200)
    print(f"✅ Saved static 3D plot to {out_path}")

    # Show interactive rotatable plot
    plt.show()


def plot_track_stability_2D(annotation_file, N, output_dir="./plots/track_stability"):
    """
    Plot the first N track IDs' bounding box top-left and bottom-right
    coordinates in 2D (X vs Y), collapsing the time dimension so all
    coordinates are overlaid in the same image plane.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(annotation_file))[0]

    # Load MOT annotations
    cols = ["frame", "id", "x", "y", "w", "h", "score", "a1", "a2", "a3"]
    df = pd.read_csv(annotation_file, header=None, names=cols)

    # Keep only first N IDs
    unique_ids_all = sorted(df["id"].unique())
    if len(unique_ids_all) < N:
        print(f"⚠️ Only {len(unique_ids_all)} tracks available, plotting all of them.")
        unique_ids = unique_ids_all
    else:
        unique_ids = unique_ids_all[:N]

    df = df[df["id"].isin(unique_ids)]

    # Setup 2D plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for idx, tid in enumerate(unique_ids):
        track = df[df["id"] == tid]

        # Top-left coordinates
        x1 = track["x"].values
        y1 = track["y"].values

        # Bottom-right coordinates
        x2 = (track["x"] + track["w"]).values
        y2 = (track["y"] + track["h"]).values

        color = cmap(idx % 10)

        # Plot top-left as solid line
        ax.plot(x1, y1, marker="o", markersize=2, linestyle="-",
                color=color, alpha=0.8, label=f"ID {tid} (top-left)")

        # Plot bottom-right as dashed line
        ax.plot(x2, y2, marker="x", markersize=2, linestyle="--",
                color=color, alpha=0.8, label=f"ID {tid} (bottom-right)")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Track Stability (2D projection, First {len(unique_ids)} IDs) – {video_name}")
    ax.invert_yaxis()  # match image coordinate system

    # Add legend
    ax.legend(loc="upper right", fontsize=8, frameon=True, title="Track IDs")

    out_path = os.path.join(output_dir, f"{video_name}_ids={N}_2D.png")
    plt.savefig(out_path, dpi=200)
    print(f"✅ Saved 2D stability plot to {out_path}")

    plt.show()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Plot track stability in 3D.")
    # parser.add_argument("annotation_file", type=str, help="Path to MOT-style annotation .txt file")
    # parser.add_argument("N", type=int, help="Number of track IDs to plot")
    # args = parser.parse_args()
    # plot_track_stability(args.annotation_file, args.N)

    os.makedirs("./plots/track_stability", exist_ok=True)

    annotation_file = PROJECT_ROOT / "outputs" / "bytetrack_results" / "MOT17-02-DPM.txt"
    N = 5

    plot_track_stability_3D(str(annotation_file), N, output_dir="./plots/track_stability")
    plot_track_stability_2D(str(annotation_file), N, output_dir="./plots/track_stability")
