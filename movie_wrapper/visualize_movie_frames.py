import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movie_wrapper.wrapper import wrapper_dvd, inverse_wrapper_dvd
from ML_framework.nwb_loading.nwb_loading import load_nwb


def get_all_annotations(nwb_data):
    """
    Return a dict mapping label_name -> indicator_function (np.ndarray).
    The indicator is 1-based frame indexed (index 0 corresponds to frame 1).
    """
    iface = nwb_data.processing["machine_learning"].data_interfaces[
        "movie_annotations_indicator_functions"
    ]
    df = iface.to_dataframe()
    annotations = {}
    for _, row in df.iterrows():
        label_name = row["label_name"]
        indicator = row["indicator_function"]
        if hasattr(indicator, "__len__") and not isinstance(indicator, (str, bytes)):
            indicator = np.asarray(indicator).ravel()
        else:
            indicator = np.asarray([indicator])
        annotations[label_name] = indicator
    return annotations


def visualize_frame_with_annotations(
    frames_dir,
    dvd_frame_number,
    patient,
    data_dir=None,
    save_path=None,
):
    """
    Visualize a DVD movie frame alongside its annotation labels.

    Parameters:
        frames_dir (str | Path): Directory containing frames named frame_{number}.jpg
        dvd_frame_number (int): Frame number in DVD numbering.
        patient (str): Patient ID for loading the NWB file.
        data_dir (str | Path, optional): Directory containing NWB files.
        save_path (str | Path, optional): If provided, save the figure to this path.

    Returns:
        matplotlib.figure.Figure
    """
    frames_dir = Path(frames_dir)

    # Load the movie frame image
    frame_path = frames_dir / f"frame_{dvd_frame_number:06d}.jpg"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_path}")
    frame_img = imread(str(frame_path))

    # Convert DVD frame number to paradigm frame number
    paradigm_frame = inverse_wrapper_dvd(dvd_frame_number)

    # Load NWB data and get annotations
    nwb_data, _ = load_nwb(patient, data_dir=data_dir)
    annotations = get_all_annotations(nwb_data)

    # Get annotation values for this frame (indicator is 0-indexed, frame numbers are 1-based)
    frame_idx = paradigm_frame - 1
    label_names = sorted(annotations.keys())
    label_values = []
    for name in label_names:
        indicator = annotations[name]
        if frame_idx < len(indicator):
            label_values.append(indicator[frame_idx] > 0)
        else:
            label_values.append(False)

    # Build figure: movie frame on left, dense annotation table on right (matched height)
    n_labels = len(label_names)
    n_cols = 4
    n_rows = int(np.ceil(n_labels / n_cols))

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05, top=0.88)

    # Left panel: movie frame
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(frame_img)
    ax_img.axis("off")

    # Right panel: dense annotation table using matplotlib table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    # Place both titles at the same y, image title centered over image axes
    title_y = 0.88
    img_center = ax_img.get_position().x0 + ax_img.get_position().width / 2
    table_center = ax_table.get_position().x0 + ax_table.get_position().width / 2
    fig.text(img_center, title_y, f"DVD Frame {dvd_frame_number} (Paradigm Frame {paradigm_frame})",
             ha="center", fontsize=10)
    fig.text(table_center, title_y, "Annotations",
             ha="center", fontsize=10)

    cell_text = []
    cell_colors = []
    for row_idx in range(n_rows):
        row_text = []
        row_colors = []
        for col_idx in range(n_cols):
            i = row_idx * n_cols + col_idx
            if i < n_labels:
                row_text.append(label_names[i])
                if label_values[i]:
                    row_colors.append("#c8f7c5")
                else:
                    row_colors.append("white")
            else:
                row_text.append("")
                row_colors.append("white")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax_table.table(
        cellText=cell_text,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        i = row_idx * n_cols + col_idx
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if i < n_labels and label_values[i]:
            cell.get_text().set_color("#1a7a1a")
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_color("black")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize a DVD movie frame with NWB annotation labels."
    )
    parser.add_argument("frames_dir", type=str, help="Directory containing frame_*.jpg files")
    parser.add_argument("dvd_frame_number", type=int, help="Frame number in DVD numbering")
    parser.add_argument("patient", type=str, help="Patient ID for NWB file lookup")
    parser.add_argument("--data-dir", type=str, default=None, help="NWB data directory")
    parser.add_argument("--save", type=str, default=None, help="Path to save the output image")

    args = parser.parse_args()

    fig = visualize_frame_with_annotations(
        frames_dir=args.frames_dir,
        dvd_frame_number=args.dvd_frame_number,
        patient=args.patient,
        data_dir=args.data_dir,
        save_path=args.save,
    )
    if not args.save:
        plt.show()
