"""Render the README figures from the public aggregate CSVs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
BASELINE = "Phase 1F"
ABLATION = "Experiment A"
LABELS = ("NETC", "SNFH", "ET", "RC")

PAPER = "#F7F9FD"
WHITE = "#FFFFFF"
INK = "#13233B"
MUTED = "#50637D"
GRID = "#DBE3F1"
BLUE = "#214FB7"
PURPLE = "#7339AB"
TEAL = "#08756E"
ORANGE = "#AA431F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_table(filename):
    with (RESULTS / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def canvas(title, subtitle, *, height=4.25):
    fig = plt.figure(figsize=(10, height), facecolor=PAPER)
    fig.text(0.07, 0.94, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.07, 0.85, subtitle, fontsize=10, color=MUTED, va="top")
    return fig


def finish(fig, name, note):
    fig.text(0.07, 0.065, note, fontsize=9, color=MUTED, va="center")
    fig.savefig(FIGURES / name, dpi=165, facecolor=PAPER)
    plt.close(fig)


def grid(ax, axis):
    ax.set_facecolor(WHITE)
    ax.grid(axis=axis, color=GRID, linewidth=0.75)
    ax.tick_params(length=0, pad=7)


def class_dice(by_run):
    fig = canvas("Segmentation quality varies by label",
                 "Phase 1F  /  Fold-0 validation  /  Active-case Dice on full MRI volumes")
    ax = fig.add_axes((0.09, 0.23, 0.82, 0.50))
    rows = [by_run[BASELINE, label] for label in LABELS]
    stats = [
        {
            "whislo": float(row["minimum"]),
            "q1": float(row["q1"]),
            "med": float(row["median"]),
            "q3": float(row["q3"]),
            "whishi": float(row["maximum"]),
            "mean": float(row["mean_dice"]),
        }
        for row in rows
    ]
    boxes = ax.bxp(stats, showfliers=False, showmeans=True, patch_artist=True, widths=0.47,
                   medianprops={"color": INK, "linewidth": 1.9},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                   meanprops={"marker": "D", "markerfacecolor": BLUE, "markeredgecolor": WHITE, "markersize": 7})
    for patch, color in zip(boxes["boxes"], ("#E7D5F5", "#CCDEF7", "#CCDEF7", "#E7D5F5")):
        patch.set(facecolor=color, edgecolor=MUTED, linewidth=1.0)
    ax.set(ylim=(-0.025, 1.035), ylabel="Dice")
    ax.set_xticks(range(1, 5), [f"{label}  /  n={row['n_cases']}" for label, row in zip(LABELS, rows)])
    grid(ax, "y")
    finish(fig, "class-dice.png", "Diamond = mean. Box = interquartile range. Whiskers = observed minimum and maximum.")


def foreground_distribution(by_run):
    fig = canvas("Foreground overlap in 96 validation cases",
                 "Phase 1F  /  Fold 0  /  Foreground is the union of NETC, SNFH, ET and RC")
    ax = fig.add_axes((0.09, 0.23, 0.82, 0.50))
    hist = read_table("foreground_dice_histogram.csv")
    starts = [float(row["lower_dice"]) for row in hist]
    widths = [float(row["upper_dice"]) - start for row, start in zip(hist, starts)]
    counts = [int(row["case_count"]) for row in hist]
    ax.bar(starts, counts, width=widths, align="edge", color=BLUE, edgecolor=WHITE, linewidth=0.7)
    metrics = by_run[BASELINE, "Foreground"]
    mean, median = float(metrics["mean_dice"]), float(metrics["median"])
    ax.axvline(mean, color=ORANGE, linewidth=2, linestyle="--", label=f"Mean {mean:.4f}")
    ax.set(xlim=(0, 1.005), ylim=(0, max(counts)*1.13), xlabel="Foreground Dice", ylabel="Cases / bin")
    grid(ax, "y")
    ax.legend(loc="upper left", frameon=False)
    finish(fig, "foreground-dice.png", f"Median {median:.4f}  /  Histogram bins are 0.05 wide. Cases are timepoints from 31 patients.")


def trailing_mean(values, window=25):
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    sums = np.r_[0.0, np.cumsum(values)]
    return (sums[ends] - sums[starts]) / (ends - starts)


def training_curves():
    fig = canvas("What the training monitored",
                 "1,000 epochs per completed run  /  Faint traces are logged values, solid traces are 25-epoch means",
                 height=4.6)
    loss_ax = fig.add_axes((0.09, 0.23, 0.38, 0.47))
    dice_ax = fig.add_axes((0.56, 0.23, 0.38, 0.47))
    rows = read_table("training_curves.csv")

    def draw(ax, run, field, color, label):
        selected = sorted((row for row in rows if row["run"] == run), key=lambda row: int(row["epoch_index"]))
        epoch = [int(row["epoch_index"]) + 1 for row in selected]
        values = [float(row[field]) for row in selected]
        ax.plot(epoch, values, color=color, alpha=0.18, linewidth=0.6)
        ax.plot(epoch, trailing_mean(values), color=color, linewidth=1.85, label=label)

    draw(loss_ax, BASELINE, "train_loss", BLUE, "Train")
    draw(loss_ax, BASELINE, "val_loss", PURPLE, "Online validation")
    draw(dice_ax, BASELINE, "mean_pseudo_dice", BLUE, BASELINE)
    draw(dice_ax, ABLATION, "mean_pseudo_dice", TEAL, ABLATION)
    for ax, title, label in ((loss_ax, "Phase 1F loss", "Compound loss"),
                            (dice_ax, "Online pseudo Dice", "Mean of 4 labels")):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=9)
        ax.set(xlim=(1, 1000), xlabel="Epoch", ylabel=label)
        ax.set_xticks([1, 250, 500, 750, 1000])
        grid(ax, "y")
        ax.legend(loc="best", fontsize=8, frameon=False)
    finish(fig, "training-curves.png", "Online pseudo Dice is a patch-based monitor. Full-volume scores use a separate inference pass.")


def experiment_delta(by_run):
    fig = canvas("Experiment A improved foreground overlap slightly",
                 "Change in mean Dice from Phase 1F to Experiment A on the same 96 validation cases",
                 height=4.4)
    ax = fig.add_axes((0.22, 0.23, 0.67, 0.5))
    ordered = ("Foreground", *LABELS)
    deltas = [100 * (float(by_run[ABLATION, label]["mean_dice"]) -
                     float(by_run[BASELINE, label]["mean_dice"])) for label in ordered]
    colors = [TEAL if delta >= 0 else ORANGE for delta in deltas]
    indices = list(range(len(ordered)))
    ax.barh(indices, deltas, color=colors, height=0.52)
    ax.axvline(0, linewidth=1, color=MUTED)
    ax.set_yticks(indices, ordered)
    ax.invert_yaxis()
    ax.set(xlim=(-2.85, 0.7), xlabel="Dice change (percentage points)")
    grid(ax, "x")
    for i, value, color in zip(indices, deltas, colors):
        offset = 0.055 if value >= 0 else -0.055
        ax.text(value + offset, i, f"{value:+.2f}", color=color, fontsize=9,
                va="center", ha="left" if value >= 0 else "right", fontweight="bold")
    finish(fig, "experiment-a-delta.png", "NETC and RC Dice declined. Foreground volume MAE also increased from 2.14 to 2.35 mL.")


def main():
    FIGURES.mkdir(exist_ok=True)
    by_run = {(row["run"], row["label"]): row for row in read_table("validation_dice.csv")}
    class_dice(by_run)
    foreground_distribution(by_run)
    training_curves()
    experiment_delta(by_run)


if __name__ == "__main__":
    main()
