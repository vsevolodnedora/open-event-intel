from datetime import datetime, timedelta
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.data_models import Publication

# Make sure PostsDatabase is imported or defined in this module
from src.publications_database import PostsDatabase
from src.tkg.tkg_utls import count_tokens

# map the key names from visualize_posts_heatmap to their proper names, e.g., "entsoe":"ENTSO-E"
name_mapping: Dict[str, str] = {
    "entsoe": "ENTSO-E",
    "eex": "EEX",
    "acer": "ACER",
    "ec": "European Commission",
    "icis": "ICIS",
    "bnetza": "BNetzA",
    "smard": "SMARD",
    "agora": "Agora Energiewende",
    "energy_wire": "Clean Energy Wire",
    "transnetbw": "TransnetBW",
    "tennet": "TenneT",
    "50hz": "50Hertz",
    "amprion": "Amprion"
}

def print_publications_stats(
    table_names: List[str],
    db_path: str = "../database/preprocessed_posts.db",
    n_past_days:int = 30
) -> None:
    """
    For each table in table_names, fetch all post dates from the past 30 days.

    Draw a GitHub-style commit heatmap: a 30-column grid (most recent day on the right),
    one row per table, with color intensity = post count on that day.
    """
    # --- 1) open database and collect counts per day ---
    db = PostsDatabase(db_path)
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in reversed(range(n_past_days))]
    date_to_idx = {d: idx for idx, d in enumerate(days)}
    n_tables = len(table_names)
    counts = np.zeros((n_tables, n_past_days), dtype=int)

    for i, tbl in enumerate(table_names):
        if not db.is_table(tbl):
            raise ValueError(f"Table '{tbl}' does not exist in {db_path}")
        all_dates = db.get_all_publication_dates(tbl)
        all_publications:list[Publication] = db.list_publications(tbl, sort_date=True)

        rows = []
        for publication in all_publications:
            text = publication.text
            length = len(text)
            tokens = count_tokens(text, model="gpt-4.1")

            # keep your date → count update
            try:
                d = publication.published_on.date()
                if d in date_to_idx:
                    counts[i, date_to_idx[d]] += 1
            except Exception:
                print(f"Could not extract date from {publication}")
                pass  # ignore bad/missing dates

            rows.append((length, tokens, publication.published_on, publication.title))

        # print top 5 by length
        for length, tokens, pub_on, title in sorted(rows, key=lambda r: r[0], reverse=True)[:5]:
            print(f"{pub_on}__{title} ({length}, {tokens})")

    db.close()

def visualize_posts_heatmap(
    table_names: List[str],
    db_path: str = "../database/preprocessed_posts.db",
    n_past_days:int = 30
) -> None:
    """
    For each table in table_names, fetch all post dates from the past 30 days.

    Draw a GitHub-style commit heatmap: a 30-column grid (most recent day on the right),
    one row per table, with color intensity = post count on that day.
    """
    # --- 1) open database and collect counts per day ---
    db = PostsDatabase(db_path)
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in reversed(range(n_past_days))]
    date_to_idx = {d: idx for idx, d in enumerate(days)}
    n_tables = len(table_names)
    counts = np.zeros((n_tables, n_past_days), dtype=int)

    for i, tbl in enumerate(table_names):
        if not db.is_table(tbl):
            raise ValueError(f"Table '{tbl}' does not exist in {db_path}")
        all_dates = db.get_all_publication_dates(tbl)
        all_publications:list[Publication] = db.list_publications(tbl, sort_date=True)
        """                "ID": pid,
                "published_on": pub_dt.isoformat() if isinstance(pub_dt, datetime) else str(pub_dt),
                "title": title,
                "added_on": add_dt.isoformat() if isinstance(add_dt, datetime) else str(add_dt),
                "url": url,
                "post": text,"""
        print(tbl, end=": ")

        lengths = []
        for publication in all_publications:
            lengths.append(str(len(publication.text)))
            d = publication.published_on.date()
            if d in date_to_idx:
                counts[i, date_to_idx[d]] += 1

        print(", ".join(lengths))
    db.close()

    # --- 2) determine color thresholds ---
    colors = ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"]
    non_zero = counts[counts > 0]
    if non_zero.size:
        q1, q2, q3 = np.percentile(non_zero, [25, 50, 75])
    else:
        q1 = q2 = q3 = 0

    def pick_color(c: int) -> str:
        if c == 0:
            return colors[0]
        elif c <= q1:
            return colors[1]
        elif c <= q2:
            return colors[2]
        elif c <= q3:
            return colors[3]
        else:
            return colors[4]

    color_grid = np.empty_like(counts, dtype=object)
    for i in range(n_tables):
        for j in range(n_past_days):
            color_grid[i, j] = pick_color(counts[i, j])

    # --- 3) draw the heatmap grid ---
    # Use constrained layout to let Matplotlib compute exact space for rotated tick labels.
    fig, ax = plt.subplots(
        figsize=(12, n_tables * 0.45 + 0.6),  # slightly tighter base height
        constrained_layout=True
    )

    for i in range(n_tables):
        for j in range(n_past_days):
            rect = Rectangle((j, i), 0.7, 0.7, facecolor=color_grid[i, j], edgecolor='none')
            ax.add_patch(rect)

    # axes formatting
    ax.set_xlim(0, n_past_days)
    ax.set_ylim(0, n_tables)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_past_days) + 0.5)

    # Generate the tick labels with conditional formatting
    tick_labels = []
    for i, d in enumerate(days):
        if i == 0 or d.month != days[i - 1].month:  # First day of a new month
            tick_labels.append(d.strftime("%b-%d"))
        else:
            tick_labels.append(d.strftime("%d"))

    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_tables) + 0.5)
    ax.set_yticklabels([name_mapping[key] for key in table_names])

    # Tighten tick/label spacing without clipping
    ax.tick_params(bottom=False, left=False, labelsize=10, pad=1)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title with minimal padding to reduce top whitespace
    # ax.set_title("Posts per Day (Last n_past_days Days)", pad=2)

    # IMPORTANT: do NOT call plt.tight_layout() together with constrained layout.
    # Instead, set minimal global pads so the figure hugs the axes+labels.
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)

    # Save with a tight bounding box (keeps tick labels) and a tiny extra pad
    fig.savefig("./figs/posts_per_day_heatmap.png", bbox_inches="tight", pad_inches=0.08, dpi=600)

    plt.show()


print_publications_stats([
    'entsoe', 'eex', 'acer', 'ec', 'icis', 'bnetza', 'smard', 'agora', 'energy_wire', 'transnetbw', 'tennet', '50hz', 'amprion'
])

visualize_posts_heatmap([
    'entsoe', 'eex', 'acer', 'ec', 'icis', 'bnetza', 'smard', 'agora', 'energy_wire', 'transnetbw', 'tennet', '50hz', 'amprion'
])

