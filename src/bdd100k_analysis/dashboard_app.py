"""Streamlit dashboard for BDD100K analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hist_to_frame(hist: Dict[str, Any]) -> pd.DataFrame:
    """Convert a histogram dict into a DataFrame."""
    bins = hist.get("bins", [])
    counts = hist.get("counts", [])
    if not bins or not counts:
        return pd.DataFrame({"bin": [], "count": []})
    centers = []
    for i in range(len(counts)):
        centers.append((bins[i] + bins[i + 1]) / 2)
    return pd.DataFrame({"bin": centers, "count": counts})


def _class_distribution(summary: Dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame of per-class counts."""
    rows = []
    for class_name, stats in summary.get("classes", {}).items():
        rows.append(
            {
                "class": class_name,
                "boxes": stats.get("box_count", 0),
                "images": stats.get("image_count", 0),
            }
        )
    return pd.DataFrame(rows)


def _object_count_hist(summary: Dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame of object-count histogram buckets."""
    hist = summary.get("object_count_hist", {})
    rows = [{"bucket": k, "count": v} for k, v in hist.items()]
    return pd.DataFrame(rows).sort_values("bucket")


def _bucket_frame(
    buckets: Dict[str, int], label: str
) -> pd.DataFrame:
    """Convert bucket counts into a DataFrame with percentages."""
    total = sum(buckets.values())
    rows = []
    for name, count in buckets.items():
        pct = (count / total) * 100 if total else 0.0
        rows.append({label: name, "count": count, "pct": pct})
    return pd.DataFrame(rows)


def _render_summary(summary: Dict[str, Any], title: str) -> None:
    """Render summary charts and tables for a split."""
    st.subheader(title)
    st.write(
        {
            "images": summary.get("image_count", 0),
            "images_with_boxes": summary.get("images_with_boxes", 0),
            "total_boxes": summary.get("total_boxes", 0),
        }
    )

    class_df = _class_distribution(summary)
    if not class_df.empty:
        fig = px.bar(class_df, x="class", y="boxes", title="Boxes per Class")
        st.plotly_chart(fig, use_container_width=True)

    hist_df = _object_count_hist(summary)
    if not hist_df.empty:
        fig = px.bar(hist_df, x="bucket", y="count", title="Objects per Image")
        st.plotly_chart(fig, use_container_width=True)

    clutter = summary.get("clutter_levels", {})
    if clutter:
        clutter_df = _bucket_frame(clutter, "level")
        fig = px.bar(
            clutter_df, x="level", y="count", title="Clutter Levels"
        )
        st.plotly_chart(fig, use_container_width=True)

    size_buckets = summary.get("size_buckets", {})
    if size_buckets:
        size_df = _bucket_frame(size_buckets, "size")
        fig = px.bar(
            size_df, x="size", y="pct", title="Object Size Share (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    distance_buckets = summary.get("distance_buckets", {})
    if distance_buckets:
        distance_df = _bucket_frame(distance_buckets, "distance")
        fig = px.bar(
            distance_df,
            x="distance",
            y="pct",
            title="Distance Proxy (Box Height %)",
        )
        st.plotly_chart(fig, use_container_width=True)

    occlusion = summary.get("occlusion", {})
    if occlusion:
        st.write(
            {
                "occluded_pct": occlusion.get("occluded_pct", 0.0),
                "truncated_pct": occlusion.get("truncated_pct", 0.0),
            }
        )

    class_names = sorted(summary.get("classes", {}).keys())
    if class_names:
        selected = st.selectbox(
            "Inspect class", class_names, key=f"{title}-class"
        )
        class_stats = summary["classes"][selected]
        st.write(
            {
                "box_count": class_stats.get("box_count", 0),
                "image_count": class_stats.get("image_count", 0),
                "area_mean": class_stats.get("area", {}).get("mean"),
                "aspect_mean": class_stats.get("aspect", {}).get("mean"),
            }
        )
        class_occ = class_stats.get("occlusion", {})
        if class_occ:
            st.write(
                {
                    "occluded_pct": class_occ.get("occluded_pct", 0.0),
                    "truncated_pct": class_occ.get("truncated_pct", 0.0),
                }
            )
        class_size = class_stats.get("size_buckets", {})
        if class_size:
            class_size_df = _bucket_frame(class_size, "size")
            fig = px.bar(
                class_size_df,
                x="size",
                y="pct",
                title="Class Size Share (%)",
            )
            st.plotly_chart(fig, use_container_width=True)
        class_distance = class_stats.get("distance_buckets", {})
        if class_distance:
            class_distance_df = _bucket_frame(class_distance, "distance")
            fig = px.bar(
                class_distance_df,
                x="distance",
                y="pct",
                title="Class Distance Proxy (%)",
            )
            st.plotly_chart(fig, use_container_width=True)
        area_df = _hist_to_frame(class_stats.get("area", {}).get("hist", {}))
        if not area_df.empty:
            st.plotly_chart(
                px.line(area_df, x="bin", y="count", title="Area Histogram"),
                use_container_width=True,
            )
        aspect_df = _hist_to_frame(
            class_stats.get("aspect", {}).get("hist", {})
        )
        if not aspect_df.empty:
            st.plotly_chart(
                px.line(
                    aspect_df,
                    x="bin",
                    y="count",
                    title="Aspect Histogram",
                ),
                use_container_width=True,
            )


def _render_comparison(compare: Dict[str, Any]) -> None:
    """Render train/val comparison charts."""
    st.subheader("Train vs Val Comparison")
    class_rows = []
    for class_name, stats in compare.get("classes", {}).items():
        class_rows.append(
            {
                "class": class_name,
                "train_boxes": stats.get("train_boxes", 0),
                "val_boxes": stats.get("val_boxes", 0),
            }
        )
    df = pd.DataFrame(class_rows)
    if not df.empty:
        fig = px.bar(
            df,
            x="class",
            y=["train_boxes", "val_boxes"],
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_visuals(outputs_dir: Path, split: str) -> None:
    """Render TopK visual samples for a split."""
    visuals_dir = outputs_dir / split / "visuals"
    if not visuals_dir.exists():
        st.warning(f"No visuals found for {split} at {visuals_dir}")
        return

    st.subheader(f"{split.title()} TopK Samples")
    class_dirs = sorted([p for p in visuals_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        st.info("No class folders found under visuals.")
        return

    class_names = [p.name for p in class_dirs]
    selected_class = st.selectbox(
        f"{split.title()} class", class_names, key=f"{split}-visual-class"
    )

    candidate_paths = (
        list((visuals_dir / selected_class).glob("*.jpg"))
        + list((visuals_dir / selected_class).glob("*.png"))
        + list((visuals_dir / selected_class).glob("*.jpeg"))
    )
    if not candidate_paths:
        st.info("No images found for this class.")
        return

    groups = sorted({p.stem.split("_", 1)[0] for p in candidate_paths})
    selected_groups = st.multiselect(
        "Groups", groups, default=groups, key=f"{split}-visual-groups"
    )
    filtered_paths = [
        p
        for p in candidate_paths
        if p.stem.split("_", 1)[0] in selected_groups
    ]

    max_images = st.slider(
        "Max images to display", 6, 60, 24, 6, key=f"{split}-visual-limit"
    )
    display_paths = sorted(filtered_paths)[:max_images]

    cols = st.columns(3)
    for idx, path in enumerate(display_paths):
        cols[idx % 3].image(str(path), caption=path.name)


def main() -> None:
    """Streamlit app entrypoint."""
    st.title("BDD100K Detection Analysis")
    raw_outputs = st.sidebar.text_input("Outputs directory", "outputs")
    outputs_dir = Path(raw_outputs)
    if not outputs_dir.is_absolute():
        cwd_candidate = Path.cwd() / outputs_dir
        if cwd_candidate.exists():
            outputs_dir = cwd_candidate
        else:
            repo_root = Path(__file__).resolve().parents[2]
            repo_candidate = repo_root / outputs_dir
            if repo_candidate.exists():
                outputs_dir = repo_candidate
            else:
                outputs_dir = cwd_candidate

    train_summary = _load_json(outputs_dir / "train" / "summary.json")
    val_summary = _load_json(outputs_dir / "val" / "summary.json")
    compare = _load_json(outputs_dir / "compare.json")

    if train_summary:
        _render_summary(train_summary, "Train Split")
    else:
        st.warning("Train summary not found.")

    if val_summary:
        _render_summary(val_summary, "Val Split")
    else:
        st.warning("Val summary not found.")

    if compare:
        _render_comparison(compare)
    else:
        st.warning(
            f"Compare summary not found at {outputs_dir / 'compare.json'}"
        )

    _render_visuals(outputs_dir, "train")
    _render_visuals(outputs_dir, "val")


if __name__ == "__main__":
    main()
