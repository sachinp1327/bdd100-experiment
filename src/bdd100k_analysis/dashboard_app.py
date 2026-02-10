"""Streamlit dashboard for BDD100K analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hist_to_frame(hist: Dict[str, Any]) -> pd.DataFrame:
    bins = hist.get("bins", [])
    counts = hist.get("counts", [])
    if not bins or not counts:
        return pd.DataFrame({"bin": [], "count": []})
    centers = []
    for i in range(len(counts)):
        centers.append((bins[i] + bins[i + 1]) / 2)
    return pd.DataFrame({"bin": centers, "count": counts})


def _class_distribution(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for class_name, stats in summary.get("classes", {}).items():
        rows.append({
            "class": class_name,
            "boxes": stats.get("box_count", 0),
            "images": stats.get("image_count", 0),
        })
    return pd.DataFrame(rows)


def _object_count_hist(summary: Dict[str, Any]) -> pd.DataFrame:
    hist = summary.get("object_count_hist", {})
    rows = [{"bucket": k, "count": v} for k, v in hist.items()]
    return pd.DataFrame(rows).sort_values("bucket")


def _render_summary(summary: Dict[str, Any], title: str) -> None:
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

    class_names = sorted(summary.get("classes", {}).keys())
    if class_names:
        selected = st.selectbox("Inspect class", class_names, key=f"{title}-class")
        class_stats = summary["classes"][selected]
        st.write(
            {
                "box_count": class_stats.get("box_count", 0),
                "image_count": class_stats.get("image_count", 0),
                "area_mean": class_stats.get("area", {}).get("mean"),
                "aspect_mean": class_stats.get("aspect", {}).get("mean"),
            }
        )
        area_df = _hist_to_frame(class_stats.get("area", {}).get("hist", {}))
        if not area_df.empty:
            st.plotly_chart(
                px.line(area_df, x="bin", y="count", title="Area Histogram"),
                use_container_width=True,
            )
        aspect_df = _hist_to_frame(class_stats.get("aspect", {}).get("hist", {}))
        if not aspect_df.empty:
            st.plotly_chart(
                px.line(aspect_df, x="bin", y="count", title="Aspect Histogram"),
                use_container_width=True,
            )


def _render_comparison(compare: Dict[str, Any]) -> None:
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
        fig = px.bar(df, x="class", y=["train_boxes", "val_boxes"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)


def _render_visuals(outputs_dir: Path) -> None:
    visuals_dir = outputs_dir / "visuals"
    if not visuals_dir.exists():
        return
    st.subheader("Sample Visuals")
    image_paths = list(visuals_dir.rglob("*.jpg"))[:30]
    for path in image_paths:
        st.image(str(path), caption=str(path.relative_to(outputs_dir)))


def main() -> None:
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
        st.warning(f"Compare summary not found at {outputs_dir / 'compare.json'}")

    _render_visuals(outputs_dir)


if __name__ == "__main__":
    main()
