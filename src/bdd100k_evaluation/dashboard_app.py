"""Streamlit dashboard for evaluation diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON if it exists."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_samples(path: Path) -> Dict[str, Any]:
    """Load sample metadata from disk."""
    return _load_json(path)


def _resolve_rendered_path(
    rendered_path: str | None, output_dir: Path
) -> Path | None:
    """Resolve rendered image path relative to output dir."""
    if not rendered_path:
        return None
    path = Path(rendered_path)
    if path.is_absolute() and path.exists():
        return path
    candidate = (output_dir / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists():
        return candidate
    if "renders" in path.parts:
        idx = path.parts.index("renders")
        suffix = Path(*path.parts[idx:])
        candidate = (output_dir / suffix).resolve()
        if candidate.exists():
            return candidate
    return None


def _parse_args() -> argparse.Namespace:
    """Parse optional CLI args passed after `--` in streamlit."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_diagnostics",
        help="Diagnostics output directory.",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """Entry point for Streamlit app."""
    st.set_page_config(page_title="BDD100K Eval Diagnostics", layout="wide")
    st.title("BDD100K Evaluation Diagnostics")

    args = _parse_args()
    output_dir = Path(
        st.sidebar.text_input(
            "Diagnostics output directory",
            args.output_dir,
        )
    ).resolve()

    overall = _load_json(output_dir / "overall_metrics.json")
    per_class = _load_json(output_dir / "per_class_metrics.json")
    attribute_metrics = _load_json(output_dir / "attribute_metrics.json")
    samples = _load_samples(output_dir / "samples.json")
    analysis_links = _load_json(output_dir / "analysis_links.json")

    if not overall:
        st.warning("Diagnostics output not found. Run the diagnostics CLI first.")
        return

    st.subheader("Overall Metrics")
    cols = st.columns(6)
    cols[0].metric("AP", f"{overall.get('AP', 0.0):.3f}")
    cols[1].metric("AP50", f"{overall.get('AP50', 0.0):.3f}")
    cols[2].metric("AP75", f"{overall.get('AP75', 0.0):.3f}")
    cols[3].metric("AP_small", f"{overall.get('AP_small', 0.0):.3f}")
    cols[4].metric("AP_medium", f"{overall.get('AP_medium', 0.0):.3f}")
    cols[5].metric("AP_large", f"{overall.get('AP_large', 0.0):.3f}")

    st.subheader("Per-Class Performance")
    class_df = pd.DataFrame.from_dict(per_class, orient="index").reset_index()
    class_df.rename(columns={"index": "class"}, inplace=True)
    class_df.sort_values("AP", ascending=False, inplace=True)

    metric = st.selectbox(
        "Per-class metric",
        options=[
            "AP",
            "AP50",
            "AP75",
            "AP_small",
            "AP_medium",
            "AP_large",
        ],
        index=0,
    )
    fig = px.bar(
        class_df,
        x="class",
        y=metric,
        title=f"{metric} by Class",
        color=metric,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(class_df, use_container_width=True)

    st.subheader("Best vs Worst Classes")
    best = class_df.head(3)
    worst = class_df.tail(3)
    st.write("Best classes (by AP):")
    st.table(best[["class", "AP", "precision", "recall"]])
    st.write("Worst classes (by AP):")
    st.table(worst[["class", "AP", "precision", "recall"]])

    st.subheader("Attribute Breakdown")
    attribute = st.selectbox(
        "Attribute",
        options=["timeofday", "weather", "scene"],
        index=0,
    )
    class_name = st.selectbox(
        "Class",
        options=class_df["class"].tolist(),
        index=0,
    )
    attr_values = attribute_metrics.get(attribute, {}).get(class_name, {})
    if attr_values:
        attr_df = pd.DataFrame.from_dict(attr_values, orient="index").reset_index()
        attr_df.rename(columns={"index": attribute}, inplace=True)
        fig_attr = px.bar(
            attr_df,
            x=attribute,
            y="f1",
            title=f"F1 by {attribute} for {class_name}",
            color="f1",
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        st.dataframe(attr_df, use_container_width=True)
    else:
        st.info("No attribute metrics available for this class.")

    st.subheader("Best/Worst Samples")
    sample_class = st.selectbox(
        "Sample class",
        options=list(samples.keys()) if samples else [],
    )
    sample_group = st.radio(
        "Group",
        options=["best", "worst"],
        horizontal=True,
    )
    if sample_class and samples:
        sample_items = samples.get(sample_class, {}).get(sample_group, [])
        if not sample_items:
            st.info("No samples available for this class/group.")
            return
        max_items = len(sample_items)
        default_k = min(5, max_items)
        k = st.slider(
            "Samples to show",
            min_value=1,
            max_value=max_items,
            value=default_k,
        )
        st.caption(
            "To show more samples, rerun diagnostics with a higher "
            "`--top-k` value."
        )
        for item in sample_items[:k]:
            st.write(
                f"Image: {item.get('file_name')} | "
                f"F1: {item.get('f1', 0.0):.3f} | "
                f"TP: {item.get('tp')} FP: {item.get('fp')} FN: {item.get('fn')}"
            )
            rendered_path = _resolve_rendered_path(
                item.get("rendered_path"), output_dir
            )
            if rendered_path:
                st.image(str(rendered_path), use_column_width=True)
            else:
                st.info("Rendered image not found on disk.")

    st.subheader("Links to Data Analysis")
    if analysis_links:
        analysis_df = pd.DataFrame(analysis_links)
        fig_link = px.scatter(
            analysis_df,
            x="box_count",
            y="AP",
            size="image_count",
            color="class",
            title="AP vs Class Frequency",
        )
        st.plotly_chart(fig_link, use_container_width=True)
        st.dataframe(analysis_df, use_container_width=True)
    else:
        st.info("Analysis summary not available.")


if __name__ == "__main__":
    main()
