from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_utils import (
    activate_uploaded_dataset,
    append_dataset_history,
    get_data_layers_status,
    get_dataset_history,
    get_database_health,
    load_datasets,
    sync_processed_to_db,
    train_models_cached,
    write_etl_status,
)
from utils.ui import render_header, setup_page

setup_page("Admin Panel | Student Analytics", route_path="/Admin_Panel")

df, _ = load_datasets()

render_header(
    "Admin Panel",
    "System controls for dataset ingestion, ETL execution, model retraining, and dashboard refresh.",
)

db_ok, db_message = get_database_health()
if db_ok:
    st.success(f"MySQL: {db_message}")
else:
    st.warning(f"MySQL: {db_message}")

layer_status = get_data_layers_status()
layer_cols = st.columns(3)
layer_cols[0].metric("Bronze Files", layer_status["bronze"]["files"], layer_status["bronze"]["last_update"])
layer_cols[1].metric("Silver Files", layer_status["silver"]["files"], layer_status["silver"]["last_update"])
layer_cols[2].metric("Gold Files", layer_status["gold"]["files"], layer_status["gold"]["last_update"])

st.subheader("1) Upload New Dataset")
uploaded_file = st.file_uploader(
    "Upload a CSV or XLSX file",
    type=["csv", "xlsx"],
    help="Uploaded files are stored in data/raw/uploads.",
)

if uploaded_file is not None:
    target_dir = ROOT_DIR / "data" / "raw" / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / uploaded_file.name
    upload_col_1, upload_col_2 = st.columns(2)
    if upload_col_1.button("Save Uploaded Dataset"):
        save_path.write_bytes(uploaded_file.getbuffer())
        append_dataset_history(
            action="Upload Dataset",
            source=uploaded_file.name,
            status="Success",
            details={
                "saved_to": str(save_path),
                "size_kb": round(len(uploaded_file.getbuffer()) / 1024, 2),
            },
        )
        st.success(f"Uploaded file saved to: {save_path}")
    if upload_col_2.button("Save + Use Across All Pages"):
        save_path.write_bytes(uploaded_file.getbuffer())
        ok, message = activate_uploaded_dataset(save_path)
        if ok:
            append_dataset_history(
                action="Activate Uploaded Dataset",
                source=uploaded_file.name,
                status="Success",
                details={"message": message},
            )
            st.success(message)
            st.cache_data.clear()
            st.cache_resource.clear()
            synced, sync_message = sync_processed_to_db()
            if synced:
                append_dataset_history(
                    action="Sync to MySQL",
                    source="Processed CSV -> MySQL tables",
                    status="Success",
                    details={"message": sync_message},
                )
                st.success(sync_message)
            else:
                append_dataset_history(
                    action="Sync to MySQL",
                    source="Processed CSV -> MySQL tables",
                    status="Failed",
                    details={"message": sync_message},
                )
                st.warning(f"MySQL sync skipped: {sync_message}")
        else:
            append_dataset_history(
                action="Activate Uploaded Dataset",
                source=uploaded_file.name,
                status="Failed",
                details={"message": message},
            )
            st.error(message)

st.subheader("2) Trigger ETL Pipeline")
if st.button("Run Data Stitching + Feature Engineering"):
    src_dir = ROOT_DIR / "src"
    processed_file = ROOT_DIR / "data" / "processed" / "final_student_dataset.csv"
    commands = [
        [sys.executable, "data_stitching.py"],
        [sys.executable, "feature_engineering.py"],
        [sys.executable, "warehouse_pipeline.py"],
    ]
    logs: list[str] = []
    errors: list[str] = []
    old_count = 0
    if processed_file.exists():
        try:
            old_count = len(pd.read_csv(processed_file))
        except Exception:
            old_count = 0

    with st.spinner("Running ETL pipeline..."):
        for cmd in commands:
            process = subprocess.run(
                cmd,
                cwd=src_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            logs.append(f"$ {' '.join(cmd)}\n{process.stdout}")
            if process.returncode != 0:
                errors.append(f"$ {' '.join(cmd)}\n{process.stderr}")

    if errors:
        write_etl_status(new_records="0", errors=errors[0][:220], status="Failed")
        append_dataset_history(
            action="Run ETL Pipeline",
            source="data_stitching.py + feature_engineering.py + warehouse_pipeline.py",
            status="Failed",
            details={"error_count": len(errors), "old_records": old_count},
        )
        st.error("ETL pipeline finished with errors.")
        st.text_area("Error Log", "\n\n".join(errors), height=200)
    else:
        new_count = old_count
        if processed_file.exists():
            try:
                new_count = len(pd.read_csv(processed_file))
            except Exception:
                new_count = old_count
        new_records = max(new_count - old_count, 0)
        write_etl_status(new_records=new_records, errors="None", status="Success")
        append_dataset_history(
            action="Run ETL Pipeline",
            source="data_stitching.py + feature_engineering.py + warehouse_pipeline.py",
            status="Success",
            details={
                "old_records": old_count,
                "new_records": new_count,
                "added_records": new_records,
            },
        )
        st.success("ETL pipeline completed successfully.")
        st.caption(f"Records before run: {old_count} | after run: {new_count}")
        synced, sync_message = sync_processed_to_db()
        if synced:
            append_dataset_history(
                action="Sync to MySQL",
                source="Processed CSV -> MySQL tables",
                status="Success",
                details={"message": sync_message},
            )
            st.success(sync_message)
            st.cache_data.clear()
        else:
            append_dataset_history(
                action="Sync to MySQL",
                source="Processed CSV -> MySQL tables",
                status="Failed",
                details={"message": sync_message},
            )
            st.warning(f"MySQL sync skipped: {sync_message}")
    st.text_area("Execution Log", "\n\n".join(logs), height=220)

st.subheader("3) Retrain ML Models")
if st.button("Retrain Models"):
    train_models_cached.clear()
    with st.spinner("Training models on latest processed dataset..."):
        ml_results = train_models_cached(df)
    append_dataset_history(
        action="Retrain Models",
        source="ML models",
        status="Success",
        details={"rows_used": len(df)},
    )

    regression_metrics = ml_results.get("regression", {}).get("metrics", {})
    classification_metrics = ml_results.get("classification", {}).get("metrics", {})

    if regression_metrics:
        reg_table = pd.DataFrame(regression_metrics).T.reset_index()
        reg_table.rename(columns={"index": "Model"}, inplace=True)
        st.write("Regression Metrics")
        st.dataframe(reg_table, width="stretch", hide_index=True)

    if classification_metrics:
        clf_table = pd.DataFrame(classification_metrics).T.reset_index()
        clf_table.rename(columns={"index": "Model"}, inplace=True)
        st.write("Classification Metrics")
        st.dataframe(clf_table, width="stretch", hide_index=True)

st.subheader("4) Refresh Dashboards")
if st.button("Refresh Dashboard Caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    append_dataset_history(
        action="Refresh Caches",
        source="Streamlit cache",
        status="Success",
        details={"cache_data": "cleared", "cache_resource": "cleared"},
    )
    st.success("All dashboard caches cleared. Reload pages to view fresh values.")

st.subheader("5) Sync Processed Data to MySQL")
if st.button("Sync Now"):
    synced, sync_message = sync_processed_to_db()
    if synced:
        append_dataset_history(
            action="Sync to MySQL",
            source="Processed CSV -> MySQL tables",
            status="Success",
            details={"message": sync_message},
        )
        st.success(sync_message)
        st.cache_data.clear()
    else:
        append_dataset_history(
            action="Sync to MySQL",
            source="Processed CSV -> MySQL tables",
            status="Failed",
            details={"message": sync_message},
        )
        st.error(sync_message)

st.subheader("6) Data Source Simulation")
sim_col_1, sim_col_2, sim_col_3 = st.columns(3)
sim_batches = int(
    sim_col_1.number_input("Batches", min_value=1, max_value=20, value=3, step=1, key="sim_batches")
)
sim_batch_size = int(
    sim_col_2.number_input(
        "Batch Size", min_value=5, max_value=500, value=40, step=5, key="sim_batch_size"
    )
)
sim_interval = float(
    sim_col_3.number_input(
        "Interval Sec", min_value=0.0, max_value=30.0, value=1.0, step=0.5, key="sim_interval"
    )
)
if st.button("Run Stream Simulation"):
    src_dir = ROOT_DIR / "src"
    command = [
        sys.executable,
        "data_simulation.py",
        "--batches",
        str(sim_batches),
        "--batch-size",
        str(sim_batch_size),
        "--interval-seconds",
        str(sim_interval),
    ]
    process = subprocess.run(
        command,
        cwd=src_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 0:
        append_dataset_history(
            action="Run Data Simulation",
            source="data_simulation.py",
            status="Success",
            details={
                "batches": sim_batches,
                "batch_size": sim_batch_size,
                "interval_seconds": sim_interval,
            },
        )
        st.success("Data simulation completed.")
        st.text_area("Simulation Log", process.stdout, height=180)
    else:
        append_dataset_history(
            action="Run Data Simulation",
            source="data_simulation.py",
            status="Failed",
            details={"stderr": process.stderr[:220]},
        )
        st.error("Data simulation failed.")
        st.text_area("Simulation Error", process.stderr, height=180)

st.subheader("7) Build Warehouse Layers")
if st.button("Build Bronze/Silver/Gold Tables"):
    src_dir = ROOT_DIR / "src"
    command = [sys.executable, "warehouse_pipeline.py"]
    process = subprocess.run(
        command,
        cwd=src_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 0:
        append_dataset_history(
            action="Build Warehouse Layers",
            source="warehouse_pipeline.py",
            status="Success",
            details={"output": "silver and gold tables refreshed"},
        )
        st.success("Warehouse layers updated successfully.")
        st.text_area("Warehouse Log", process.stdout, height=180)
    else:
        append_dataset_history(
            action="Build Warehouse Layers",
            source="warehouse_pipeline.py",
            status="Failed",
            details={"stderr": process.stderr[:220]},
        )
        st.error("Warehouse build failed.")
        st.text_area("Warehouse Error", process.stderr, height=180)

st.subheader("8) Dataset History")
history_df = get_dataset_history(limit=500)
if history_df.empty:
    st.info("No dataset history yet. Upload, ETL, or sync actions will appear here.")
else:
    filter_col_1, filter_col_2, filter_col_3 = st.columns([2, 2, 1])
    action_options = ["All"] + sorted(history_df["Action"].dropna().unique().tolist())
    status_options = ["All"] + sorted(history_df["Status"].dropna().unique().tolist())

    selected_action = filter_col_1.selectbox("Filter by Action", action_options)
    selected_status = filter_col_2.selectbox("Filter by Status", status_options)
    row_limit = int(
        filter_col_3.number_input("Rows", min_value=10, max_value=500, value=100, step=10)
    )

    filtered = history_df.copy()
    if selected_action != "All":
        filtered = filtered[filtered["Action"] == selected_action]
    if selected_status != "All":
        filtered = filtered[filtered["Status"] == selected_status]

    st.dataframe(filtered.head(row_limit), width="stretch", hide_index=True)
    st.download_button(
        "Download History CSV",
        data=filtered.head(row_limit).to_csv(index=False).encode("utf-8"),
        file_name="dataset_history.csv",
        mime="text/csv",
    )

st.warning(
    "Admin actions modify runtime state and can overwrite processed outputs. "
    "Use after validating raw data quality."
)

