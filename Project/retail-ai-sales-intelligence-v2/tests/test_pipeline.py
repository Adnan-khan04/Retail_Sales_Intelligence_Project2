import json
from pathlib import Path
from src.retail_ai.pipeline import run_pipeline
from src.retail_ai.data import default_data_path


def test_pipeline_writes_manifest_and_outputs(tmp_path):
    result = run_pipeline(
        input_path=default_data_path(),
        output_dir=tmp_path,
        forecast_products=1,
        forecast_horizon=7,
        backtest_days=7,
        n_clusters=3,
    )
    manifest_path = Path(tmp_path) / "pipeline_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert (Path(tmp_path) / "sales_summary.json").exists()
    assert (Path(tmp_path) / "customer_segment_profiles.csv").exists()
    assert (Path(tmp_path) / "inventory_plan.csv").exists()
