from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


# -------------------------
# DEFAULTS
# -------------------------
default_args = {
    "owner": "ml-team",
    "depends_on_past": True,   # 🚨 IMPORTANT
    "retries": 1,
}

# -------------------------
# DAG
# -------------------------
with DAG(
    dag_id="gmail_ml_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),  # arbitrary past date
    catchup=False,
    tags=["gmail", "ml", "pipeline"],
) as dag:

    # ======================================================
    # FULL PIPELINE (FIRST RUN + EVERY 3 MONTHS)
    # ======================================================
    full_pipeline = BashOperator(
        task_id="full_pipeline",
        bash_command="""
        export PIPELINE_MODE=retrain
        python -m src.pipelines.full_pipeline
        """,
        schedule_interval="0 2 1 */3 *",  # every 3 months
    )

    # ======================================================
    # INCREMENTAL PIPELINE (EVERY 3 DAYS)
    # ======================================================
    incremental_pipeline = BashOperator(
        task_id="incremental_pipeline",
        bash_command="""
        export PIPELINE_MODE=incremental
        python -m src.pipelines.incremental_pipeline
        """,
        schedule_interval="0 2 */3 * *",  # every 3 days
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ======================================================
    # CRITICAL DEPENDENCY
    # ======================================================
    full_pipeline >> incremental_pipeline