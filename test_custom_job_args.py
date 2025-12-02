from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

try:
    op = CustomTrainingJobOp(
        project="p",
        location="l",
        display_name="d",
        worker_pool_specs=[],
        scheduling={"strategy": "SPOT"}
    )
    print("Success with scheduling")
except TypeError as e:
    print(f"Failed with scheduling: {e}")

try:
    op = CustomTrainingJobOp(
        project="p",
        location="l",
        display_name="d",
        worker_pool_specs=[],
        timeout="1s"
    )
    print("Success with timeout")
except TypeError as e:
    print(f"Failed with timeout: {e}")
