import os
import kfp
from kfp import dsl
from kfp.compiler import Compiler
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

# Global configuration for accelerator type
ACCELERATOR_TYPE = 'NVIDIA_L4'
# Read image URI from environment variable. 
# This allows compiling the pipeline with a specific image without passing it as a PipelineParam,
# which avoids issues with dsl.ContainerSpec.
DOCKER_IMAGE_URI = os.environ.get("DOCKER_IMAGE_URI", "gcr.io/nzp-nanochat/nanochat:latest")

@dsl.container_component
def tokenizer_step(gcs_bucket: str) -> dsl.ContainerSpec:
    """
    Tokenizer component.
    """
    return dsl.ContainerSpec(
        image=DOCKER_IMAGE_URI,
        command=["python", "vertex_pipelines/tokenizer_step.py"],
        args=["--gcs-bucket", gcs_bucket],
    )



@dsl.container_component
def midtraining_step(gcs_bucket: str, wandb_run: str, vertex_experiment: str, vertex_tensorboard: str) -> dsl.ContainerSpec:
    """
    Midtraining component.
    """
    return dsl.ContainerSpec(
        image=DOCKER_IMAGE_URI,
        command=["python", "vertex_pipelines/midtraining_step.py"],
        args=["--gcs-bucket", gcs_bucket, "--wandb-run", wandb_run, "--vertex-experiment", vertex_experiment, "--vertex-tensorboard", vertex_tensorboard],
    )

@dsl.container_component
def sft_step(gcs_bucket: str, wandb_run: str, vertex_experiment: str, vertex_tensorboard: str) -> dsl.ContainerSpec:
    """
    SFT component.
    """
    return dsl.ContainerSpec(
        image=DOCKER_IMAGE_URI,
        command=["python", "vertex_pipelines/sft_step.py"],
        args=["--gcs-bucket", gcs_bucket, "--wandb-run", wandb_run, "--vertex-experiment", vertex_experiment, "--vertex-tensorboard", vertex_tensorboard],
    )

@dsl.container_component
def data_download_step(gcs_bucket: str, num_shards: int = 50):
    """
    Data download component - downloads training data from HuggingFace to GCS.
    """
    return dsl.ContainerSpec(
        image=DOCKER_IMAGE_URI,
        command=["python", "vertex_pipelines/data_download_step.py"],
        args=["--gcs-bucket", gcs_bucket, "--num-shards", str(num_shards)],
    )

@dsl.container_component
def report_step(gcs_bucket: str) -> dsl.ContainerSpec:
    """
    Report component.
    """
    return dsl.ContainerSpec(
        image=DOCKER_IMAGE_URI,
        command=["python", "vertex_pipelines/report_step.py"],
        args=["--gcs-bucket", gcs_bucket],
    )



# Let's rewrite the function to use the global ACCELERATOR_TYPE which we will ensure is set BEFORE the function is decorated/called.
# Actually, dsl.pipeline is a decorator. It runs when the module is loaded.
# So 'nanochat_pipeline' is compiled/registered immediately.
# If we want to change the structure based on args, we should define the pipeline function INSIDE __main__ or 
# create a function that returns the pipeline function.

def create_pipeline_func(accelerator_type, accelerator_count, is_preemptible):
    @dsl.pipeline(
        name="nanochat-pipeline",
        description="A pipeline to train NanoChat",
    )
    def nanochat_pipeline(
        gcs_bucket: str, 
        project: str,
        location: str,
        wandb_run: str = "dummy", 
        vertex_experiment: str = "", 
        vertex_tensorboard: str = "",
        num_data_shards: int = 20,
        scheduling_strategy: str = "FLEX_START",
        max_wait_duration: str = "0s",
        service_account: str = "",
        device_batch_size: int = 8
    ):
        # Data download step
        data_download_task = data_download_step(
            gcs_bucket=gcs_bucket,
            num_shards=num_data_shards
        )
        data_download_task.set_cpu_limit('8').set_memory_limit('32G')

        # Tokenizer step
        tokenizer_task = tokenizer_step(gcs_bucket=gcs_bucket)
        tokenizer_task.set_cpu_limit('8').set_memory_limit('32G')

        # Pretraining step using CustomTrainingJobOp
        # Define worker pool specs
        # Note: We use the same image and command as before
        
        worker_pool_specs = [{
            "machine_spec": {
                "machine_type": "a2-highgpu-1g" if accelerator_type == "NVIDIA_TESLA_A100" and accelerator_count == 1 else "a2-highgpu-8g" if accelerator_type == "NVIDIA_TESLA_A100" and accelerator_count == 8 else "n1-standard-16", # Fallback/Logic needs to be robust
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 500,
            },
            "container_spec": {
                "image_uri": DOCKER_IMAGE_URI,
                "command": ["python", "vertex_pipelines/pretraining_step.py"],
                "args": [
                    "--gcs-bucket", gcs_bucket,
                    "--wandb-run", wandb_run,
                    "--vertex-experiment", vertex_experiment,
                    "--vertex-tensorboard", vertex_tensorboard,
                    "--device-batch-size", str(device_batch_size)
                ],
            },
        }]
        
        # Refine machine type logic based on accelerator
        # A100 40GB: a2-highgpu-1g (1 GPU), a2-highgpu-2g (2 GPUs), a2-highgpu-4g (4 GPUs), a2-highgpu-8g (8 GPUs)
        # L4: g2-standard-4 (1 GPU), etc.
        # For now, let's assume the user passes valid combinations or we map them.
        # Given the user specifically asked for 8x A100, we target a2-highgpu-8g.
        
        machine_type = "n1-standard-16" # Default
        if accelerator_type == "NVIDIA_TESLA_A100":
            if accelerator_count == 1: machine_type = "a2-highgpu-1g"
            elif accelerator_count == 2: machine_type = "a2-highgpu-2g"
            elif accelerator_count == 4: machine_type = "a2-highgpu-4g"
            elif accelerator_count == 8: machine_type = "a2-highgpu-8g"
        elif accelerator_type == "NVIDIA_L4":
             if accelerator_count == 1: machine_type = "g2-standard-4"
             elif accelerator_count == 8: machine_type = "g2-standard-96"

        worker_pool_specs[0]["machine_spec"]["machine_type"] = machine_type
        
        # Scheduling strategy is now a runtime parameter
        # Common values:
        #   FLEX_START: Dynamic Workload Scheduler - queues jobs when resources unavailable
        #   SPOT: Preemptible instances (deprecated in favor of FLEX_START)
        #   STANDARD: Standard on-demand instances
        # max_wait_duration: "0s" = wait indefinitely, "3600s" = 1 hour, "86400s" = 24 hours

        pretraining_task = CustomTrainingJobOp(
            project=project,
            location=location,
            display_name="nanochat-pretraining-job",
            worker_pool_specs=worker_pool_specs,
            base_output_directory=f"{gcs_bucket}/pipeline_root",
            timeout="604800s", # 7 days
            restart_job_on_worker_restart=True,
            strategy=scheduling_strategy,
            max_wait_duration=max_wait_duration,
            service_account=service_account,
            tensorboard=vertex_tensorboard,
        ).after(tokenizer_task)
        
        # CustomTrainingJobOp returns a Model (if configured) or just the job resource.
        # We don't need to set resources/accelerators on the task itself because they are in worker_pool_specs.
        
        # Mid-training step - use same resources as pretraining (A100s on FLEX)
        mid_worker_pool_specs = [{
            "machine_spec": worker_pool_specs[0]["machine_spec"],
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 500,
            },
            "container_spec": {
                "image_uri": DOCKER_IMAGE_URI,
                "command": ["python", "vertex_pipelines/midtraining_step.py"],
                "args": [
                    "--gcs-bucket", gcs_bucket,
                    "--wandb-run", wandb_run,
                    "--vertex-experiment", vertex_experiment,
                    "--vertex-tensorboard", vertex_tensorboard,
                    "--device-batch-size", str(device_batch_size),
                ],
            },
        }]

        midtraining_task = CustomTrainingJobOp(
            project=project,
            location=location,
            display_name="nanochat-midtraining-job",
            worker_pool_specs=mid_worker_pool_specs,
            base_output_directory=f"{gcs_bucket}/pipeline_root",
            service_account=service_account,
            strategy=scheduling_strategy,
            max_wait_duration=max_wait_duration,
        ).after(pretraining_task)
        
        # SFT step - use same resources as pretraining (A100s on FLEX)
        sft_worker_pool_specs = [{
            "machine_spec": worker_pool_specs[0]["machine_spec"],
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 500,
            },
            "container_spec": {
                "image_uri": DOCKER_IMAGE_URI,
                "command": ["python", "vertex_pipelines/sft_step.py"],
                "args": [
                    "--gcs-bucket", gcs_bucket,
                    "--wandb-run", wandb_run,
                    "--vertex-experiment", vertex_experiment,
                    "--vertex-tensorboard", vertex_tensorboard,
                ],
            },
        }]

        sft_task = CustomTrainingJobOp(
            project=project,
            location=location,
            display_name="nanochat-sft-job",
            worker_pool_specs=sft_worker_pool_specs,
            base_output_directory=f"{gcs_bucket}/pipeline_root",
            service_account=service_account,
            strategy=scheduling_strategy,
            max_wait_duration=max_wait_duration,
        ).after(midtraining_task)
        
        report_task = report_step(gcs_bucket=gcs_bucket).after(sft_task)
        report_task.set_cpu_limit('2').set_memory_limit('8G')
        
    return nanochat_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gcp-project", type=str, required=False) # Optional if we don't run it here
    parser.add_argument("--gcs-bucket", type=str, required=True)
    parser.add_argument("--pipeline-root", type=str, required=False)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--wandb-run", type=str, default="dummy")
    parser.add_argument("--vertex-experiment", type=str, default="")
    parser.add_argument("--vertex-tensorboard", type=str, default="")
    parser.add_argument("--accelerator-type", type=str, default="NVIDIA_L4")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--num-data-shards", type=int, default=20)
    parser.add_argument("--preemptible", type=str, default="false")
    parser.add_argument("--scheduling-strategy", type=str, default=None, help="Scheduling strategy: FLEX_START, SPOT, or STANDARD")
    parser.add_argument("--max-wait-duration", type=str, default=None, help="Max wait duration for FLEX_START, e.g., '0s', '3600s'")
    parser.add_argument("--service-account", type=str, required=False, help="Service account to run the pipeline")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--template_path", type=str, default="nanochat_pipeline.json")
    args = parser.parse_args()

    is_preemptible = args.preemptible.lower() == "true"
    
    # Set smart defaults for scheduling strategy based on preemptible flag
    if args.scheduling_strategy is None:
        scheduling_strategy = "FLEX_START" if is_preemptible else "STANDARD"
    else:
        scheduling_strategy = args.scheduling_strategy
    
    if args.max_wait_duration is None:
        max_wait_duration = "0s" if is_preemptible else "86400s"
    else:
        max_wait_duration = args.max_wait_duration
    
    # Create the pipeline function dynamically with captured arguments
    pipeline_func = create_pipeline_func(
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        is_preemptible=is_preemptible
    )

    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=args.template_path,
    )



    # Initialize Vertex AI SDK
    if args.gcp_project:
        aiplatform.init(project=args.gcp_project, location=args.region)

        job = aiplatform.PipelineJob(
            display_name="nanochat-pipeline",
            template_path=args.template_path,
            pipeline_root=args.pipeline_root,
            parameter_values={
                "gcs_bucket": args.gcs_bucket,
                "project": args.gcp_project,
                "location": args.region,
                "wandb_run": args.wandb_run,
                "vertex_experiment": args.vertex_experiment,
                "vertex_tensorboard": args.vertex_tensorboard,
                "num_data_shards": args.num_data_shards,
                "scheduling_strategy": scheduling_strategy,
                "max_wait_duration": max_wait_duration,
                "service_account": args.service_account,
                "device_batch_size": args.device_batch_size,
            },
        )

        # Run the pipeline
        # service_account is optional but recommended
        job.run(
            service_account=args.service_account,
            sync=True # Block until completion or failure to ensure we see logs
        )

