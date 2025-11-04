import kfp
from kfp.v2 import dsl
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

@dsl.pipeline(name="nanochat-pipeline")
def nanochat_pipeline(gcs_bucket: str, docker_image_uri: str, wandb_run: str = "dummy"):
    """
    A Vertex AI pipeline for training and evaluating a nanochat model.
    """
    tokenizer_op = dsl.ContainerOp(
        name="tokenizer",
        image=docker_image_uri,
        command=["python", "vertex_pipelines/tokenizer_step.py"],
        arguments=["--gcs-bucket", gcs_bucket],
    )

    pretraining_op = dsl.ContainerOp(
        name="pretraining",
        image=docker_image_uri,
        command=["python", "vertex_pipelines/pretraining_step.py"],
        arguments=["--gcs-bucket", gcs_bucket, "--wandb-run", wandb_run],
    ).after(tokenizer_op)

    midtraining_op = dsl.ContainerOp(
        name="midtraining",
        image=docker_image_uri,
        command=["python", "vertex_pipelines/midtraining_step.py"],
        arguments=["--gcs-bucket", gcs_bucket, "--wandb-run", wandb_run],
    ).after(pretraining_op)

    sft_op = dsl.ContainerOp(
        name="sft",
        image=docker_image_uri,
        command=["python", "vertex_pipelines/sft_step.py"],
        arguments=["--gcs-bucket", gcs_bucket, "--wandb-run", wandb_run],
    ).after(midtraining_op)

    report_op = dsl.ContainerOp(
        name="report",
        image=docker_image_uri,
        command=["python", "vertex_pipelines/report_step.py"],
        arguments=["--gcs-bucket", gcs_bucket],
    ).after(sft_op)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gcp-project", type=str, required=True)
    parser.add_argument("--gcs-bucket", type=str, required=True)
    parser.add_argument("--pipeline-root", type=str, required=True)
    parser.add_argument("--docker-image-uri", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    args = parser.parse_args()

    Compiler().compile(
        pipeline_func=nanochat_pipeline,
        package_path="nanochat_pipeline.json",
    )

    aiplatform.init(project=args.gcp_project, location=args.region)

    job = aiplatform.PipelineJob(
        display_name="nanochat-pipeline",
        template_path="nanochat_pipeline.json",
        pipeline_root=args.pipeline_root,
        parameter_values={
            "gcs_bucket": args.gcs_bucket,
            "docker_image_uri": args.docker_image_uri,
        },
    )

    job.run()
