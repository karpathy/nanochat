"""Download and run the CORE benchmark bundle."""

import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from typing import Mapping

import yaml

from nanochat.common import download_file_with_lock, eval_tasks_dir, print0
from nanochat.evaluation.core_eval import evaluate_task

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(base_dir: str, file_path: str) -> None:
    """Unzip eval_bundle.zip and place it in the base directory."""
    eval_bundle_dir = eval_tasks_dir(base_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(
    base_dir: str, model: object, tokenizer: object, device: object, max_per_task: int = -1
) -> Mapping[str, object]:
    """
    Evaluate a model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    eval_bundle_dir = eval_tasks_dir(base_dir)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            base_dir,
            EVAL_BUNDLE_URL,
            "eval_bundle.zip",
            postprocess_fn=lambda file_path: place_eval_bundle(base_dir, file_path),
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}
