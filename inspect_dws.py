#!/usr/bin/env python3
"""Inspect CustomTrainingJobOp for DWS parameters."""
import inspect
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

print("CustomTrainingJobOp signature:")
print(inspect.signature(CustomTrainingJobOp))
print("\n" + "="*80 + "\n")

# Get the component function
component_fn = CustomTrainingJobOp.component_spec
print("Component spec:")
print(component_fn)
