import inspect
from kfp import dsl

try:
    print("Available methods:", [m for m in dir(dsl.PipelineTask) if 'pod' in m or 'label' in m or 'annotation' in m or 'env' in m])
except Exception as e:
    print(e)
