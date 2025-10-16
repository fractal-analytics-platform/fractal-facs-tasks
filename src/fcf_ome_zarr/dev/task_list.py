"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterNonParallelTask, ParallelTask

AUTHORS = "Joel Luethi"


DOCS_LINK = None


INPUT_MODELS = [
    ("ngio", "images/_image.py", "ChannelSelectionModel"),
    (
        "fcf_ome_zarr",
        "utils.py",
        "MaskingConfiguration",
    ),
    (
        "fcf_ome_zarr",
        "utils.py",
        "IteratorConfiguration",
    ),
]

TASK_LIST = [
    ParallelTask(
        name="Threshold Segmentation",
        executable="threshold_segmentation_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/threshold_segmentation_task.md",
    ),
    ConverterNonParallelTask(
        name="FCF S8 Converter",
        executable="fcf_s8_converter.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        tags=["FACS"],
        docs_info="file:docs_info/fcf_s8_converter.md",
    ),
    ParallelTask(
        name="Region Props Features",
        executable="region_props_features_task.py",
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["Region Properties", "Intensity", "Morphology"],
        docs_info="file:docs_info/region_props_features_task.md",
    ),
]
