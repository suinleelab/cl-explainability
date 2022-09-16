"""Global constants for project scripts."""

DATA_PATH = "/projects/leelab/data/image"
ENCODER_PATH = "/projects/leelab/cl-explainability/encoders"
RESULT_PATH = "/projects/leelab/cl-explainability/results"
SUPERPIXEL_ATTRIBUTION_METHODS = ["kernel_shap"]
IMAGENETTE_SYNSETS = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
NUM_CLASSES_CIFAR = 10
NUM_CLASSES_MURA = 2
MODEL_OUTPUT_PATH = "/projects/leelab/models/image"
CLIP_DATA_PATH = "/projects/leelab/cl-explainability/archive/clip_use_case"
SEED_LIST = [123, 456, 789, 42, 91]
TRAIN_VAL_SPLIT_SEED = 42
