# Contrastive Corpus Attribution (COCOA)

Code repository for Contrastive Corpus Attribution for Explaining Representations.

## Environment setup
1. Git clone or download this repository.
2. `cd cl-explainability`.
3. Create and activate the specified conda environment by running
    ```
    conda env create -f environment.yml
    conda activate cl-explain-env
    ```
4. Install the `cl_explain` package and the necessary dependencies for
development by running `pip install -e ".[dev]"`.

## Set up project paths
Modify global constants in `scripts/constants.py` for paths where the image data,
encoder models, and results are stored.

## Run experiments
- To train a ResNet18 model for MURA, execute `python scripts/train_classifier.py`. Run
`python scripts/train_classifier.py --help` to see how to use each command line
argument. Please see our paper for how to obtain a trained SimCLR model for
ImageNet and a trained SimSiam model for CIFAR-10.
- To run feature attributions, execute `python scripts/attribute.py`. Run
`python scripts/attribute.py --help` to see how to use each command line argument.
- To evaluate feature attributions, execute `python scripts/eval.py`. Run
`python scripts/eval.py --help` to see how to use each command line argument.
