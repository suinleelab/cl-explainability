"""Run feature attribution and then evaluation."""

import os

from experiment_utils import parse_args


def main():
    """Main function."""
    args = parse_args(evaluate=True)
    command_args = args.encoder_name
    command_args += f" {args.attribution_name}"
    command_args += f" --dataset-name {args.dataset_name}"
    command_args += f" --explicand-size {args.explicand_size}"
    command_args += f" --corpus-size {args.corpus_size}"
    if args.contrast:
        command_args += " --contrast"
    command_args += f" --foil-size {args.foil_size}"
    command_args += f" --batch-size {args.batch_size}"
    command_args += f" --superpixel-dim {args.superpixel_dim}"
    command_args += f" --blur-strength {args.blur_strength}"
    if args.use_gpu:
        command_args += " --use-gpu"
    command_args += f" --gpu-num {args.gpu_num}"
    command_args += f" --seed {args.seed}"
    eval_command_args = command_args.replace(
        f" --batch-size {args.batch_size}", " --batch-size 32"
    )  # Always use a batch size of 32 for efficient evaluation.
    eval_command_args += f" --eval-superpixel-dim {args.eval_superpixel_dim}"
    eval_command_args += f" --eval-foil-size {args.eval_foil_size}"
    os.system("python scripts/attribute.py " + command_args)
    os.system("python scripts/eval.py " + eval_command_args)
    os.system("python scripts/eval.py " + eval_command_args + " --take-attribution-abs")


if __name__ == "__main__":
    main()
