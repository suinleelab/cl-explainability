"""Run feature attribution and then evaluation across multiple random seeds."""

import os

import constants
from experiment_utils import parse_args


def main():
    """Main function."""
    args = parse_args(evaluate=True, meta=True)
    if args.one_seed:
        seed_list = [args.seed]
    else:
        seed_list = constants.SEED_LIST
    for seed in seed_list:
        for different_classes in [False, True]:
            command_args = args.encoder_name
            command_args += f" {args.explanation_name}"
            command_args += f" {args.attribution_name}"
            if args.normalize_similarity:
                command_args += " --normalize-similarity"
            if different_classes:
                command_args += " --different-classes"
            command_args += f" --dataset-name {args.dataset_name}"
            command_args += f" --explicand-size {args.explicand_size}"
            command_args += f" --corpus-size {args.corpus_size}"
            command_args += f" --foil-size {args.foil_size}"
            command_args += f" --batch-size {args.batch_size}"
            command_args += f" --superpixel-dim {args.superpixel_dim}"
            command_args += f" --blur-strength {args.blur_strength}"
            if args.use_gpu:
                command_args += " --use-gpu"
            command_args += f" --gpu-num {args.gpu_num}"
            command_args += f" --seed {seed}"
            if args.randomize_model:
                command_args += " --randomize-model"
            eval_command_args = command_args.replace(
                f" --batch-size {args.batch_size}", " --batch-size 32"
            )  # Always use a batch size of 32 for efficient evaluation.
            eval_command_args += f" --eval-superpixel-dim {args.eval_superpixel_dim}"
            if args.resample_eval_foil:
                eval_command_args += " --resample-eval-foil"
            eval_command_args += f" --eval-foil-size {args.eval_foil_size}"
            if args.comprehensive:
                eval_command_args += " --comprehensive"

            if args.mode == "attribute_only":
                run_attribution = True
                run_eval = False
            elif args.mode == "eval_only":
                run_attribution = False
                run_eval = True
            else:
                run_attribution = True
                run_eval = True

            if run_attribution:
                os.system("python scripts/attribute.py " + command_args)
            if run_eval:
                os.system("python scripts/eval.py " + eval_command_args)


if __name__ == "__main__":
    main()
