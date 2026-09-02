"""
Train a diffusion model on images.
"""

import argparse
import os

from DDDM import dist_util, logger
from DDDM.image_datasets import load_data
from DDDM.resample import create_named_schedule_sampler
from DDDM.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from DDDM.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------
    if not args.data_dir:
        raise ValueError("--data_dir must be provided.")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {args.data_dir}"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")

    if args.microbatch == 0 or args.microbatch < -1:
        raise ValueError("--microbatch must be -1 or a positive integer.")

    # ---------------------------------------------------------
    # Distributed setup
    # ---------------------------------------------------------
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(
            args,
            model_and_diffusion_defaults().keys(),
        )
    )

    model.to(dist_util.dev())

    # ---------------------------------------------------------
    # Schedule sampler
    # ---------------------------------------------------------
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler,
        diffusion,
    )

    # ---------------------------------------------------------
    # Data loader
    # ---------------------------------------------------------
    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        epochs=args.epochs,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",

        # Optimization
        lr=1e-4,
        weight_decay=0.0,

        # Training
        epochs=1000,
        batch_size=1,
        microbatch=-1,

        # EMA / logging / checkpointing
        ema_rate="0.9999",
        log_interval=10,
        save_interval=2,
        resume_checkpoint="",

        # Mixed precision
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
