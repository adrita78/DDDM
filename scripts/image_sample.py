"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from DDDM import dist_util, logger
from DDDM.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(
            args,
            model_and_diffusion_defaults().keys()
        )
    )

    model.load_state_dict(
        dist_util.load_state_dict(
            args.model_path,
            map_location="cpu"
        )
    )

    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")

    all_images = []
    all_labels = []
    all_diagnostics = []

    while len(all_images) * args.batch_size < args.num_samples:

        model_kwargs = {}

        if args.class_cond:

            classes = th.randint(
                low=0,
                high=NUM_CLASSES,
                size=(args.batch_size,),
                device=dist_util.dev(),
            )

            model_kwargs["y"] = classes

        sample, diagnostics = diffusion.p_sample_loop(
            model,
            (
                args.batch_size,
                3,
                args.image_size,
                args.image_size,
            ),
            model_kwargs=model_kwargs,
            sample_steps=args.sample_steps,
            sigma=args.sigma,
            diagnostics=args.diagnostics,
        )

        # ---------------------------------------------
        # Save generated images
        # ---------------------------------------------

        sample = (
            (sample + 1) * 127.5
        ).clamp(0, 255).to(th.uint8)

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [
            th.zeros_like(sample)
            for _ in range(dist.get_world_size())
        ]

        dist.all_gather(
            gathered_samples,
            sample,
        )

        all_images.extend(
            [
                sample.cpu().numpy()
                for sample in gathered_samples
            ]
        )

        # ---------------------------------------------
        # Save diagnostics
        # ---------------------------------------------

        if args.diagnostics:

            all_diagnostics.extend(diagnostics)

        # ---------------------------------------------
        # Labels
        # ---------------------------------------------

        if args.class_cond:

            gathered_labels = [
                th.zeros_like(classes)
                for _ in range(dist.get_world_size())
            ]

            dist.all_gather(
                gathered_labels,
                classes,
            )

            all_labels.extend(
                [
                    labels.cpu().numpy()
                    for labels in gathered_labels
                ]
            )

        logger.log(
            f"created {len(all_images) * args.batch_size} samples"
        )

    # ---------------------------------------------
    # Images
    # ---------------------------------------------

    arr = np.concatenate(
        all_images,
        axis=0,
    )

    arr = arr[:args.num_samples]

    # ---------------------------------------------
    # Labels
    # ---------------------------------------------

    if args.class_cond:

        label_arr = np.concatenate(
            all_labels,
            axis=0,
        )

        label_arr = label_arr[:args.num_samples]

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    if dist.get_rank() == 0:

        shape_str = "x".join(
            [str(x) for x in arr.shape]
        )

        out_path = os.path.join(
            logger.get_dir(),
            f"samples_{shape_str}.npz",
        )

        if args.class_cond:

            np.savez(
                out_path,
                arr,
                label_arr,
            )

        else:

            np.savez(
                out_path,
                arr,
            )

        logger.log(
            f"saving to {out_path}"
        )

        # Diagnostics
        if args.diagnostics:

            import pandas as pd

            diagnostic_path = os.path.join(
                logger.get_dir(),
                "dddm_diagnostics.csv",
            )

            pd.DataFrame(
                all_diagnostics
            ).to_csv(
                diagnostic_path,
                index=False,
            )

            logger.log(
                f"saved diagnostics to {diagnostic_path}"
            )

    dist.barrier()

    logger.log("sampling complete")


def create_argparser():

    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        model_path="",
        sample_steps=1,

        diagnostics=False,
        sigma=1.0,
    )

    defaults.update(
        model_and_diffusion_defaults()
    )

    parser = argparse.ArgumentParser()

    add_dict_to_argparser(
        parser,
        defaults
    )

    return parser

if __name__ == "__main__":
    main()
