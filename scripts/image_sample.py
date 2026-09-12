"""
Generate image samples from a DDDM model on a single GPU.

This version does NOT use torch.distributed/NCCL and is intended
for a single GPU such as an RTX 3090.
"""

import argparse
import os

import numpy as np
import torch as th

from DDDM import logger
from DDDM.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def load_checkpoint(model, model_path, device):
    """
    Load a DDDM checkpoint.

    Handles both ordinary state_dict checkpoints and checkpoints
    containing a 'state_dict' entry. Also removes a possible
    'module.' prefix from DDP-trained checkpoints.
    """

    checkpoint = th.load(
        model_path,
        map_location=device,
    )

    # Some checkpoints are wrapped inside "state_dict".
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Remove DDP "module." prefix if present.
    cleaned_checkpoint = {}

    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_checkpoint[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_checkpoint,
        strict=False,
    )

    if len(missing) > 0:
        print("\nWARNING: Missing checkpoint keys:")
        for key in missing:
            print("  ", key)

    if len(unexpected) > 0:
        print("\nWARNING: Unexpected checkpoint keys:")
        for key in unexpected:
            print("  ", key)

    print("\nCheckpoint loaded successfully.")


def main():

    args = create_argparser().parse_args()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if th.cuda.is_available():
        device = th.device("cuda")
        print("Using GPU:", th.cuda.get_device_name(0))
    else:
        device = th.device("cpu")
        print("CUDA unavailable. Using CPU.")

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------

    logger.configure()

    logger.log("creating model and diffusion...")

    # ---------------------------------------------------------
    # Create model and diffusion
    # ---------------------------------------------------------

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(
            args,
            model_and_diffusion_defaults().keys(),
        )
    )

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    logger.log(
        f"loading model checkpoint from {args.model_path}"
    )

    load_checkpoint(
        model,
        args.model_path,
        device,
    )

    model.to(device)
    model.eval()

    logger.log("model loaded")
    logger.log("sampling...")

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    all_images = []
    all_labels = []
    all_diagnostics = []

    num_generated = 0

    # ---------------------------------------------------------
    # Sampling loop
    # ---------------------------------------------------------

    while num_generated < args.num_samples:

        # Don't generate more than necessary on the final batch.
        current_batch_size = min(
            args.batch_size,
            args.num_samples - num_generated,
        )

        model_kwargs = {}

        # -----------------------------------------------------
        # Class conditioning
        # -----------------------------------------------------

        if args.class_cond:

            classes = th.randint(
                low=0,
                high=NUM_CLASSES,
                size=(current_batch_size,),
                device=device,
            )

            model_kwargs["y"] = classes

        # -----------------------------------------------------
        # Sample
        # -----------------------------------------------------
        if args.diagnostics:
          with th.enable_grad():

            sample, diagnostics = diffusion.p_sample_loop(
                model,
                (
                    current_batch_size,
                    3,
                    args.image_size,
                    args.image_size,
                ),
                model_kwargs=model_kwargs,
                sample_steps=args.sample_steps,
                #sigma=args.sigma,
                diagnostics=args.diagnostics,
            )
        else:
          with th.no_grad():
            sample, diagnostics = diffusion.p_sample_loop(
            model,
            (
                current_batch_size,
                3,
                args.image_size,
                args.image_size,
            ),
            model_kwargs=model_kwargs,
            sample_steps=args.sample_steps,
            diagnostics=False,
        )  
        # -----------------------------------------------------
        # Convert [-1, 1] -> [0, 255]
        # -----------------------------------------------------

        sample = (
            (sample + 1) * 127.5
        ).clamp(
            0,
            255,
        ).to(
            th.uint8
        )

        # NCHW -> NHWC
        sample = sample.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        # -----------------------------------------------------
        # Store images
        # -----------------------------------------------------

        all_images.append(
            sample.cpu().numpy()
        )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------

        if args.diagnostics:

            # diagnostics is expected to be a list of
            # dictionaries, one per sample/step depending
            # on your DDDM implementation.
            all_diagnostics.extend(diagnostics)

        # -----------------------------------------------------
        # Labels
        # -----------------------------------------------------

        if args.class_cond:

            all_labels.append(
                classes.cpu().numpy()
            )

        num_generated += current_batch_size

        logger.log(
            f"created {num_generated} / {args.num_samples} samples"
        )

    # ---------------------------------------------------------
    # Combine images
    # ---------------------------------------------------------

    arr = np.concatenate(
        all_images,
        axis=0,
    )

    arr = arr[:args.num_samples]

    # ---------------------------------------------------------
    # Combine labels
    # ---------------------------------------------------------

    if args.class_cond:

        label_arr = np.concatenate(
            all_labels,
            axis=0,
        )

        label_arr = label_arr[:args.num_samples]

    # ---------------------------------------------------------
    # Save images
    # ---------------------------------------------------------

    shape_str = "x".join(
        str(x) for x in arr.shape
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

    # ---------------------------------------------------------
    # Save diagnostics
    # ---------------------------------------------------------

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

    logger.log("sampling complete")


def create_argparser():

    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        model_path="",
        sample_steps=1,

        diagnostics=True,
    )

    defaults.update(
        model_and_diffusion_defaults()
    )

    parser = argparse.ArgumentParser()

    add_dict_to_argparser(
        parser,
        defaults,
    )

    return parser


if __name__ == "__main__":
    main()

