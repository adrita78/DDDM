from PIL import Image
import blobfile as bf
import numpy as np

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
):
    """
    Create a DataLoader over images.

    Each returned item is:

        image, condition_dict, global_index

    where:

        image:
            NCHW float tensor/array in [-1, 1]

        condition_dict:
            contains class labels if class_cond=True

        global_index:
            stable index identifying the original dataset sample.

    The global index is important for DDDM because the model maintains
    a persistent x_bar state indexed by dataset sample.
    """

    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)

    if len(all_files) == 0:
        raise ValueError(
            f"No image files found in dataset directory: {data_dir}"
        )

    # ---------------------------------------------------------
    # Determine distributed rank/world size.
    #
    # The training is launched with torchrun, so use PyTorch
    # distributed rather than MPI for dataset sharding.
    # ---------------------------------------------------------
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # ---------------------------------------------------------
    # Class labels
    # ---------------------------------------------------------
    classes = None

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [
            bf.basename(path).split("_")[0]
            for path in all_files
        ]

        sorted_classes = {
            x: i for i, x in enumerate(sorted(set(class_names)))
        }

        classes = [
            sorted_classes[x]
            for x in class_names
        ]

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=rank,
        num_shards=world_size,
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    return loader


def _list_image_files_recursively(data_dir):
    """
    Recursively find image files in data_dir.
    """

    results = []

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]

        if "." in entry and ext.lower() in [
            "jpg",
            "jpeg",
            "png",
            "gif",
        ]:
            results.append(full_path)

        elif bf.isdir(full_path):
            results.extend(
                _list_image_files_recursively(full_path)
            )

    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()

        self.resolution = resolution

        self.shard = shard
        self.num_shards = num_shards

        # -----------------------------------------------------
        # Keep the ORIGINAL GLOBAL dataset indices.
        #
        # Example with 2 GPUs:
        #
        # rank 0 -> 0, 2, 4, 6, ...
        # rank 1 -> 1, 3, 5, 7, ...
        #
        # These indices remain stable even though each rank
        # has its own local dataset.
        # -----------------------------------------------------
        self.local_indices = list(
            range(shard, len(image_paths), num_shards)
        )

        self.local_images = [
            image_paths[i]
            for i in self.local_indices
        ]

        self.local_classes = (
            None
            if classes is None
            else [
                classes[i]
                for i in self.local_indices
            ]
        )

        print(
            f"rank={shard}, "
            f"num_shards={num_shards}, "
            f"local size={len(self.local_images)}"
        )

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # -----------------------------------------------------
        # Local index -> image
        # -----------------------------------------------------
        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # -----------------------------------------------------
        # Downsample large images progressively.
        # -----------------------------------------------------
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(
                    x // 2
                    for x in pil_image.size
                ),
                resample=Image.BOX,
            )

        # -----------------------------------------------------
        # Resize so that the shorter side equals resolution.
        # -----------------------------------------------------
        scale = self.resolution / min(*pil_image.size)

        pil_image = pil_image.resize(
            tuple(
                round(x * scale)
                for x in pil_image.size
            ),
            resample=Image.BICUBIC,
        )

        # -----------------------------------------------------
        # Convert to RGB numpy array.
        # -----------------------------------------------------
        arr = np.array(
            pil_image.convert("RGB")
        )

        # -----------------------------------------------------
        # Center crop.
        # -----------------------------------------------------
        crop_y = (
            arr.shape[0] - self.resolution
        ) // 2

        crop_x = (
            arr.shape[1] - self.resolution
        ) // 2

        arr = arr[
            crop_y : crop_y + self.resolution,
            crop_x : crop_x + self.resolution,
        ]

        # -----------------------------------------------------
        # Normalize [0, 255] -> [-1, 1].
        # -----------------------------------------------------
        arr = (
            arr.astype(np.float32) / 127.5
            - 1.0
        )

        # -----------------------------------------------------
        # Class condition.
        # -----------------------------------------------------
        out_dict = {}

        if self.local_classes is not None:
            out_dict["y"] = np.array(
                self.local_classes[idx],
                dtype=np.int64,
            )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Return the GLOBAL dataset index, NOT the local idx.
        #
        # This is what allows:
        #
        #     model.x_bar[global_index]
        #
        # to refer to the same image across epochs/ranks.
        # -----------------------------------------------------
        global_index = self.local_indices[idx]

        return (
            np.transpose(arr, [2, 0, 1]),
            out_dict,
            global_index,
        )



