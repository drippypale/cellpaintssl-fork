import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple, Optional
import glob

# Handle both relative and absolute imports
try:
    from . import image_ops as imo
except ImportError:
    import image_ops as imo


class JumpDataset(Dataset):
    """
    Dataset class for JUMP Cell Painting data.

    Handles the JUMP dataset structure:
    - submission CSV with job paths
    - parquet files at each job path with image metadata
    - images stored in /content/drive/MyDrive/jump_data/images

    Returns: (views, metadata_dict)
    where views is a list of augmented image tensors
    """

    def __init__(
        self,
        submission_csv: str,
        images_base_path: str = "/content/drive/MyDrive/jump_data/images",
        max_samples: Optional[int] = None,
        filter_conditions: Optional[Dict] = None,
    ):
        """
        Args:
            submission_csv: Path to submission CSV with job_paths
            images_base_path: Base path where images are stored
            max_samples: Maximum number of samples to load (for debugging)
            filter_conditions: Dict of column: value pairs to filter data
        """
        print(f"📊 [JUMP] Initializing dataset from {submission_csv}")
        self.images_base_path = images_base_path

        # Load submission CSV
        print(f"📋 [JUMP] Loading submission CSV...")
        self.submission_df = pd.read_csv(submission_csv)
        print(f"  - Found {len(self.submission_df)} job paths in submission CSV")

        # Load all parquet files and combine metadata
        print(f"📁 [JUMP] Loading parquet files...")
        self.metadata_list = []
        total_jobs = len(self.submission_df)

        for idx, (_, row) in enumerate(self.submission_df.iterrows()):
            if idx % 10 == 0:  # Progress every 10 jobs
                print(f"  - Processing job {idx + 1}/{total_jobs}: {row['batch']}")

            job_path = row["job_path"]
            if os.path.exists(job_path):
                try:
                    # Load parquet file
                    parquet_df = pd.read_parquet(job_path)
                    # Add source and batch info from submission
                    parquet_df["source"] = row["source"]
                    parquet_df["batch"] = row["batch"]
                    parquet_df["plate"] = row["plate"]

                    # Convert to list of dicts
                    for _, parquet_row in parquet_df.iterrows():
                        self.metadata_list.append(parquet_row.to_dict())

                    if idx % 10 == 0:  # Progress every 10 jobs
                        print(
                            f"    - Loaded {len(parquet_df)} samples from {row['batch']}"
                        )

                except Exception as e:
                    print(f"  ⚠️  Warning: Could not load {job_path}: {e}")
            else:
                print(f"  ⚠️  Warning: Job path not found: {job_path}")

        print(f"📊 [JUMP] Converting to DataFrame...")
        # Convert to DataFrame for easier filtering
        self.metadata_df = pd.DataFrame(self.metadata_list)
        print(f"  - Total samples loaded: {len(self.metadata_df)}")

        # Apply filters if specified
        if filter_conditions:
            print(f"🔍 [JUMP] Applying filters: {filter_conditions}")
            for col, value in filter_conditions.items():
                if col in self.metadata_df.columns:
                    if isinstance(value, list):
                        self.metadata_df = self.metadata_df[
                            self.metadata_df[col].isin(value)
                        ]
                    else:
                        self.metadata_df = self.metadata_df[
                            self.metadata_df[col] == value
                        ]
                    print(f"  - After filtering {col}: {len(self.metadata_df)} samples")

        # Limit samples if specified
        if max_samples and len(self.metadata_df) > max_samples:
            print(f"✂️  [JUMP] Limiting to {max_samples} samples...")
            self.metadata_df = self.metadata_df.sample(
                n=max_samples, random_state=42
            ).reset_index(drop=True)

        print(f"✅ [JUMP] Dataset ready: {len(self.metadata_df)} samples")

        # Create batch to index mapping for domain labels
        unique_batches = sorted(self.metadata_df["batch"].unique())
        self.batch_to_index = {b: i for i, b in enumerate(unique_batches)}
        print(f"🏷️  [JUMP] Domain mapping: {self.batch_to_index}")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        try:
            row = self.metadata_df.iloc[idx]

            # Construct image paths
            # The structure is: images_base_path/source/batch/plate/well/site/
            source = row["source"]
            batch = row["batch"]
            plate = row["plate"]
            well = row["Metadata_Well"]
            site = row["Metadata_Site"]

            # Construct full paths
            image_paths = []

            # Channel mapping based on the actual file naming
            channel_mapping = {
                "FileName_OrigDNA": "DNA",
                "FileName_OrigAGP": "AGP",
                "FileName_OrigER": "ER",
                "FileName_OrigMito": "Mito",
                "FileName_OrigRNA": "RNA",
            }

            for channel_key, channel_name in channel_mapping.items():
                # Construct filename based on the actual pattern: source_5__{plate}__{well}__{site}__{channel}.png
                filename = f"source_5__{plate}__{well}__{site}__{channel_name}.png"

                # Construct path: images_base_path/source/batch/plate/filename
                full_path = os.path.join(
                    self.images_base_path, source, batch, plate, filename
                )
                image_paths.append(full_path)

            # Load and combine channels
            channels = []
            missing_channels = 0
            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    try:
                        # Load PNG image (instead of TIFF)
                        import skimage.io

                        img = skimage.io.imread(path).astype(np.float32)
                        # Normalize to [0, 1] range
                        if img.max() > 1.0:
                            img = img / 255.0
                        channels.append(img)
                    except Exception as e:
                        if (
                            idx % 1000 == 0
                        ):  # Only print every 1000th error to avoid spam
                            print(f"⚠️  [JUMP] Could not load {path}: {e}")
                        missing_channels += 1
                        # Create zero image as fallback
                        channels.append(np.zeros((768, 768), dtype=np.float32))
                else:
                    if (
                        idx % 1000 == 0
                    ):  # Only print every 1000th missing file to avoid spam
                        print(f"⚠️  [JUMP] Image not found: {path}")
                    missing_channels += 1
                    # Create zero image as fallback
                    channels.append(np.zeros((768, 768), dtype=np.float32))

            # Log missing channels occasionally
            if missing_channels > 0 and idx % 1000 == 0:
                print(f"⚠️  [JUMP] Sample {idx}: {missing_channels}/5 channels missing")

            # Stack channels: (C, H, W)
            img = np.stack(channels, axis=0)

            # Convert to tensor
            img = torch.from_numpy(img.astype(np.float32))

            # Note: Transforms are applied by wrapper classes, not here
            # This ensures the base dataset returns raw images

            # Ensure we never return None
            if img is None:
                print(f"⚠️  [JUMP] Image is None for sample {idx}, creating zero tensor")
                img = torch.zeros((5, 768, 768), dtype=torch.float32)

            # Create metadata dict
            metadata_dict = {
                "source": source,
                "batch": batch,
                "plate": plate,
                "well": well,
                "site": site,
                "compound": row.get("Metadata_Name", "") or "",
                "smiles": row.get("Metadata_SMILES", "") or "",
                "pert_type": row.get("Metadata_pert_type", "") or "",
            }

            return img, metadata_dict

        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            # Return a safe fallback
            img = torch.zeros((5, 768, 768), dtype=torch.float32)
            metadata_dict = {
                "source": "unknown",
                "batch": "unknown",
                "plate": "unknown",
                "well": "unknown",
                "site": "unknown",
                "compound": "",
                "smiles": "",
                "pert_type": "",
            }
            return img, metadata_dict


class JumpDatasetWithDomainLabels(Dataset):
    """
    Wrapper around JumpDataset that also returns domain labels for GRL training.

    Returns: (views, metadata_dict, domain_label)
    """

    def __init__(self, jump_dataset: JumpDataset):
        self.jump_dataset = jump_dataset
        self.batch_to_index = jump_dataset.batch_to_index

    def __len__(self):
        return len(self.jump_dataset)

    def __getitem__(self, idx):
        img, metadata = self.jump_dataset[idx]
        batch = metadata["batch"]
        domain_label = self.batch_to_index.get(batch, 0)
        return img, metadata, domain_label


class JumpSubsetWithDomainLabels(Dataset):
    """
    Wrapper around Subset that preserves batch_to_index and adds domain labels.
    This is needed after random_split() which returns Subset objects.
    """

    def __init__(self, subset: Subset, batch_to_index: Dict):
        self.subset = subset
        self.batch_to_index = batch_to_index

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            img, metadata = self.subset[idx]

            if img is None:
                img = torch.zeros((5, 768, 768), dtype=torch.float32)

            batch = metadata.get("batch", "unknown")
            domain_label = self.batch_to_index.get(batch, 0)

            return img, metadata, domain_label
        except Exception as e:
            print(f"Error in JumpSubsetWithDomainLabels.__getitem__({idx}): {e}")
            # Return safe fallback
            img = torch.zeros((5, 768, 768), dtype=torch.float32)
            metadata = {
                "source": "unknown",
                "batch": "unknown",
                "plate": "unknown",
                "well": "unknown",
                "site": "unknown",
                "compound": "",
                "smiles": "",
                "pert_type": "",
            }
            domain_label = 0
            return img, metadata, domain_label


class JumpDatasetWithTransform(Dataset):
    """
    Wrapper around JumpDataset that applies SimCLR transform to create two views.
    This is needed because JUMP data are single tensors, but SimCLR expects two augmented views.
    """

    def __init__(self, jump_dataset: JumpDataset, transform=None):
        self.jump_dataset = jump_dataset
        self.transform = transform

    def __len__(self):
        return len(self.jump_dataset)

    def __getitem__(self, idx):
        img, metadata = self.jump_dataset[idx]

        if self.transform is not None:
            # Apply transform to create two views
            # For JUMP data, we don't have otsuth metadata, so pass None
            try:
                views = self.transform(img, metadata=None)
                return views, metadata
            except Exception as e:
                if idx % 100 == 0:  # Print transform errors occasionally
                    print(f"⚠️  [JUMP] Transform failed for sample {idx}: {e}")
                # Fallback: return single image as list
                return [img], metadata
        else:
            # No transform - return single image as list for compatibility
            return [img], metadata


class JumpDatasetWithTransformAndDomainLabels(Dataset):
    """
    Wrapper that applies transform and adds domain labels.
    Returns: (views, metadata_dict, domain_label)
    """

    def __init__(self, jump_dataset: JumpDataset, transform=None):
        self.jump_dataset = jump_dataset
        self.transform = transform
        self.batch_to_index = jump_dataset.batch_to_index

    def __len__(self):
        return len(self.jump_dataset)

    def __getitem__(self, idx):
        img, metadata = self.jump_dataset[idx]
        batch = metadata["batch"]
        domain_label = self.batch_to_index.get(batch, 0)

        if self.transform is not None:
            # Apply transform to create two views
            # For JUMP data, we don't have otsuth metadata, so pass None
            try:
                views = self.transform(img, metadata=None)
                return views, metadata, domain_label
            except Exception as e:
                if idx % 100 == 0:  # Print transform errors occasionally
                    print(f"⚠️  [JUMP] Transform failed for sample {idx}: {e}")
                # Fallback: return single image as list
                return [img], metadata, domain_label
        else:
            # No transform - return single image as list for compatibility
            return [img], metadata, domain_label


class JumpSubsetWithTransformAndDomainLabels(Dataset):
    """
    Wrapper around Subset that applies transform and preserves domain labels.
    """

    def __init__(self, subset: Subset, batch_to_index: Dict, transform=None):
        self.subset = subset
        self.batch_to_index = batch_to_index
        self.transform = transform
        print(f"🔍 [DEBUG] JumpSubsetWithTransformAndDomainLabels initialized:")
        print(f"  - Subset length: {len(self.subset)}")
        # Access the original dataset through the subset's dataset attribute
        if hasattr(self.subset, "dataset"):
            print(f"  - Original dataset length: {len(self.subset.dataset)}")
        else:
            print(f"  - Original dataset not accessible")

        # Pre-compute domain indices for faster balanced sampling
        self.domain_indices = self._precompute_domain_indices()
        print(f"  - Pre-computed domain indices for {len(self.domain_indices)} domains")

    def _precompute_domain_indices(self):
        """Pre-compute domain indices for fast balanced sampling"""
        domain_indices = {}

        # Get the original dataset and subset indices
        if hasattr(self.subset, "dataset") and hasattr(self.subset, "indices"):
            original_dataset = self.subset.dataset
            subset_indices = self.subset.indices

            # Group by domain using the original metadata
            for subset_idx, original_idx in enumerate(subset_indices):
                try:
                    # Get metadata from original dataset without loading image
                    row = original_dataset.metadata_df.iloc[original_idx]
                    batch = row["batch"]
                    domain = self.batch_to_index.get(batch, 0)

                    if domain not in domain_indices:
                        domain_indices[domain] = []
                    domain_indices[domain].append(subset_idx)
                except Exception as e:
                    print(f"⚠️  Error pre-computing domain index {subset_idx}: {e}")
                    continue

        return domain_indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            # Get the item from the subset directly
            item = self.subset[idx]

            if len(item) == 2:
                img, metadata = item
            elif len(item) == 3:
                img, metadata, _ = item
            else:
                raise ValueError(f"Unexpected item format: {len(item)} elements")

            if img is None:
                img = torch.zeros((5, 768, 768), dtype=torch.float32)

            batch = metadata.get("batch", "unknown")
            domain_label = self.batch_to_index.get(batch, 0)

            if self.transform is not None:
                # Apply transform to create two views
                # For JUMP data, we don't have otsuth metadata, so pass None
                try:
                    views = self.transform(img, metadata=None)
                    return views, metadata, domain_label
                except Exception as e:
                    if idx % 100 == 0:  # Print transform errors occasionally
                        print(f"⚠️  [JUMP] Transform failed for sample {idx}: {e}")
                    # Fallback: return single image as list
                    return [img], metadata, domain_label
            else:
                # No transform - return single image as list for compatibility
                return [img], metadata, domain_label

        except Exception as e:
            print(
                f"❌ [DEBUG] Error in JumpSubsetWithTransformAndDomainLabels.__getitem__({idx}): {e}"
            )

            # Return safe fallback
            img = torch.zeros((5, 768, 768), dtype=torch.float32)
            metadata = {
                "source": "unknown",
                "batch": "unknown",
                "plate": "unknown",
                "well": "unknown",
                "site": "unknown",
                "compound": "",
                "smiles": "",
                "pert_type": "",
            }
            domain_label = 0

            if self.transform is not None:
                try:
                    views = self.transform(img, metadata=None)
                    return views, metadata, domain_label
                except Exception:
                    return [img], metadata, domain_label
            else:
                return [img], metadata, domain_label


def get_jump_dataloaders(
    submission_csv: str,
    images_base_path: str = "/content/drive/MyDrive/jump_data/images",
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    filter_conditions: Optional[Dict] = None,
    with_domain_labels: bool = False,
):
    """
    Create train/val dataloaders for JUMP dataset.

    Args:
        submission_csv: Path to submission CSV
        images_base_path: Base path for images
        transform: Augmentation transform
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_ratio: Ratio of data to use for training
        max_samples: Maximum samples to load (for debugging)
        filter_conditions: Dict of filters to apply
        with_domain_labels: Whether to return domain labels for GRL

    Returns:
        train_loader, val_loader, batch_to_index
    """
    from torch.utils.data import DataLoader, random_split

    print(f"🔧 [DATALOADER] Creating JUMP dataloaders...")
    print(f"  - Submission CSV: {submission_csv}")
    print(f"  - Images base path: {images_base_path}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Train ratio: {train_ratio}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - With domain labels: {with_domain_labels}")

    # Create dataset
    print(f"📊 [DATALOADER] Creating dataset...")
    dataset = JumpDataset(
        submission_csv=submission_csv,
        images_base_path=images_base_path,
        max_samples=max_samples,
        filter_conditions=filter_conditions,
    )

    # Split into train/val
    print(f"✂️  [DATALOADER] Splitting dataset...")
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    print(f"  - Total samples: {total_size}")
    print(f"  - Train samples: {train_size}")
    print(f"  - Val samples: {val_size}")

    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Add domain labels if requested
    print(f"🏷️  [DATALOADER] Setting up domain labels...")
    if with_domain_labels:
        print(f"  - Creating datasets with domain labels for GRL")
        train_dataset = JumpSubsetWithTransformAndDomainLabels(
            train_subset, dataset.batch_to_index, transform
        )
        val_dataset = JumpSubsetWithTransformAndDomainLabels(
            val_subset, dataset.batch_to_index, transform
        )
    else:
        print(f"  - Creating datasets without domain labels")
        train_dataset = JumpDatasetWithTransform(train_subset, transform)
        val_dataset = JumpDatasetWithTransform(val_subset, transform)

    # Create dataloaders
    print(f"🚀 [DATALOADER] Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"✅ [DATALOADER] Dataloaders ready!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Domain mapping: {dataset.batch_to_index}")

    return train_loader, val_loader, dataset.batch_to_index
