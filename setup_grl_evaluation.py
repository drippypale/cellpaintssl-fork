#!/usr/bin/env python3
"""
Setup script for GRL-SimCLR evaluation.
Organizes directory structure and copies repository models for comparison.
"""

import os
import shutil
import argparse
from pathlib import Path


def setup_evaluation_directory(base_dir="SSL_data/embeddings"):
    """
    Create the evaluation directory structure.
    """
    grl_comparison_dir = os.path.join(base_dir, "grl_comparison")

    print(f"Setting up evaluation directory: {grl_comparison_dir}")

    # Create main directory
    os.makedirs(grl_comparison_dir, exist_ok=True)

    # Create subdirectories for each model
    model_dirs = ["GRL-SimCLR", "DINO", "MAE", "SimCLR", "CellProfiler"]

    for model_dir in model_dirs:
        model_path = os.path.join(grl_comparison_dir, model_dir)
        os.makedirs(model_path, exist_ok=True)
        print(f"  - Created: {model_path}")

    return grl_comparison_dir


def copy_repository_models(base_dir="SSL_data/embeddings", grl_comparison_dir=None):
    """
    Copy repository models to the comparison directory.
    """
    if grl_comparison_dir is None:
        grl_comparison_dir = os.path.join(base_dir, "grl_comparison")

    print(f"Copying repository models to: {grl_comparison_dir}")

    # Check if singlesource directory exists
    singlesource_dir = os.path.join(base_dir, "singlesource")
    if not os.path.exists(singlesource_dir):
        print(
            f"⚠️  Warning: {singlesource_dir} not found. Skipping repository model copy."
        )
        print("   You may need to run download_data.sh first or manually copy models.")
        return False

    # Copy models from singlesource
    models_to_copy = ["DINO", "MAE", "SimCLR", "CellProfiler"]

    for model in models_to_copy:
        src_dir = os.path.join(singlesource_dir, model)
        dst_dir = os.path.join(grl_comparison_dir, model)

        if os.path.exists(src_dir):
            # Copy the well_features.csv file
            src_file = os.path.join(src_dir, "well_features.csv")
            dst_file = os.path.join(dst_dir, "well_features.csv")

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"  ✅ Copied: {model}/well_features.csv")
            else:
                print(f"  ⚠️  Warning: {src_file} not found")
        else:
            print(f"  ⚠️  Warning: {src_dir} not found")

    return True


def create_evaluation_script(grl_comparison_dir):
    """
    Create a script to run the evaluation.
    """
    script_content = f"""#!/bin/bash
# Evaluation script for GRL-SimCLR comparison

echo "Running GRL-SimCLR evaluation..."

# Run evaluation
python evaluate.py -i grl_comparison -b SSL_data/embeddings

echo "Evaluation completed!"
echo "Results saved in: {grl_comparison_dir}/figures/"
echo "Metrics saved in: {grl_comparison_dir}/"
"""

    script_path = "run_grl_evaluation.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)

    print(f"✅ Created evaluation script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup GRL-SimCLR evaluation environment"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="SSL_data/embeddings",
        help="Base directory for embeddings",
    )
    parser.add_argument(
        "--skip_copy", action="store_true", help="Skip copying repository models"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 SETTING UP GRL-SIMCLR EVALUATION")
    print("=" * 80)

    # Setup directory structure
    grl_comparison_dir = setup_evaluation_directory(args.base_dir)

    # Copy repository models (unless skipped)
    if not args.skip_copy:
        copy_repository_models(args.base_dir, grl_comparison_dir)

    # Create evaluation script
    create_evaluation_script(grl_comparison_dir)

    print("=" * 80)
    print("✅ SETUP COMPLETED!")
    print("=" * 80)
    print(f"Evaluation directory: {grl_comparison_dir}")
    print()
    print("Next steps:")
    print("1. Run inference on your GRL-SimCLR model:")
    print("   python inference_grl_simclr.py --ckpt your_checkpoint.ckpt \\")
    print("       --submission_csv your_data.csv \\")
    print("       --output_dir SSL_data/embeddings/grl_comparison/GRL-SimCLR")
    print()
    print("2. Run evaluation:")
    print("   ./run_grl_evaluation.sh")
    print()
    print("3. Check results:")
    print(f"   - Metrics: {grl_comparison_dir}/well_metrics.csv")
    print(f"   - Plots: {grl_comparison_dir}/figures/")


if __name__ == "__main__":
    main()
