"""Minimal CPDM entry point.

This repository uses script-level entry points for training, sampling, and
reproduction. Download the Zenodo artifacts before running reproduction.
"""


def main() -> None:
    message = """
    CPDM reproduction entry point.

    This file does not launch training directly.

    To reproduce the Continuous CPDM preview/sweep results:

    1. Download the released Zenodo artifacts.

    2. Place the artifacts under:
        outputs/leaf_flower/prototypes/
        outputs/leaf_flower/clip_bank/
        outputs/leaf_flower/fid_stats/
        outputs/leaf_flower/continuous_cpdm/weights/
        outputs/leaf_flower/continuous_cpdm/tf_ckpt/

        The lightweight prototype NPZ files are included in the released Zenodo artifacts.

    3. Run:
        python 
        scripts/sample.py 
        sweep-preview
        --model continuous_cpdm
        --output_dir ./outputs/leaf_flower
        --step 50000

        
        bash scripts/reproduce_continuous_cpdm_sweep.sh

    For custom training, configure your local dataset paths and use:

        python scripts/train.py --help
    """
    raise SystemExit(message.strip())


if __name__ == "__main__":
    main()