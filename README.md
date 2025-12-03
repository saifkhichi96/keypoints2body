# joints2smpl
Fit SMPL models to 3D joints and render the resulting meshes.

![SMPL Fitting Pipeline](docs/overview.png)

## Prerequisites
- Tested on Ubuntu 24.04 with CUDA 12.6 and Python 3.10.
- SMPL model files (neutral/female/male) are required.

## Installation
Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate fit3d
```

## Download SMPL models
Download SMPL [Male, Female](https://smpl.is.tue.mpg.de/) and [Neutral](https://smplify.is.tue.mpg.de/) body models and place them under `<repo>/data/models/body_models/smpl/` so the layout is:
```
<repo>
└── data/models/body_models/smpl/
    ├── SMPL_FEMALE.pkl
    ├── SMPL_MALE.pkl
    └── SMPL_NEUTRAL.pkl
```

## Usage

The main workflow, described in detail in the [technical report](docs/smpl_fit_tech_notes.md), consists of two scripts: fitting SMPL to 3D joints and rendering the fitted meshes.

### 1) Fit SMPL to a joints sequence
`smpl_fit.py` optimizes SMPL parameters for each frame of a `.npy` joints sequence.
```bash
python smpl_fit.py \
  --file ./data/demo/test_motion1.npy \
  --work-dir ./work_dirs \
  --joint-category AMASS \
  --num-smplify-iters 100
```
Key arguments:
- `--file`: Full path of the input `.npy` file containing a sequence of 3D joints (N x J x 3).
- `--work-dir`: Where per-frame `.pkl` results are written.
- `--cuda/--cpu`, `--gpu-id`: Control device selection.
- `--num-smplify-iters`, `--fix-foot`: Tuning knobs for optimization.

Output layout under `<work-dir>/<sequence-name>/<timestamp>/`:
- One `.pkl` per frame containing:
  - `pose`: (1, 72) SMPL pose (global orient + body pose) in axis-angle.
  - `beta`: (1, 10) SMPL shape coefficients.
  - `cam`: (1, 3) translation vector.
  - `root`: (3,) root joint position from input joints (for reference).

### 2) Render fitted meshes
Use `smpl_render.py` to visualize `.pkl` outputs as meshes, save PNG frames, or write a video:
```bash
python smpl_render.py \
  ./work_dirs/test_motion1/<timestamp> \
  --write-video ./work_dirs/test_motion1/video.mp4
```
Useful flags:
- `--results-dir`: Folder containing the per-frame `.pkl` files.
- `--write-video`: Write a video of the whole sequence (requires `imageio` and `imageio-ffmpeg`).
- `--write-frames`: Directory to save rendered PNGs (omit to disable offscreen rendering).
- `--display-frame N`: Open a single frame interactively (no PNGs written).
- `--width/--height`: Render resolution.
- `--smpl-dir`, `--gender`: SMPL model selection.

## Demo
Run the end-to-end demo with the provided sample:
```bash
python smpl_fit.py --data-folder ./data/demo --file test_motion1.npy --work-dir ./work_dirs
python smpl_render.py ./work_dirs/test_motion1/<timestamp> --write-video ./work_dirs/test_motion1/video.mp4
```

## Citation
If you find this project useful for your research, please consider citing:
```
@article{zuo2021sparsefusion,
  title={Sparsefusion: Dynamic human avatar modeling from sparse rgbd images},
  author={Zuo, Xinxin and Wang, Sen and Zheng, Jiangbin and Yu, Weiwei and Gong, Minglun and Yang, Ruigang and Cheng, Li},
  journal={IEEE Transactions on Multimedia},
  volume={23},
  pages={1617--1629},
  year={2021}
}
```

## References
We indicate if a function or script is borrowed externally inside each file. Here are some great resources we benefit from:
- Shape/Pose prior and some functions are borrowed from [VIBE](https://github.com/mkocabas/VIBE).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
