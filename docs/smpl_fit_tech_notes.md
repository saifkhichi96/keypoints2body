# SMPL Fit: Technical Notes

This document summarizes the optimization pipeline used in `smpl_fit.py` to recover SMPL parameters from 3D joints. Equations use inline LaTeX for clarity.

## Inputs and model
- 3D joints sequence `J_t ∈ ℝ^{N×J×3}` (default AMASS mapping with `J=22`).
- SMPL model (neutral) with parameters: pose `θ ∈ ℝ^{72}` (global orientation + 23 joints), shape `β ∈ ℝ^{10}`, translation `t ∈ ℝ^{3}`.
- Mean pose/shape initialization from `neutral_smpl_mean_params.h5`.
- Optional confidence boosts on foot joints (heavier weights when `--fix-foot`).

## Robust penalty
We use the Geman–McClure function
$\rho(x) = \frac{\sigma^2 x^2}{\sigma^2 + x^2}$ (see `gmof`), which downweights large residuals.

## Initialization
1. Load mean pose/shape into `init_mean_pose`, `init_mean_shape`.
2. For the first frame, seed `pred_pose/pred_betas/pred_cam_t` with these means and zero translation.
3. For later frames, reuse the previous frame’s optimized `pose/beta/cam` as warm start (shape is fixed after frame 0).
4. Camera translation is given a coarse guess via torso similarity (`guess_init_3d`) using hips/shoulders.

## Optimization stages per frame
The solver runs two LBFGS stages (or Adam if configured in `SMPLify3D`):

1) **Camera + global orientation fit**  
Optimize `{R, t}` with body pose and betas frozen. Loss (`camera_fitting_loss_3d`):
$E_\text{cam} = \sum_i \lVert (j_i + t) - \hat{j}_i \rVert^2 + \lambda_d \lVert t - t_0 \rVert^2,$
where $\hat{j}_i$ are target torso joints (mapped by joint category) and $t_0$ is the initial guess. This aligns the model to observed torso depth.

2) **Body pose (and optionally shape) fit**  
Optimize `{body pose, global orient, t, betas (frame 0 only)}` with loss (`body_fitting_loss_3d`):
$$
E = w_j^2 \sum_i c_i^2 \, \rho\!\left((j_i + t) - \hat{j}_i\right)
    + w_p^2 \, \Pi(\theta)
    + w_a^2 \, A(\theta)
    + w_s^2 \lVert \beta \rVert^2
    + w_{\text{pres}}^2 \lVert \theta - \theta_{\text{prev}}\rVert^2
    + w_{\text{coll}} E_{\text{coll}}
$$
- $\Pi(\theta)$: pose prior (MoG) from `MaxMixturePrior`.
- $A(\theta)$: angle prior to discourage elbow/knee hyperextension.
- $E_{\text{coll}}$: optional interpenetration term (if collision enabled).
- `c_i`: joint confidences (feet can be upweighted).
- $\theta_{\text{prev}}$: pose from previous frame (stabilizer).

Both stages call the SMPL layer to produce joints/vertices; gradients flow through the layer during optimization.

## Outputs per frame
Stored under `<work-dir>/<sequence>/<timestamp>/`:
- `.pkl` with keys: `pose` (1×72), `beta` (1×10), `cam` (1×3), `root` (3, input root joint).

## Notes and references
- The structure and priors follow SMPLify-style fitting as used in VIBE/HMR variants; the citation in the README covers the broader context.
- Joint mappings (orig vs AMASS) and SMPL paths are defined in `src/config.py`.
