# Experimental neural mesh editor: local GPU trials

These are development trials, not a benchmark. Each run supplies the **existing OBJ mesh**, two aligned training photos, a compiled region and an English instruction to our custom SD 1.5 SDS optimizer. Both cases are cat meshes, so generalization is untested. The region and wording were selected for these trials; they are not presented as an autonomous VLM success.

Hardware: RTX 4070 Laptop, 8 GB VRAM; Intel i7-12650H; 16 GB RAM. Frozen FP16 SD 1.5, 256 px differentiable rendering, 60 optimization steps. First-use kernel compilation is excluded from the timings below.

| Input run | Training IoU before / after | Untouched control IoU before / after | Time | Peak allocated VRAM |
|---|---|---|---|---|
| `7049563cefe5`, iteration 0 | 0.9254 / 0.9258 | 0.8616 / 0.8616 | 9.2 s | 2296 MiB |
| `e9022e6cad59`, iteration 0 | 0.8358 / 0.8364 | 0.9026 / 0.9026 | 12.4 s | 2296 MiB |

IoU here uses the existing 192 px CPU rasterizer and frozen target masks, not newly calibrated cameras or 3D ground truth. All unselected vertices remain fixed up to OBJ serialization precision and triangle connectivity is unchanged. Both output meshes remain connected and watertight. Training/control IDs and detailed traces are recorded in the local `.cache/neural_experiments/*/edit_report.json` files.

The changes are small. These trials demonstrate GPU feasibility and real mesh deformation, **not a substantial or independently confirmed quality improvement**. They do not justify relaxing the acceptance threshold. Production candidates still require held-out metric improvement and independent VLM verification; these standalone trials have not been accepted into a saved run.

Fixed topology cannot remove a handle or add a missing component. A pretrained 2D diffusion prior may suggest plausible but incorrect geometry. MeshUp is a closer research match for semantic mesh editing, but its official README documents a 48 GB A40 test environment and does not establish feasibility on this 8 GB laptop. See the main README for sources, installation and limitations.
