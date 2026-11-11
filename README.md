# PEGAsus: 3D Personalization of Geometry and Appearance [SIGGRAPH 2026]

[arXiv](https://arxiv.org/abs/2602.08198)

## Installation

PEGAsus is built on [TRELLIS](https://github.com/microsoft/TRELLIS), which is included in this repository. Set up the environment with the TRELLIS setup script:

```bash
git clone https://github.com/hujingyucd/PEGAsus.git
cd PEGAsus
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

The pre-trained weights are downloaded automatically on first use.

## Run Examples

```bash
python inference.py --config configs/examples.yaml
```

The concepts are `house_frame`, `spiky_beetle`, `block_castle`, `palm_tree` and `catapult`; their reference meshes are in `references/`.

Results are saved to `outputs/` as `.glb`, `.ply` and `.mp4`.

### Note on Reproducibility

Generation is stochastic, so results can vary across runs. If a result does not match the expected one, try running it again with a different `--seed`.

## Acknowledgements

We sincerely thank [TRELLIS](https://github.com/microsoft/TRELLIS) for their inspirational help to our work. The `trellis/` folder and `setup.sh` are taken from TRELLIS (commit 193e050) under its MIT License.
