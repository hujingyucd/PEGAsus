"""PEGAsus inference: generate a 3D object from a pre-trained geometry concept.

A concept is one learned text token plus a fine-tuned TRELLIS stage-1 (sparse structure) flow model.
Stage 2 uses the stock TRELLIS model, conditioned on the word "object".

    python inference.py --concept spiky_beetle --prompt cat --seed 42
    python inference.py --config configs/examples.yaml
"""
import os

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import argparse

import yaml

import imageio
import torch
from safetensors.torch import load_file
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

BASE_MODEL = "JeffreyXiang/TRELLIS-text-xlarge"
WEIGHTS_REPO = "HJY123/PEGAsus"  # Hugging Face repo with the pre-trained concepts
SAMPLER = {"steps": 25, "cfg_strength": 7.5}


def encode_text(pipeline, text, token=None):
    """CLIP-encode `text`. If `token` is given, it replaces the embedding of the word "object"."""
    tcm = pipeline.text_cond_model
    tokenizer, clip = tcm["tokenizer"], tcm["model"]
    ids = tokenizer([text], max_length=77, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    ids = ids.to(pipeline.device)
    handle = None
    if token is not None:
        object_id = tokenizer(["object"], return_tensors="pt")["input_ids"][0][1].item()
        position = (ids[0] == object_id).nonzero().item()

        def replace_object(_module, _inputs, output):
            output = output.clone()
            output[:, position:position + 1, :] = token.to(output.device, output.dtype)
            return output

        handle = clip.text_model.embeddings.token_embedding.register_forward_hook(replace_object)
    try:
        cond = clip(input_ids=ids).last_hidden_state
    finally:
        if handle is not None:
            handle.remove()
    return {"cond": cond, "neg_cond": tcm["null_cond"]}


def concept_dir(weights, concept):
    """Local folder of a concept; downloads it from Hugging Face on first use."""
    directory = os.path.join(weights, concept)
    if not os.path.exists(os.path.join(directory, "denoiser.safetensors")):
        from huggingface_hub import snapshot_download
        snapshot_download(WEIGHTS_REPO, allow_patterns=[f"{concept}/*"], local_dir=weights)
    return directory


def load_concept(pipeline, directory):
    """Load the fine-tuned stage-1 weights into the pipeline and return the learned token [1, 1, 768]."""
    state = load_file(os.path.join(directory, "denoiser.safetensors"))
    missing, unexpected = pipeline.models["sparse_structure_flow_model"].load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{directory}: {len(missing)} missing and {len(unexpected)} unexpected keys")
    token = torch.load(os.path.join(directory, "token.pt"), map_location=pipeline.device)
    if token.shape[1] > 1:
        # a full [1, 77, 768] word-embedding file from training: the learned token sits at position 1 ("object")
        token = token[:, 1:2, :]
    return token


def generate(pipeline, token, prompt, seed, out_stem):
    with torch.no_grad():
        stage1 = encode_text(pipeline, f"{prompt} of object", token)
        stage2 = encode_text(pipeline, "object")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        coords = pipeline.sample_sparse_structure(stage1, 1, SAMPLER)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        slat = pipeline.sample_slat(stage2, coords, SAMPLER)
        outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])
    video = render_utils.render_video(outputs["gaussian"][0])["color"]
    imageio.mimsave(out_stem + ".mp4", video, fps=30)
    outputs["gaussian"][0].save_ply(out_stem + ".ply")
    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
    glb.export(out_stem + ".glb")
    print("saved", out_stem + ".{glb,ply,mp4}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept", help="concept name, see README")
    parser.add_argument("--prompt", help='object category; the full prompt is "<prompt> of <concept>"')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", help="YAML file with a list of examples (concept, prompt, seed)")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--base_model", default=BASE_MODEL, help="TRELLIS text model: Hugging Face id or local folder")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            jobs = [(str(e["concept"]), e["prompt"], int(e.get("seed", args.seed))) for e in yaml.safe_load(f)["examples"]]
    elif args.concept and args.prompt:
        jobs = [(args.concept, args.prompt, args.seed)]
    else:
        parser.error("give --concept and --prompt, or --config")

    os.makedirs(args.output, exist_ok=True)
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.base_model)
    pipeline.cuda()
    loaded = None
    for concept, prompt, seed in jobs:
        if concept != loaded:
            token = load_concept(pipeline, concept_dir(args.weights, concept))
            loaded = concept
        generate(pipeline, token, prompt, seed, os.path.join(args.output, f"{concept}_{prompt}_{seed}"))


if __name__ == "__main__":
    main()
