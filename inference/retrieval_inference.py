"""
============================================================================
inference_demo.py
----------------------------------------------------------------------------
Multi-modal conditional medical image retrieval -- inference demo.

This is a pure image-to-image (i2i) retrieval demo. No text is involved.

It produces two outputs:
  Output 1: a coarse-retrieval embedding for each input image
            (L2-normalized `fused_latents`, ready for cosine retrieval).
  Output 2: with the 1st image as the query and the rest as candidates,
            run the reranker and report the rerank scores / final order.

NOTE (the B==1 fix):
  All input images are now encoded in a SINGLE forward pass (batch = number
  of cases, e.g. 3). This mirrors the evaluation loop, where the DataLoader
  feeds B>=2 samples at once. Encoding images one-by-one (B==1) made the model
  squeeze away the batch dimension, so `fused_latents` came out as a 1-D / 0-D
  tensor and `F.normalize(..., dim=1)` raised
  "IndexError: Dimension out of range". Batching avoids that entirely.

Supported modalities (see medical_image_preprocess.py, all 4 are usable):
  2D-CXR / 3D-CT / 3D-Brain-MRI / 2D-Ultrasound

Dependencies: torch, transformers, nibabel (3D only), pillow, numpy
============================================================================
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from retrieval_eval_dataset import load_image_tensor, modality_dict


# ==========================================================================
# Config (use placeholders when open-sourcing; do not leak internal paths)
# ==========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT           = "xx.pt"
TEXT_ENCODER_NAME    = "FremyCompany/BioLORD-2023"
CONDITION_INDEX_JSON = "configs/all_condition_index.json"
DEMO_MANIFEST        = "inference/examples/demo_cases.json"   # produced by prepare_demo_cases.py

# Coarse + rerank fusion: Score = alpha * cosine + (1 - alpha) * rerank_prob
ALPHA = 0.7


# ==========================================================================
# Model initialization
# ==========================================================================
def build_model():
    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=8, temporal_depth=4,
        dim_head=32, heads=8,
    )
    # The text encoder/tokenizer are only needed to construct CTCLIP; the demo
    # itself never feeds any text (local_text=None below).
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
    text_encoder = AutoModel.from_pretrained(TEXT_ENCODER_NAME)

    clip = CTCLIP(
        image_encoder=image_encoder, text_encoder=text_encoder, tokenizer=tokenizer,
        dim_text=768, dim_image=512, dim_latent=512,
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False,
    ).to(device)

    pkg = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state_dict = pkg['model'] if 'model' in pkg else pkg
    clip.load_state_dict(state_dict, strict=False)
    clip.eval()
    print(f"[model] loaded checkpoint from {CHECKPOINT}")
    return clip


# ==========================================================================
# Helper (kept consistent with the evaluation code)
# ==========================================================================
def select_top_k_by_attention(attn_weights, image_tokens, pos_embeddings, top_k):
    """Pick the top-k image tokens by attention score (add positional embeddings)."""
    if attn_weights.dim() == 4:
        attn_score = attn_weights.mean(dim=1)
    else:
        attn_score = attn_weights

    with torch.no_grad():
        _, indices = attn_score.topk(top_k, dim=-1)

    B, _, K = indices.shape
    D = image_tokens.shape[-1]
    indices_expanded = indices.transpose(1, 2).expand(-1, -1, D)

    sel_tokens = torch.gather(image_tokens, 1, indices_expanded)
    sel_pos = torch.gather(pos_embeddings, 1, indices_expanded)
    return sel_tokens + sel_pos


# ==========================================================================
# Encode ALL images in ONE forward pass (batch = len(cases)).
#
# This is the key change vs. the original one-image-at-a-time encode(): by
# stacking every case into a single (B, 1, D, H, W) batch we reproduce exactly
# what the DataLoader feeds the model during evaluation, so the batch dim is
# never squeezed away. No text is used (local_text=None).
# ==========================================================================
@torch.no_grad()
def encode_batch(clip, cases, modality, condition_index):
    """
    cases: list of dicts, each with 'image' (path) and 'condition' (anatomy name).
    Returns a list (same order as `cases`) of dicts:
        fused / condition / tokens / pos / attn.
    """
    B = len(cases)

    # 1) Preprocess each image -> (1, D, H, W), then stack into a batch.
    #    torch.stack(..., dim=0) on (1, D, H, W) tensors yields (B, 1, D, H, W),
    #    which is exactly the DataLoader's collated output for batch_size=B.
    vts = [load_image_tensor(c['image'], modality) for c in cases]   # each (1, D, H, W)
    video = torch.stack(vts, dim=0).to(device)   # (B, 1, D, H, W)  == DataLoader output
    video = video.unsqueeze(1)                   # (B, 1, 1, D, H, W) matches eval's video.unsqueeze(1)

    modal_idx = torch.tensor([modality_dict[modality]] * B, device=device)              # (B,)

    # `condition` here is a categorical anatomy index (NOT text) -- it selects a
    # learned condition embedding inside the model.
    cond_idx = torch.tensor(
        [condition_index[modality][c['condition']] for c in cases], device=device       # (B,)
    )

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = clip(
            cond_idx, video, return_latents_all=True, device=device,
            is_condition=True, modal_indexs=modal_idx,
            modal_embedding=True, local_text=None,   # i2i: no text
        )
    local_latents, fused_latents, condition_latents, image_tokens, _, pos_out, last_attn_map, _ = outputs

    # Sanity print: with B>=2 the leading batch dim must be preserved.
    print(f"[encode] B={B} | fused={tuple(fused_latents.shape)} "
          f"condition={tuple(condition_latents.shape)} tokens={tuple(image_tokens.shape)} "
          f"pos={tuple(pos_out.shape)} attn={tuple(last_attn_map.shape)}")

    # 2) Split the batched outputs back into per-image dicts.
    #    Indexing with [i] reproduces the original encode()'s [0] for each image.
    encoded = []
    for i in range(B):
        encoded.append({
            'fused':     fused_latents[i].float(),      # coarse-retrieval embedding (512,)
            'condition': condition_latents[i].float(),  # condition-level CLS for reranking (512,)
            'tokens':    image_tokens[i].float(),       # (N, 512)
            'pos':       pos_out[i].float(),            # (N, 512)
            'attn':      last_attn_map[i].float(),      # (1, N) attention map
        })
    return encoded


# ==========================================================================
# Rerank: score a set of candidates against the query
# ==========================================================================
@torch.no_grad()
def rerank(clip, query, candidates, modality, alpha=ALPHA):
    """Return (init_scores, rerank_probs, final_scores), each shape (N,)."""
    N = len(candidates)

    # Step 1: coarse cosine similarity
    q_norm = F.normalize(query['fused'], dim=0)
    cand_norm = F.normalize(torch.stack([c['fused'] for c in candidates]), dim=1)
    init_scores = cand_norm @ q_norm                                       # (N,)

    # Step 2: assemble query / candidate token sets (query expanded to N copies)
    q_attn = query['attn'].unsqueeze(0).expand(N, -1, -1).to(device)       # (N,1,Ntok)
    q_tok  = query['tokens'].unsqueeze(0).expand(N, -1, -1).to(device)     # (N,Ntok,512)
    q_pos  = query['pos'].unsqueeze(0).expand(N, -1, -1).to(device)
    q_cls  = query['condition'].view(1, 1, -1).expand(N, 1, -1).to(device) # (N,1,512)

    c_attn = torch.stack([c['attn'] for c in candidates]).to(device)       # (N,1,Ntok)
    c_tok  = torch.stack([c['tokens'] for c in candidates]).to(device)
    c_pos  = torch.stack([c['pos'] for c in candidates]).to(device)
    c_cls  = torch.stack([c['condition'] for c in candidates]).unsqueeze(1).to(device)  # (N,1,512)

    # Step 3: attention token selection -> prepend CLS -> rank_module
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        t1 = select_top_k_by_attention(q_attn, q_tok, q_pos, clip.rerank_topk)
        t2 = select_top_k_by_attention(c_attn, c_tok, c_pos, clip.rerank_topk)
        t1 = torch.cat([q_cls, t1], dim=1)
        t2 = torch.cat([c_cls, t2], dim=1)

        modal_idx = torch.tensor(modality_dict[modality], device=device).expand(N)
        rank_logits = clip.rank_module(t1, t2, modal_idx).squeeze(1)
        rank_probs = torch.sigmoid(rank_logits)

    # Step 4: fuse scores
    final_scores = alpha * init_scores.to(device) + (1 - alpha) * rank_probs
    return init_scores.cpu(), rank_probs.cpu(), final_scores.cpu()


# ==========================================================================
# Main
# ==========================================================================
def main():
    if not os.path.exists(DEMO_MANIFEST):
        raise FileNotFoundError(
            f"{DEMO_MANIFEST} not found. Run prepare_demo_cases.py first to "
            f"generate the 3 demo cases."
        )

    with open(DEMO_MANIFEST, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    modality = manifest['modality']            # e.g. '2D-CXR'
    cases = manifest['cases']                  # length 3: [query, cand_1, cand_2]
    assert len(cases) >= 3, "demo needs at least 3 cases (1 query + 2 candidates)"

    clip = build_model()
    with open(CONDITION_INDEX_JSON, 'r', encoding='utf-8') as f:
        condition_index = json.load(f)

    # Attach the shared condition (from the manifest top level) to each case.
    for c in cases:
        c.setdefault('condition', manifest['condition'])

    # ---- Encode all images in ONE batched forward (batch = len(cases)) ----
    encoded = encode_batch(clip, cases, modality, condition_index)

    # ================== Output 1: coarse-retrieval embedding ==================
    print("\n" + "=" * 72)
    print(f"Output 1 | coarse embedding  (modality={modality}, condition={manifest['condition']})")
    print("=" * 72)
    os.makedirs("outputs", exist_ok=True)
    for i, (c, enc) in enumerate(zip(cases, encoded)):
        emb = F.normalize(enc['fused'], dim=0).cpu()
        save_path = f"inference/outputs/embedding_{i}.npy"
        np.save(save_path, emb.numpy())
        role = "query    " if i == 0 else f"candidate{i}"
        print(f"  [{role}] {os.path.basename(c['image']):<24} dim={tuple(emb.shape)} -> {save_path}")

    # ================== Output 2: rerank ==================
    print("\n" + "=" * 72)
    print(f"Output 2 | Rerank  (Score = {ALPHA}*cosine + {1 - ALPHA}*rerank)")
    print(f"          query = {os.path.basename(cases[0]['image'])}")
    print("=" * 72)

    query, candidates = encoded[0], encoded[1:]
    cand_cases = cases[1:]
    init_scores, rerank_probs, final_scores = rerank(clip, query, candidates, modality, alpha=ALPHA)

    print(f"  {'Candidate':<24} | {'CosSim':>8} | {'RerankP':>8} | {'Final':>8}")
    print("  " + "-" * 58)
    for j, c in enumerate(cand_cases):
        print(f"  {os.path.basename(c['image']):<24} | {init_scores[j]:>8.4f} | "
              f"{rerank_probs[j]:>8.4f} | {final_scores[j]:>8.4f}")

    order = torch.argsort(final_scores, descending=True)
    ranked = " > ".join(os.path.basename(cand_cases[idx]['image']) for idx in order.tolist())
    print("\n  Final order (high -> low): " + ranked)
    print("\nDone.")


if __name__ == "__main__":
    main()