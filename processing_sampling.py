import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import anndata

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")

stem_path = os.getenv("STEM_PATH", ".")
sys.path.append(stem_path)

from Stem.models import Stem_models
from Stem.diffusion import create_diffusion


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device)
    
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return checkpoint


def extract_embeddings_if_needed(slide_id, data_path, device, tif_file, st_file):
    """
    Verifică dacă embeddings există. Dacă nu, le generează pe loc din imaginea TIF și fișierul h5ad.
    """
    uni_path = os.path.join(data_path, "processed_data", "1spot_uni_ebd", f"{slide_id}_uni.pt")
    conch_path = os.path.join(data_path, "processed_data", "1spot_conch_ebd", f"{slide_id}_conch.pt")
    
    if os.path.exists(uni_path) and os.path.exists(conch_path):
        print(f"[PIPELINE] Embeddings for {slide_id} already exist. Skipping extraction.")
        return

    print(f"[PIPELINE] Embeddings not found. Starting on-the-fly extraction for {slide_id}...")

    
    if not os.path.exists(tif_file) or not os.path.exists(st_file):
        raise FileNotFoundError(f"Missing raw data for {slide_id}. Ensure .tif and .h5ad exist.")

    img = Image.open(tif_file)
    adata = anndata.read_h5ad(st_file)
    
    if hf_token:
        login(token=hf_token)
    else:
        print("WARNING: No HF_TOKEN found in .env. Model download might fail.")

    from conch.open_clip_custom import create_model_from_pretrained
    pretrained_CONCH, preprocess_CONCH = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", device=device, hf_auth_token=hf_token)
    
    from uni import get_encoder
    model_UNI, transform_UNI = get_encoder(enc_name='uni', device=device)

    def get_img_embd_conch(patch):
        patch_resized = patch.resize((256, 256), Image.Resampling.LANCZOS)
        patch_processed = preprocess_CONCH(patch_resized).unsqueeze(0)
        with torch.inference_mode():
            feature_emb = pretrained_CONCH.encode_image(patch_processed.to(device), proj_contrast=False, normalize=False)
        return torch.clone(feature_emb)
    
    def get_img_embd_uni(patch):
        patch_resized = patch.resize((224, 224), Image.Resampling.LANCZOS)
        img_transformed = transform_UNI(patch_resized).unsqueeze(dim=0)
        with torch.inference_mode():
            feature_emb = model_UNI(img_transformed.to(device))
        return torch.clone(feature_emb)

    spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    radius = 112 if spot_diameter < 224 else int(spot_diameter // 2)
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    all_patch_ebd_conch = None
    all_patch_ebd_uni = None
    first = True

    print(f"Extracting patches for {len(x)} spots...")
    for spot_idx in tqdm(range(len(x))):
        patch = img.crop((x[spot_idx]-radius, y[spot_idx]-radius, x[spot_idx]+radius, y[spot_idx]+radius))
        patch_ebd_conch = get_img_embd_conch(patch)
        patch_ebd_uni   = get_img_embd_uni(patch)

        if first:
            all_patch_ebd_conch = patch_ebd_conch
            all_patch_ebd_uni   = patch_ebd_uni
            first = False
        else:
            all_patch_ebd_conch = torch.cat((all_patch_ebd_conch, patch_ebd_conch), dim=0)
            all_patch_ebd_uni   = torch.cat((all_patch_ebd_uni, patch_ebd_uni), dim=0)

    os.makedirs(os.path.dirname(conch_path), exist_ok=True)
    os.makedirs(os.path.dirname(uni_path), exist_ok=True)

    torch.save(all_patch_ebd_conch.detach().cpu(), conch_path)
    torch.save(all_patch_ebd_uni.detach().cpu(), uni_path)
    print(f"[PIPELINE] Saved embeddings to disk successfully.")

    del pretrained_CONCH
    del model_UNI
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print("[PIPELINE] Cleared Encoders from memory to make room for Diffusion Model.")


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    
    device = args.device

    model = Stem_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,
    )  
    
    ckpt_path = args.ckpt
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = find_model(ckpt_path, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    loader = DataLoader(args.dataset, batch_size=args.sampling_batch_size, shuffle=False)
    all_samples = None
    first_batch = True
    i = 1
    
    print(f"Starting sampling process on {device} (Steps: {args.num_sampling_steps}, Samples/Cond: {args.sample_num_per_cond})...")
    
    for _, y in loader:
        y = y.to(device)
        z = torch.randn(y.shape[0], 1, args.input_gene_size, device=device)
        model_kwargs = dict(y=y)
        
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        
        if first_batch:
            all_samples = samples.detach().cpu()
            first_batch = False
        else:
            all_samples = torch.cat((all_samples, samples.detach().cpu()), dim=0)
        
        print(f"Batch {i}/{len(loader)} DONE")
        i += 1
        
    os.makedirs(args.save_path, exist_ok=True)
    save_filename = args.save_path + "generated_samples_" + args.ckpt.split("/")[-1].split(".")[0] + "_" + str(args.sample_num_per_cond) + "sample.pt"
    torch.save(all_samples, save_filename)
    print(f"Successfully saved samples to: {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(Stem_models.keys()), default="Stem")
    parser.add_argument("--DiT_num_blocks", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)

    parser.add_argument("--slide_out", type=str, default="SPA125", help="Test slide ID")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list.txt")
    
    parser.add_argument("--sample_num_per_cond", type=int, default=20)
    parser.add_argument("--num_sampling_steps", type=int, default=1000) 
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_batch_size", type=int, default=32)
    
    parser.add_argument("--save_path", type=str, default="./sample/") 
    parser.add_argument("--ckpt", type=str, default="./0200000.pt") 
    parser.add_argument("--data_path", type=str, default="./")
    
    parser.add_argument("--tif_path", type=str, required=True, help="Absolute path to the TIF image from DB")
    parser.add_argument("--st_path", type=str, required=True, help="Absolute path to the h5ad file from DB")
    
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print("Apple Silicon GPU (MPS) detected! Accelerating sampling...")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("NVIDIA GPU detected! Accelerating sampling...")
    else:
        args.device = torch.device("cpu")
        print("No GPU detected. Running on CPU ...")
    
    data_path = args.data_path
    
    extract_embeddings_if_needed(args.slide_out, args.data_path, args.device, args.tif_path, args.st_path)
    
    print(f"Loading multimodal embeddings for {args.slide_out}...")
    img_ebd_uni   = torch.load(os.path.join(data_path, "processed_data", "1spot_uni_ebd", f"{args.slide_out}_uni.pt"), map_location="cpu")
    img_ebd_conch = torch.load(os.path.join(data_path, "processed_data", "1spot_conch_ebd", f"{args.slide_out}_conch.pt"), map_location="cpu")
    
    all_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], dim=1)
    args.raw_cond = all_img_ebd
    args.cond_size = all_img_ebd.shape[1]

    print(f"Combined Condition Vector Size: {args.cond_size}")

    args.cond = torch.zeros_like(args.raw_cond.repeat((args.sample_num_per_cond, 1)))
    for i in range(args.sample_num_per_cond):
        args.cond[i::args.sample_num_per_cond] = args.raw_cond.clone()

    selected_genes = np.genfromtxt(os.path.join(data_path, "processed_data", args.gene_list_filename), dtype=str)
    args.input_gene_size = len(selected_genes)

    args.dataset = CustomDataset(args.cond, args.cond)
    
    main(args)