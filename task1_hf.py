"""
Task 1: Hugging Face API Interaction
=====================================
1. Follow the 'freznelai' user on Hugging Face.
2. Download model: freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm
3. Download model: freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm
"""

import os
import sys
import requests as req
from huggingface_hub import HfApi, snapshot_download, login


def main():
    # --- Authentication ---
    # Reads from HF_TOKEN env var or prompts for login.
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("[✓] Authenticated with HF_TOKEN environment variable.")
    else:
        print("[!] HF_TOKEN not found in environment.")
        print("    Please set it: export HF_TOKEN='hf_...'")
        print("    Or run `huggingface-cli login` interactively.")
        sys.exit(1)

    api = HfApi()

    # --- Step 1: Follow the 'freznelai' user ---
    print("\n--- Step 1: Following 'freznelai' ---")
    try:
        # Use the HF REST API (POST /api/users/{username}/follow).
        # Requires a token with 'write' scope.
        resp = req.post(
            "https://huggingface.co/api/users/freznelai/follow",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code in (200, 201):
            print("[✓] Successfully followed 'freznelai'.")
        elif resp.status_code == 403:
            print("[!] Token lacks 'write' scope for follow action.")
            print("    To fix: go to https://huggingface.co/settings/tokens")
            print("    and create a token with 'write' permissions.")
            print("    Alternatively, follow manually: https://huggingface.co/freznelai")
        elif resp.status_code == 409:
            print("[✓] Already following 'freznelai'.")
        else:
            print(f"[!] Follow response: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        print(f"[!] Could not follow 'freznelai': {e}")

    # --- Step 2: Download Model 1 ---
    model_1 = "freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm"
    download_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(download_dir, exist_ok=True)

    print(f"\n--- Step 2: Downloading '{model_1}' ---")
    try:
        path_1 = snapshot_download(
            repo_id=model_1,
            local_dir=os.path.join(download_dir, "Face-Detector"),
        )
        print(f"[✓] Model 1 downloaded to: {path_1}")
    except Exception as e:
        print(f"[✗] Failed to download Model 1: {e}")

    # --- Step 3: Download Model 2 ---
    model_2 = "freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm"
    print(f"\n--- Step 3: Downloading '{model_2}' ---")
    try:
        path_2 = snapshot_download(
            repo_id=model_2,
            local_dir=os.path.join(download_dir, "Face-Landmarker"),
        )
        print(f"[✓] Model 2 downloaded to: {path_2}")
    except Exception as e:
        print(f"[✗] Failed to download Model 2: {e}")

    print("\n=== Task 1 Complete ===")


if __name__ == "__main__":
    main()
