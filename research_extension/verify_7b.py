#!/usr/bin/env python3
import os
import sys
import time
import torch
from pathlib import Path

def ensure_symlinks():
    hub_dir = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    blob_file = hub_dir / "blobs/ae0fafb4df71c85c020891278004a13f01b99c242e0e5905970e26d7f0554ae3"
    snapshot_dir = hub_dir / "snapshots/4858886f896bd05db823557fc5cbc4ac28342af7"
    target_link = snapshot_dir / "model.safetensors"

    if not blob_file.exists():
        print(f"Error: Blob file not found at {blob_file}")
        return False

    size = blob_file.stat().st_size
    print(f"Blob file exists: {size} bytes ({size / (1024**3):.2f} GB)")
    expected_size = 5547254632
    if size != expected_size:
        print(f"Warning: File size ({size}) does not match expected size ({expected_size}). Download may still be in progress.")
        return False

    if not target_link.exists():
        rel_target = Path("../../blobs/ae0fafb4df71c85c020891278004a13f01b99c242e0e5905970e26d7f0554ae3")
        target_link.symlink_to(rel_target)
        print(f"Created symlink: {target_link} -> {rel_target}")
    else:
        print(f"Symlink already exists: {target_link}")

    return True

def test_load_and_inference():
    from research_extension.base_agent import HuggingFaceSLMBackend, SLMConfig

    print("\n--- Initializing Qwen2.5-Coder-7B 4-bit NF4 Backend ---")
    cfg = SLMConfig(
        model_name_or_path="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        quantization="4bit",
        max_new_tokens=128,
        do_sample=False,
    )
    backend = HuggingFaceSLMBackend(cfg)
    t0 = time.perf_counter()
    backend.load()
    load_time = time.perf_counter() - t0

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Load time: {load_time:.2f}s")
        print(f"GPU VRAM Allocated: {allocated:.2f} MB")
        print(f"GPU VRAM Reserved: {reserved:.2f} MB")
        print(f"GPU VRAM Peak: {peak:.2f} MB")

    # Run quick test inference
    messages = [
        {"role": "system", "content": "You are a cyber decision agent."},
        {"role": "user", "content": "Respond with JSON: {\"action\": \"ScanNetwork\", \"parameters\": {\"target_network\": \"<TARGET_CIDR>\"}}"}
    ]
    response, lat, tokens = backend.generate(messages)
    print(f"\nTest Generation Latency: {lat*1000:.1f}ms ({tokens} tokens, {tokens/lat:.1f} tok/s)")
    print(f"Response: {response}")

if __name__ == "__main__":
    if ensure_symlinks():
        test_load_and_inference()
    else:
        sys.exit(1)
