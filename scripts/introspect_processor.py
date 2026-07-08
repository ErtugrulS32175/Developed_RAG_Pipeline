"""Introspect the Gemma table-model processor to find the exact image
resolution / pan-and-scan knobs, before wiring higher resolution into
gemma_service. Run on the GPU host (RunPod), in the gemma_env venv:

    gemma_env/bin/python scripts/introspect_processor.py

Only loads the processor config (no model weights), so it's fast.
"""
import inspect
import os

from transformers import AutoProcessor

mid = os.getenv("GEMMA_TABLE_MODEL", "google/gemma-4-E4B-it")
print("model id:", mid)

p = AutoProcessor.from_pretrained(mid)
print("processor:", type(p).__name__)

ip = getattr(p, "image_processor", None)
print("image_processor:", type(ip).__name__ if ip else None)

if ip is not None:
    d = ip.to_dict()
    keys = [
        "size", "do_resize", "resample", "do_rescale",
        "do_pan_and_scan", "pan_and_scan_min_crop_size",
        "pan_and_scan_max_num_crops", "pan_and_scan_min_ratio_to_activate",
        "image_seq_length", "max_image_size", "crop_size", "longest_edge",
    ]
    print("\n-- image_processor config --")
    for k in keys:
        if k in d:
            print(f"  {k} = {d[k]}")
    try:
        print("\npreprocess params:", list(inspect.signature(ip.preprocess).parameters))
    except (ValueError, TypeError):
        pass

print("\nprocessor __call__ params:", list(inspect.signature(p.__call__).parameters))
