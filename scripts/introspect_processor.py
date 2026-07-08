"""Find the exact knob that raises the Gemma table-model's visual resolution.
Dumps the full image-processor config and checks whether changing
image_seq_length actually changes the encoded image (pixel_values / token
count). Run on the GPU host (RunPod):

    python scripts/introspect_processor.py

Loads only the processor (no model weights) + a blank test image, so it's fast.
"""
import json
import os

from PIL import Image
from transformers import AutoProcessor

mid = os.getenv("GEMMA_TABLE_MODEL", "google/gemma-4-E4B-it")
print("model id:", mid)

p = AutoProcessor.from_pretrained(mid)
print("processor:", type(p).__name__, "| image_processor:", type(p.image_processor).__name__)

print("\n=== FULL image_processor config ===")
print(json.dumps(p.image_processor.to_dict(), indent=2, default=str))

# Encode a blank image the size of the real scan to observe shapes.
img = Image.new("RGB", (1131, 1600))
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img}, {"type": "text", "text": "x"}]}]


def encode(proc):
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=txt, images=img, return_tensors="pt")
    pv = tuple(enc["pixel_values"].shape) if "pixel_values" in enc else None
    return pv, int(enc["input_ids"].shape[-1])


pv, ids = encode(p)
print(f"\n=== DEFAULT (image_seq_length={p.image_processor.to_dict().get('image_seq_length')}) ===")
print(f"  pixel_values: {pv}   input_ids: {ids}")

print("\n=== try higher image_seq_length ===")
for n in (560, 1120):
    try:
        p2 = AutoProcessor.from_pretrained(mid, image_seq_length=n)
        pv2, ids2 = encode(p2)
        cfg = p2.image_processor.to_dict().get("image_seq_length")
        print(f"  image_seq_length={n}: cfg_now={cfg}  pixel_values={pv2}  input_ids={ids2}")
    except Exception as e:
        print(f"  image_seq_length={n}: ERROR {type(e).__name__}: {e}")
