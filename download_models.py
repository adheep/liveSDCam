"""Pre-download (and validate) every model LiveSDCam uses.

Run once after install so the first app launch isn't interrupted by downloads,
and so offline use works. Models land in the shared Hugging Face / torch caches
(e.g. ~/.cache/huggingface). Each loader below mirrors how the engines load the
model, so it fetches exactly the files that are actually used.

    python download_models.py
"""
import gc
import sys

import torch

DTYPE = torch.float16
OK, FAIL = [], []


def step(desc, fn):
    print(f"\n>>> {desc}")
    try:
        fn()
        gc.collect()
        print(f"    OK")
        OK.append(desc)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        FAIL.append(desc)


def get_sd_turbo():
    from diffusers import AutoPipelineForImage2Image
    AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None)


def get_wardrobe_inpaint():
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=DTYPE, safety_checker=None)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")  # LCM-LoRA


def get_clothes_parser():
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


def get_person_segmenter():
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
    )
    deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)


def get_captioner():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


print("Pre-fetching LiveSDCam models (this may take a while on first run)...")
step("SD-Turbo               (Lite base)", get_sd_turbo)
step("Dreamshaper-8 inpaint + LCM-LoRA (Wardrobe / Try-On)", get_wardrobe_inpaint)
step("Segformer clothes parser (Wardrobe)", get_clothes_parser)
step("DeepLabV3-MobileNet    (Keep background)", get_person_segmenter)
step("BLIP captioner         (Try-On garment description)", get_captioner)

print("\n" + "=" * 50)
print(f"Done. {len(OK)} ok, {len(FAIL)} failed.")
if FAIL:
    print("Failed:", ", ".join(FAIL))
    sys.exit(1)
print("All models cached. Run:  python app.py")
