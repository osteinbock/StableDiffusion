from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
).to("cuda")

pipe.load_lora_weights(
    "OUTPUT_DIR",
    weight_name="pytorch_lora_weights.safetensors"
)

# Generate N images with seeds for reproducibility
SEED_START, SEED_END = 100, 110  # Example: generate 10 images
for idx, seed in enumerate(range(SEED_START, SEED_END), start=1):
    img = pipe(
        "a photo of a salt crystal deposit",
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    img.save(f"salt_{idx}.png")
