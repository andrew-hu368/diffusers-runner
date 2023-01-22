import os
# make sure you're logged in with `huggingface-cli login`
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

import torch 

PROMPT = "a beautiful woman, symmetrical, detailed face, (torn brown leather jacket), ((indiana jones hat:1.3)), nostalgia professional majestic oil painting by Ed Binkley, Intricate, High Detail, Sharp focus, 8k, dramatic, inside cave"
NEGATIVE_PROMPT = "(((holding an item))), cowboy, mountains in background, cartoon, 3d, ((disfigured)), poorly drawn eyes, ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"
GUIDANCE_SCALE = 10
INFERENCE_STEPS = 35
WIDTH = 512
HEIGHT = 512

pipe = StableDiffusionPipeline.from_pretrained("name/sd-diffusers-repo",  torch_dtype=torch.float16)
pipe = pipe.to("mps")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

seed = int.from_bytes(os.urandom(2), "big")
print(f"Using seed: {seed}")

generator = torch.Generator().manual_seed(seed)
# First-time "warmup" pass (see explanation above)
# _ = pipe(PROMPT, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt = [PROMPT], 
    negative_prompt= [NEGATIVE_PROMPT],
    guidance_scale= GUIDANCE_SCALE,
    generator= generator,
    num_inference_steps= INFERENCE_STEPS,
    width= WIDTH,
    height= HEIGHT
).images[0]

image.save("./output/" + str(seed) + ".jpg")