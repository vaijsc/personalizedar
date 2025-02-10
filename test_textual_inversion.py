import torch
from models import SwittiPipeline
from torchvision.utils import make_grid
from calculate_metrics import to_PIL_image
import os
from datetime import datetime


device = 'cuda:0'
# model_path = "yresearch/Switti"
model_path = "yresearch/Switti-AR"

pipe = SwittiPipeline.from_pretrained(model_path, device=device, torch_dtype=torch.bfloat16)
# pipe.load_textual_inversion('/root/code/personalizedar/switti/exps/local_output_backpack_use_captions/learned_embeds-steps-800.safetensors', token="<sks>", text_encoder=pipe.text_encoder.transformer, tokenizer=pipe.text_encoder.tokenizer)
# pipe.load_textual_inversion('/root/code/personalizedar/switti/exps/local_output_backpack_use_captions/learned_embeds_2-steps-800.safetensors', token="<sks>", text_encoder=pipe.text_encoder_2.transformer, tokenizer=pipe.text_encoder_2.tokenizer)


# prompts = ["a photo of white dog on the left and the brown cat on the right", 
#         "A close-up photograph of a Corgi dog. The dog is wearing a black hat and round, dark sunglasses. The Corgi has a joyful expression, with its mouth open and tongue sticking out, giving an impression of happiness or excitement", 
#         "a photo of <sks>",
#         "a photo of <sks> in the jungle",
#         "a photo of <sks> in the snow",
#         "a photo of <sks> on the beach",
#         "a <sks> with a tree and autumn leaves in the background"
#         ]


prompts = ["a dog with a tree and autumn leaves in the background"]


# print(len(pipe.text_encoder.tokenizer))
# print(len(pipe.text_encoder_2.tokenizer))


# prompts = ["a photo of backpack"]
images, scaled_images = pipe(prompts,
              cfg=6.0,
              top_k=400,
              top_p=0.95,
              more_smooth=True,
              return_pil=True,
              smooth_start_si=2,
              turn_on_cfg_start_si=0,
              turn_off_cfg_start_si=8,
              last_scale_temp=0.1,
              seed=2025,
              return_intermediate_scaled_image=True,
             )

# add datetime to the output directory
output_dir = f"output/visualized_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(images):
    img.save(f"{output_dir}/image_{i}.png")

for i, img in enumerate(scaled_images):
    img.save(f"{output_dir}/scaled_image_{i}.png")

# for i, img in enumerate(images):
#     img.save(f"{output_dir}/image_{i}.png")

# images[0].save(f"{output_dir}/image.png")
# scaled_images[0].save(f"{output_dir}/scaled_image.png")





