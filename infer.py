import torch
from models import SwittiPipeline
from torchvision.utils import make_grid
from calculate_metrics import to_PIL_image
import os

device = 'cuda:0'
model_path = "yresearch/Switti"

pipe = SwittiPipeline.from_pretrained(model_path, device=device, torch_dtype=torch.bfloat16)

prompts = ["a photo of white dog on the left and the blue cat on the right"
          ]
images = pipe(prompts,
              cfg=6.0,
              top_k=400,
              top_p=0.95,
              more_smooth=True,
              return_pil=True,
              smooth_start_si=2,
              turn_on_cfg_start_si=0,
              turn_off_cfg_start_si=8,
              last_scale_temp=0.1,
              seed=2024,
             )

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
for i, img in enumerate(images):
    img.save(f"{output_dir}/image_{i}.png")

