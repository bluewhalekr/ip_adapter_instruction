
import torch
print(torch.__version__)
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image
import datetime
from ip_adapter.pipeline_stable_diffusion_extra_cfg_control import StableDiffusionControlNetPipelineCFG
from ip_adapter.ip_adapter_instruct import IPAdapterInstruct

controlnet_path = '/noah/ckpt/pretrain_ckpt/StableDiffusion/instructpix2pix_controlnet'
base_model_path = "/noah/ckpt/pretrain_ckpt/StableDiffusion/rv"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "/noah/ckpt/pretrain_ckpt/StableDiffusion/lora_instruct/ip-adapter-instruct-sd15.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path,torch_dtype=torch.float16,)

# load SD pipeline
pipe = StableDiffusionControlNetPipelineCFG.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    scheduler=noise_scheduler,
    vae=vae,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)

ori_image = Image.open('/noah/inference/magna/background/images/2022-06-02-08-49-35_001970_right_rectilinear_rgb.jpg')
image = Image.open('/noah/inference/magna/background/images/2022-02-07-13-29-49_001734_rear_rectilinear_rgb.jpg')
target_size = (image.width//2, image.height//2)
target_size_8 = ((target_size[0]//8)*8, (target_size[1]//8)*8)
image = image.resize(target_size_8)
ori_image = ori_image.resize(target_size_8)

ip_model = IPAdapterInstruct(pipe, image_encoder_path, ip_ckpt, device,dtypein=torch.float16,num_tokens=16)

# only image prompt
# query="use the background"
query="use everything from the image"
images = ip_model.generate(prompt="make it snowy",ori_image=ori_image, pil_image=image, num_samples=1, seed=52222246, query=query,scale=1.0, guidance_scale=5.0,instruct_guidance_scale=2.0, image_guidance_scale=1.0, controlnet_conditioning_scale=1.1, width=image.width,height=image.height)[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_folder = '/workspace/synthetic-software/IP-Adapter-Instruct'
save_path = f"{save_folder}/grid_{timestamp}.png"

grid = image_grid([ori_image, image, images], rows=1, cols=3)

# Ensure the folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save the grid
print("Saving images grid...")
grid.save(save_path)
print(f"Images grid saved to {save_path}")
