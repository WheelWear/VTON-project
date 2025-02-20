import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from CatVTON.model.attn_processor import SkipAttnProcessor
from CatVTON.model.utils import get_trainable_module, init_adapter
from CatVTON.utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)


class CatVTONPipeline_Train:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cuda",
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
    ):
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler", weight_dtype=weight_dtype, num_train_timesteps = 1000)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        is_train=True, 
        eta=1.0,
        **kwargs
    ):
        import matplotlib.pyplot as plt
        # Helper function to visualize latent tensors (e.g., by averaging channels)
        # 출력 디렉토리가 없으면 생성
        os.makedirs("vis", exist_ok=True)

        # 잠재 변수 저장 헬퍼 함수 (채널 평균을 계산해 저장)
        def save_latent(latent, filename, cmap='gray'):
            latent_mean = latent.mean(dim=1).cpu().float().numpy()  # 채널 평균
            plt.figure(figsize=(5, 5))
            plt.imshow(latent_mean[0], cmap=cmap)
            plt.title(filename.split('.')[0])
            plt.axis('off')
            plt.savefig(os.path.join("vis", filename))
            plt.close()

        # 이미지 저장 헬퍼 함수 ([B, C, H, W] 형식의 텐서 예상)
        def save_image(input_data, filename):
    # Handle PyTorch tensor input
            if isinstance(input_data, torch.Tensor):
                img = input_data[0].cpu().permute(1, 2, 0).float().numpy()
            # Handle NumPy array input
            elif isinstance(input_data, np.ndarray):
                if input_data.ndim == 4:  # [B, H, W, C]
                    img = input_data[0]
                elif input_data.ndim == 3:  # [H, W, C]
                    img = input_data
                else:
                    raise ValueError(f"Unsupported NumPy array shape: {input_data.shape}")
                # Ensure channel dimension is last (e.g., convert [C, H, W] to [H, W, C])
                if img.shape[0] in [1, 3, 4]:  # Assuming channels-first
                    img = img.transpose(1, 2, 0)
            else:
                raise TypeError("Input must be a PyTorch tensor or NumPy array")
            
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())
            
            # Plot and save
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(filename.split('.')[0])
            plt.axis('off')
            plt.savefig(os.path.join("vis", filename))
            plt.close()

        concat_dim = -2  # FIXME: y axis concat (원래 코드와 동일)
        # 1. 입력 전처리: PIL 또는 Tensor를 받아 지정된 크기로 변환
        save_image(mask, "before_check_mask.png")
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        save_image(mask, "before_prepare_mask.png")
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        #-----------------
        # 저장: 원본 이미지, 조건 이미지, 마스크
        save_image(image, "original_image.png")
        save_image(condition_image, "condition_image.png")
        save_image(mask, "after_prepare_mask.png")
        #-----------------
        # 2. 마스크 적용 (예: inpainting에서 복원할 영역 지정)
        masked_image = image * (mask < 0.5)
        save_image(masked_image, "masked_image")
        # 3. VAE 인코딩 (보통 VAE는 고정하므로 no_grad 사용)
        with torch.no_grad():
            masked_latent = compute_vae_encodings(masked_image, self.vae)
            condition_latent = compute_vae_encodings(condition_image, self.vae)
        # print("masked_latent : ",masked_latent.shape)
        # print("condition_latent : ", condition_latent.shape)
        # 저장: VAE 인코딩 후 잠재 변수
        save_latent(masked_latent, "masked_latent.png")
        save_latent(condition_latent, "condition_latent.png")

        # 4. mask latent 생성: masked_latent와 동일한 spatial 해상도로 보간
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        save_latent(mask_latent, "mask_latent.png")
        # print("mask_latent : ", mask_latent.shape)
        del image, mask, condition_image  # 메모리 해제
        
        # 5. 조건 결합: 높이 차원(-2)에서 연결 -> (B, C, H+H, W) = (B, C, 2H, W)
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # print("masked_latent_concat : ", masked_latent_concat.shape)
        # print("mask_latent_concat : ", mask_latent_concat.shape)
        save_latent(masked_latent_concat, "masked_latent_concat.png")
        save_latent(mask_latent_concat, "mask_latent_concat.png")
        if is_train:
            # 학습 모드: 단일 timestep 노이즈 예측
            batch_size = masked_latent_concat.shape[0]
            num_timesteps = self.noise_scheduler.config.num_train_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
            t = torch.randint(0, num_timesteps, (batch_size,), device=self.device)

            noise = randn_tensor(masked_latent_concat.shape, generator=generator, device=self.device, dtype=self.weight_dtype)
            noisy_latent = self.noise_scheduler.add_noise(masked_latent_concat, noise, t)
            
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                masked_latent_concat = torch.cat([
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ])
                mask_latent_concat = torch.cat([mask_latent_concat] * 2)
                t = torch.cat([t]* 2, dim=0)
            
            non_inpainting_latent_model_input = (
                torch.cat([noisy_latent] * 2, dim=0) if do_classifier_free_guidance else noisy_latent
            )
            non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
            inpainting_latent_model_input = torch.cat(
                [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1
            )
            save_latent(noise, "added_noise.png")
            save_latent(noisy_latent, "noisy_latent.png")
            noise_pred = self.unet(
                inpainting_latent_model_input,
                t.to(self.device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = noise_pred.split(noise_pred.shape[concat_dim] // 2, dim=concat_dim)[0]
            noise = noise.split(noise.shape[concat_dim] // 2, dim=concat_dim)[0]
            # scaling_factor 적용
            noise = 1 / self.vae.config.scaling_factor * noise
            noise_pred = 1 / self.vae.config.scaling_factor * noise_pred
            return noise, noise_pred.to(dtype=self.weight_dtype)
        
        else:
            # 추론 모드: 풀 denoising 루프
            latents = randn_tensor(
                masked_latent_concat.shape,
                generator=generator,
                device=self.device,
                dtype=self.weight_dtype,
            )
            save_latent(latents, "initial_random_latent.png")

            self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps
            latents = latents * self.noise_scheduler.init_noise_sigma
            
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                masked_latent_concat = torch.cat([
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ])
                mask_latent_concat = torch.cat([mask_latent_concat] * 2)

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            i = 0
            for t in timesteps:
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1
                )
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # 10단계마다 중간 결과 저장
                if i % 10 == 0 or i == len(timesteps) - 1:
                    with torch.no_grad():
                        latent_to_decode = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
                        latent_to_decode = 1 / self.vae.config.scaling_factor * latent_to_decode
                        decoded_image = self.vae.decode(latent_to_decode.to(self.weight_dtype)).sample
                        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                        plt.figure(figsize=(5, 5))
                        plt.imshow(decoded_image[0].cpu().permute(1, 2, 0).float().numpy())
                        plt.title(f"Step {i}")
                        plt.axis('off')
                        plt.savefig(os.path.join("vis", f"step_{i}.png"))
                        plt.close()
                i += 1

            latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
            latents = 1 / self.vae.config.scaling_factor * latents
            with torch.no_grad():
                image = self.vae.decode(latents.to(self.weight_dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            save_image(image, "final_decoded_image.png")
            return numpy_to_pil(image)

# 기존 추론용 파이프라인 건들지 마시오
class CatVTONPipeline:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(base_ckpt, subfolder="feature_extractor")
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(base_ckpt, subfolder="safety_checker").to(device, dtype=weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(self.weight_dtype)
            )
        return image, has_nsfw_concept
    
    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -2  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model
                inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
                # predict the noise residual
                noise_pred= self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None, # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image
    
class CatVTONPix2PixPipeline(CatVTONPipeline):
    def auto_attn_ckpt_load(self, attn_ckpt, version):
        # TODO: Temperal fix for the model version
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, version, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, version, 'attention'))
    
    def check_inputs(self, image, condition_image, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(torch.Tensor):
            return image, condition_image
        image = resize_and_crop(image, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image

    @torch.no_grad()
    def __call__(
        self, 
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -1
        # Prepare inputs to Tensor
        image, condition_image = self.check_inputs(image, condition_image, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        # VAE encoding
        image_latent = compute_vae_encodings(image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        del image, condition_image
        # Concatenate latents
        condition_latent_concat = torch.cat([image_latent, condition_latent], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            condition_latent_concat.shape,
            generator=generator,
            device=condition_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            condition_latent_concat = torch.cat(
                [
                    torch.cat([image_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    condition_latent_concat,
                ]
            )

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                # prepare the input for the inpainting model
                p2p_latent_model_input = torch.cat([latent_model_input, condition_latent_concat], dim=1)
                # predict the noise residual
                noise_pred= self.unet(
                    p2p_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None, 
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)
        
        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(os.path.dirname(current_script_directory), 'resource', 'img', 'NSFW.jpg')
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image)
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image
