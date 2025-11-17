import argparse
import os
# import numpy as np
import random
import time
import datetime
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from diffusers import DDPMPipeline, DDPMScheduler
from datasets import caption_dataset_llava, save_result
from models import LLaVA_F3
import utils
from attacks import AutoAttack_LVLM
import XTransferBench
import XTransferBench.zoo

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model


def apply_random_masking(image_tensor: torch.Tensor, perturbation_percentage: float) -> torch.Tensor:
    """
    Randomly masks (sets to zero) a percentage of pixels in a batched image tensor.

    Args:
        image_tensor: Tensor shaped (B, C, H, W) with values in [0, 1].
        perturbation_percentage: Percentage of pixels to mask in [0, 100].
    """
    if perturbation_percentage <= 0.0:
        return image_tensor
    clipped = min(max(perturbation_percentage, 0.0), 100.0)
    if clipped == 100.0:
        return torch.zeros_like(image_tensor)

    prob = clipped / 100.0
    mask = torch.rand(
        image_tensor.shape[0],
        1,
        image_tensor.shape[2],
        image_tensor.shape[3],
        device=image_tensor.device,
        dtype=image_tensor.dtype,
    ) < prob
    masked_image = image_tensor.clone()
    masked_image[mask.expand_as(masked_image)] = 0
    return masked_image


def load_diffpure_components(model_id: str, device: torch.device) -> Tuple[torch.nn.Module, DDPMScheduler]:
    pipeline = DDPMPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.eval()
    return pipeline.unet, pipeline.scheduler


def purify_with_diffpure(
    adv_image: torch.Tensor,
    unet: torch.nn.Module,
    scheduler,
    t_star: float,
    total_steps: int,
) -> torch.Tensor:
    if adv_image.shape[0] != 1:
        raise ValueError("DiffPure currently supports batch size 1.")

    t_star = float(max(0.0, min(1.0, t_star)))
    total_steps = max(1, total_steps)
    scheduler.set_timesteps(total_steps)

    unet_param = next(unet.parameters())
    unet_device = unet_param.device
    unet_dtype = unet_param.dtype

    adv = adv_image.to(device=unet_device, dtype=unet_dtype)
    adv = adv.clamp(0.0, 1.0) * 2.0 - 1.0

    timesteps = scheduler.timesteps
    effective_steps = len(timesteps)
    t_index = max(1, min(effective_steps, int(t_star * effective_steps) or 1))
    start_idx = effective_steps - t_index
    start_idx = max(0, min(len(timesteps) - 1, start_idx))
    t = timesteps[start_idx]
    with torch.no_grad():
        noise = torch.randn_like(adv)
        current = scheduler.add_noise(adv, noise, t)

        for timestep in timesteps[start_idx:]:
            model_output = unet(current, timestep).sample
            scheduler_output = scheduler.step(model_output, timestep, current)
            current = scheduler_output.prev_sample

    purified = (current / 2.0 + 0.5).clamp(0.0, 1.0)
    return purified.to(dtype=adv_image.dtype, device=adv_image.device)

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    # np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model_path = os.path.join(args.huggingface_root, "llava-v1.5-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=device, device=device)
    image_processor.do_normalize = False

    #### Dataset ####
    print("Creating MSCOCO caption dataset")
    datasets = caption_dataset_llava(args.coco_images_dir, tokenizer, image_processor, model.config, conv_mode="vicuna_v1", limit=args.limit)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True, sampler=sampler, shuffle=False, collate_fn=None, drop_last=False)

    print("Creating model")
    model = LLaVA_F3(model_path, tokenizer, model, image_processor, device, args.f3_alpha, args.f3_beta, args.f3_v3)

    xtransfer_attacker = None
    diffpure_unet = None
    diffpure_scheduler = None
    if args.attack_method=="unadaptive":
        attack = AutoAttack_LVLM(model.get_logit, eps=args.aa_eps, device=device, version="standard")
    elif args.attack_method=="adaptive":
        attack = AutoAttack_LVLM(model.get_logit_with_purify_f3, eps=args.aa_eps, device=device, version="rand")
    elif args.attack_method=="targeted":
        # Use XTransferBench targeted attacker per user spec
        if args.attacker_name is None or args.threat_model is None:
            raise ValueError("targeted attack requires --attacker_name and --threat_model for XTransferBench")
        xtransfer_attacker = XTransferBench.zoo.load_attacker(args.threat_model, args.attacker_name).to(device)
        if hasattr(xtransfer_attacker, 'delta'):
            # args.aa_eps is already normalized to [0,1]
            xtransfer_attacker.interpolate_epsilon(args.aa_eps)
        attack = None
    else:
        attack = None

    if args.purify_method == "diffpure":
        diffpure_unet, diffpure_scheduler = load_diffpure_components(args.diffpure_model_id, device)
    print("Start captioning")
    start_time = time.time()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Captioning with attack={args.attack_method}:"
    print_freq = 5
    
    clean_result = []
    attack_result = []
    purify_result = []

    for n, (image_id, image, input_ids, image_size) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device=device, non_blocking=True)
        clean_image = image.to(dtype=torch.float16, device=device, non_blocking=True)
        input_ids = input_ids.to(device=device)
        # caption = model.predict(clean_image, input_ids, image_size)
        # clean_result.append({
        #     "image_id": str(image_id[0]),
        #     "caption": caption
        # })

        # torchvision.utils.save_image(image, os.path.join(args.image_dir, "{}_clean.png".format(image_id[0])))

        if attack or xtransfer_attacker is not None:
            model.input_ids = input_ids.to(device=device).long()
            model.input_ids.requires_grad = False
            model.image_sizes = image_size
            image.requires_grad = True
            if xtransfer_attacker is not None:
                # Resize to 224 for attacker, attack, then resize back
                orig_h, orig_w = image.shape[-2], image.shape[-1]
                img_224 = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
                adv_224 = xtransfer_attacker.attack(img_224)
                # Some attackers may return a tuple
                if isinstance(adv_224, (tuple, list)):
                    adv_224 = adv_224[0]
                adv_image = F.interpolate(adv_224, size=(orig_h, orig_w), mode='bilinear', align_corners=False).clamp(0, 1)
            else:
                clean_logit = model.get_logit(image)
                clean_label = torch.argmax(clean_logit, dim=1).to(device)
                adv_image = attack.run_standard_evaluation(image, clean_label, bs=1, return_labels=False, state_path=None, strong=True)

            # attack_caption = model.predict(adv_image, input_ids, image_size)
            # Purification
            if args.purify_method == "f3":
                purify_image = model.purify_f3(adv_image)
            elif args.purify_method == "noise":
                noise = (torch.rand_like(adv_image) * 2.0 - 1.0) * args.f3_beta
                purify_image = torch.clamp(adv_image + noise, 0.0, 1.0)
            elif args.purify_method == "mask":
                purify_image = apply_random_masking(adv_image, args.mask_percentage)
            elif args.purify_method == "diffpure":
                if diffpure_unet is None or diffpure_scheduler is None:
                    raise RuntimeError("DiffPure components are not initialized.")
                purify_image = purify_with_diffpure(
                    adv_image,
                    diffpure_unet,
                    diffpure_scheduler,
                    args.diffpure_t_star,
                    args.diffpure_total_steps,
                )
            elif args.purify_method == "none":
                purify_image = adv_image
            else:
                raise ValueError(f"Unknown purify_method: {args.purify_method}")
            purify_caption = model.predict(purify_image, input_ids, image_size)

            # attack_result.append({
            #     "image_id": str(image_id[0]),
            #     "caption": attack_caption
            # })
            # torchvision.utils.save_image(adv_image, os.path.join(args.image_dir, "{}_adv.png".format(image_id[0])))
            purify_result.append({
                "image_id": str(image_id[0]),
                "caption": purify_caption
            })
            torchvision.utils.save_image(purify_image, os.path.join(args.image_dir, "{}_purify_{}.png".format(image_id[0], args.purify_method)))

        if (n + 1) >= args.limit:
            break

    # save_result(clean_result, args.output_dir, 'captions_clean')
    # if len(attack_result)>0:
        # save_result(attack_result, args.output_dir, 'captions_adv')

    save_result(purify_result, args.output_dir, 'captions_purify_{}'.format(args.purify_method))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    
    # default config
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # model config
    parser.add_argument('--huggingface_root', default="liuhaotian", type=str)
   
    # dataset config
    parser.add_argument('--coco_images_dir', required=True, type=str, help='Path to MSCOCO images directory, e.g., val2014')
    parser.add_argument('--limit', default=1000, type=int, help='Number of images to caption')

    # attack config
    parser.add_argument('--attack_method', type=str, default="none", choices=["none", "unadaptive", "adaptive", "targeted"])
    parser.add_argument('--attacker_name', type=str, default='xtransfer_large_linf_eps12_targeted_template8', help='XTransferBench attacker name for targeted attacks')
    parser.add_argument('--threat_model', type=str, default='linf_targeted', help='XTransferBench threat model for targeted attacks')
    
    # f3 confg
    parser.add_argument('--f3_alpha', default=16, type=int)
    parser.add_argument('--f3_beta', default=32, type=int)
    parser.add_argument('--f3_v3', action="store_true")

    # aa config
    parser.add_argument('--aa_eps', type=int, default=16)

    # output config
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--purify_method', type=str, default='f3', choices=['f3', 'noise', 'mask', 'diffpure', 'none'], help='Purification method applied to adversarial image')
    parser.add_argument('--diffpure_model_id', type=str, default='google/ddpm-cifar10-32', help='Hugging Face model id for the DiffPure DDPM')
    parser.add_argument('--diffpure_t_star', type=float, default=0.1, help='Normalized timestep (0-1] for DiffPure forward diffusion')
    parser.add_argument('--diffpure_total_steps', type=int, default=1000, help='Number of DDPM steps to run for DiffPure purification')
    parser.add_argument('--mask_percentage', type=float, default=10.0, help='Percentage of pixels to zero out when using mask purification')

    args = parser.parse_args()

    args.f3_alpha = float(args.f3_alpha)/255
    args.f3_beta = float(args.f3_beta)/255
    args.aa_eps = float(args.aa_eps)/255
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.image_dir = os.path.join(args.output_dir, "image")
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    main(args)


