import argparse
import os
import random
import math

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from optims.rank_wsd_schedulers import RankAwareWarmupStableLinearScheduler
from optims.muon import SingleDeviceMuonWithAuxAdam
from optims.lr_inc import SingleDeviceFinalIncWithAuxAdam
from optims.lr_sign import SingleDeviceSignWithAuxAdam
from optims.lr_svd import SingleDeviceLrSVDWithAuxAdam

from optims.auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam

from models import (
    GaussFFN,
    GaussMLP,
    ReluFFN,
    ReluMLP,
    ReluPosEncoding,
    SirenMLP,
    WireMLP,
    FinerMLP,
    WireRealMLP,
)


def set_seed(seed):
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_image(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)


def parse_list(param_str, num_layers):
    """
    Parse the parameter from command line.

    Args:
        param_str: String from command line (e.g., "0.05" or "0.1,0.05,0.02,0.01")
        num_layers: Number of layers with activations (excludes final output layer)

    Returns:
        Single float or list of floats
    """
    if ',' in param_str:
        param_values = [float(x.strip()) for x in param_str.split(',')]
        if len(param_values) != (num_layers - 1):
            raise ValueError(
                f"Number of parameter values ({len(param_values)}) must match num_layers -1 ({num_layers-1})."
            )
        return param_values
    else:
        param_value = float(param_str)
        param_values = [param_value] * (num_layers - 1)
        return param_values


def get_coordinates(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return coords


def psnr(img1, img2, max_val=1.0):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        max_val (float): The maximum possible pixel value of the images
                         (e.g., 1.0 for normalized floats, 255.0 for 8-bit images).
    """
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_val


def compute_lpips(img1, img2, lpips_model, device):
    """Compute LPIPS between two images"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    img1 = (img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    img2 = (img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)

    with torch.no_grad():
        lpips_val = lpips_model(img1, img2)

    return lpips_val.item()


def create_comparison_image(gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
    """Create side-by-side comparison of ground truth and reconstructed image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(np.clip(recon_img, 0, 1))
    axes[1].set_title(f'Reconstruction\nEpoch {epoch}', fontsize=12)
    axes[1].axis('off')

    diff = np.abs(gt_img - recon_img)
    diff_amplified = np.clip(diff, 0, 1)
    axes[2].imshow(diff_amplified, cmap='viridis')

    title = f'Difference \nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}'

    axes[2].set_title(title, fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def get_output_dir(args):
    base_dir = args.folder_name if args.folder_name is not None else 'results'
    output_dir = os.path.join(base_dir, args.model, args.optimizer)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_image(image_array, save_path):
    image_array = np.clip(image_array, 0, 1)

    if image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = image_array.squeeze(-1)

    image_uint8 = (image_array * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(save_path)


def save_args(args, output_dir):
    args_path = os.path.join(output_dir, 'args.txt')
    with open(args_path, 'w', encoding='utf-8') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def save_readme(output_dir, final_psnr, final_ssim, final_lpips):
    readme_path = os.path.join(output_dir, 'Readme.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('Final Evaluation Metrics\n')
        f.write('========================\n')
        f.write(f'PSNR: {final_psnr:.6f}\n')
        f.write(f'SSIM: {final_ssim:.6f}\n')
        f.write(f'LPIPS: {final_lpips:.6f}\n')


def save_metrics_csv(metrics, layer_metrics, log_records, output_dir):
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    if log_records:
        log_df = pd.DataFrame(log_records)
        log_df.to_csv(os.path.join(output_dir, 'train_log.csv'), index=False)

    if layer_metrics:
        layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
        for key, values in layer_metrics.items():
            layer_df[key] = values
        layer_df.to_csv(os.path.join(output_dir, 'layer_metrics.csv'), index=False)


def modcrop(img, scale):
    """Crop image so that height/width are divisible by scale (standard SR modcrop)."""
    if scale <= 1:
        return img

    h, w, _ = img.shape
    h_mod = h - (h % scale)
    w_mod = w - (w % scale)

    if h_mod != h or w_mod != w:
        print(f'Applying modcrop: ({h}, {w}) -> ({h_mod}, {w_mod}) for scale x{scale}.')
        img = img[:h_mod, :w_mod, :]

    return img


def forward_in_chunks(model, coords, chunk_size):
    """Memory-safe forward used for SISR HR evaluation on large images."""
    if chunk_size is None or chunk_size <= 0 or coords.shape[0] <= chunk_size:
        return model(coords)

    outputs = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        outputs.append(model(coords[start:end]))
    return torch.cat(outputs, dim=0)


def save_bicubic_baseline(img_lr_bchw, h_hr, w_hr, output_dir):
    bicubic_up = F.interpolate(
        img_lr_bchw,
        size=(h_hr, w_hr),
        mode='bicubic',
    )
    bicubic_np = bicubic_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    save_image(np.clip(bicubic_np, 0, 1), os.path.join(output_dir, 'bicubic_baseline.png'))
    return bicubic_np


def build_model(args, c):
    if args.model == 'relu_ffn':
        model = ReluFFN(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            sigma=args.fourier_sigma,
        )
    elif args.model == 'relu_mlp':
        model = ReluMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
        )
    elif args.model == 'relu_pos_enc':
        model = ReluPosEncoding(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
        )
    elif args.model == 'gauss_ffn':
        model = GaussFFN(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            sigma=args.fourier_sigma,
            a=args.gauss_scale,
        )
    elif args.model == 'gauss_mlp':
        model = GaussMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            a=args.gauss_scale,
        )
    elif args.model == 'siren_mlp':
        model = SirenMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.siren_omega,
        )
    elif args.model == 'wire_mlp':
        model = WireMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.wire_omega,
            sigma=args.wire_sigma,
        )
    elif args.model == 'real_wire':
        model = WireRealMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.wire_omega,
            sigma=args.wire_sigma,
        )
    elif args.model == 'finer_mlp':
        model = FinerMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.finer_omega,
            init_bias=args.finer_init_bias,
            bias_scale=args.finer_bias_scale,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model


AUTO_COS_INC_CANONICAL_NAME = 'auto_cos_inc'
AUTO_COS_INC_OPTIMIZER_ALIASES = {
    'auto_cos_inc',
    'auto-cos-inc',
    'auto_cos_inc_rank',
    'auto-cos-inc-rank',
    'lr-sign10-rsclF',
}
LOWRANK_OPTIMIZER_NAMES = {'lr-inc', 'lr-sign', 'lr-svd'}
LOWRANK_LIKE_OPTIMIZERS = {'muon'} | LOWRANK_OPTIMIZER_NAMES | {AUTO_COS_INC_CANONICAL_NAME}


def normalize_optimizer_name(name):
    if name in AUTO_COS_INC_OPTIMIZER_ALIASES:
        return AUTO_COS_INC_CANONICAL_NAME
    return name


def resolve_auto_cos_inc_rank_args(args):
    """Resolve and validate rank-growth arguments for auto_cos_inc_rank.py."""
    if args.rank <= 0:
        raise ValueError('--rank must be > 0.')
    if args.rank_start is None:
        args.rank_start = int(args.rank)
    if args.rank_end is None:
        args.rank_end = int(args.rank_start)

    args.rank_start = int(args.rank_start)
    args.rank_end = int(args.rank_end)

    if args.rank_start <= 0:
        raise ValueError('--rank_start must be > 0 when provided.')
    if args.rank_end <= 0:
        raise ValueError('--rank_end must be > 0 when provided.')
    if args.rank_end < args.rank_start:
        raise ValueError('--rank_end must be >= --rank_start for cosine rank growth.')

    if args.rank_warmup_steps is None:
        if args.scheduler == 'rank_wsd':
            if args.rank_wsd_decay_start_step is None:
                args.rank_warmup_steps = max(1, int(round(0.8 * args.epochs)))
            else:
                args.rank_warmup_steps = max(1, int(args.rank_wsd_decay_start_step))
        else:
            args.rank_warmup_steps = max(1, int(args.epochs))
    else:
        args.rank_warmup_steps = int(args.rank_warmup_steps)

    if args.rank_warmup_steps <= 0:
        raise ValueError('--rank_warmup_steps must be > 0 when provided.')
    if args.rank_oversample < 0:
        raise ValueError('--rank_oversample must be >= 0.')
    if args.muon_ns_steps <= 0:
        raise ValueError('--muon_ns_steps must be > 0.')
    if not (0.0 < args.init_energy <= 1.0):
        raise ValueError('--init_energy must be in (0, 1].')
    if args.init_probe_steps <= 0:
        raise ValueError('--init_probe_steps must be > 0.')
    if args.init_round_multiple <= 0:
        raise ValueError('--init_round_multiple must be > 0.')

    if args.rank_start < args.rank:
        print(
            'WARNING: --rank_start is smaller than --rank. '
            'auto_cos_inc_rank.py floors the applied rank by --rank, so the effective start rank will be --rank.'
        )
    if args.rank_end < args.rank:
        print(
            'WARNING: --rank_end is smaller than --rank. '
            'auto_cos_inc_rank.py will floor every applied rank by --rank; rank growth will be disabled.'
        )


def build_optimizer(args, model):
    """
    This optimizer construction intentionally mirrors fit_image_optim_modified.py.
    The only goal here is optimizer parity between overfitting and SISR runs.
    """
    if args.optimizer == 'muon':
        print("INFO: Setting up Muon optimizer.")
        muon_params = []
        other_params = []

        first_layer_muon_models = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}
        is_special_model = args.model in first_layer_muon_models

        if hasattr(model, 'mlp') and isinstance(model.mlp, torch.nn.Sequential):
            num_mlp_layers = len(model.mlp)

            for name, param in model.named_parameters():
                is_muon_target = False

                if 'mlp' in name and 'weight' in name and param.ndim >= 2:
                    try:
                        layer_idx = int(name.split('.')[1])

                        is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                        is_first_layer_for_muon = (
                            is_special_model
                            and args.optimize_first_layer_with_muon
                            and layer_idx == 0
                        )

                        if is_hidden_layer or is_first_layer_for_muon:
                            is_muon_target = True

                    except (ValueError, IndexError):
                        pass

                if is_muon_target:
                    muon_params.append(param)
                else:
                    other_params.append(param)
        else:
            print("WARNING: Model does not have a standard 'mlp' attribute. Cannot separate params for Muon.")
            other_params = list(model.parameters())

        if muon_params:
            if is_special_model and args.optimize_first_layer_with_muon:
                print(
                    "INFO: --optimize_first_layer_with_muon=True. "
                    "The first MLP layer will also be optimized by Muon."
                )

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=args.muon_lr,
                    weight_decay=args.muon_weight_decay,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    weight_decay=args.muon_aux_weight_decay,
                ),
            ]

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print(
                f"INFO: Muon optimizer configured. "
                f"Muon params: {len(muon_params)}, Other params: {len(other_params)}."
            )

        else:
            raise ValueError(
                f"Muon optimizer was selected (model: {args.model}), but no suitable "
                f"parameters were identified for Muon."
            )
    elif args.optimizer in LOWRANK_OPTIMIZER_NAMES or args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
        print("INFO: Setting up lowrank-* optimizer.")
        muon_params = []
        other_params = []

        first_layer_muon_models = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}
        is_special_model = args.model in first_layer_muon_models

        if hasattr(model, 'mlp') and isinstance(model.mlp, torch.nn.Sequential):
            num_mlp_layers = len(model.mlp)

            for name, param in model.named_parameters():
                is_muon_target = False

                if 'mlp' in name and 'weight' in name and param.ndim >= 2:
                    try:
                        layer_idx = int(name.split('.')[1])

                        is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                        is_first_layer_for_muon = (
                            is_special_model
                            and args.optimize_first_layer_with_muon
                            and layer_idx == 0
                        )

                        if is_hidden_layer or is_first_layer_for_muon:
                            is_muon_target = True

                    except (ValueError, IndexError):
                        pass

                if is_muon_target:
                    muon_params.append(param)
                else:
                    other_params.append(param)
        else:
            print("WARNING: Model does not have a standard 'mlp' attribute. Cannot separate params for Muon.")
            other_params = list(model.parameters())

        if muon_params:
            if is_special_model and args.optimize_first_layer_with_muon:
                print(
                    "INFO: --optimize_first_layer_with_muon=True. "
                    "The first MLP layer will also be optimized by Muon."
                )

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=args.muon_lr,
                    weight_decay=args.muon_weight_decay,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    weight_decay=args.muon_aux_weight_decay,
                ),
            ]

            if args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
                param_groups[0].update(
                    momentum=args.muon_momentum,
                    ns_steps=args.muon_ns_steps,
                    nesterov=not args.muon_no_nesterov,
                    rank=args.rank,
                    rank_start=args.rank_start,
                    rank_end=args.rank_end,
                    warmup_steps=args.rank_warmup_steps,
                    oversample=args.rank_oversample,
                    lowrank_rescale=args.lowrank_rescale,
                    eps=args.lowrank_eps,
                    small_ns_bfloat16=args.small_ns_bfloat16,
                    auto_init_rank_start=args.auto_init_rank_start,
                    init_probe_steps=args.init_probe_steps,
                    init_energy=args.init_energy,
                    init_round_multiple=args.init_round_multiple,
                )

            if args.optimizer == 'lr-inc':
                optimizer = SingleDeviceFinalIncWithAuxAdam(param_groups)
                optimizer_name = 'lr-inc'
            elif args.optimizer == 'lr-sign':
                optimizer = SingleDeviceSignWithAuxAdam(param_groups)
                optimizer_name = 'lr-sign'
            elif args.optimizer == 'lr-svd':
                optimizer = SingleDeviceLrSVDWithAuxAdam(param_groups)
                optimizer_name = 'lr-svd'
            elif args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
                optimizer = SingleDeviceAutoCosIncWithAuxAdam(param_groups)
                optimizer_name = AUTO_COS_INC_CANONICAL_NAME
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
                optimizer_name = args.optimizer
            print(
                f"INFO: {optimizer_name} optimizer configured. "
                f"Muon params: {len(muon_params)}, Other params: {len(other_params)}."
            )

        else:
            raise ValueError(
                f"Muon optimizer was selected (model: {args.model}), but no suitable "
                f"parameters were identified for Muon."
            )

    elif args.optimizer == 'lbfgs':
        print("INFO: Using L-BFGS optimizer.")
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
        )

    elif args.optimizer == 'adam':
        print("INFO: Using Adam optimizer.")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    return optimizer


def is_rank_wsd_scheduler(scheduler):
    return (
        RankAwareWarmupStableLinearScheduler is not None
        and isinstance(scheduler, RankAwareWarmupStableLinearScheduler)
    )


def _infer_rank_wsd_base_lrs(args, optimizer):
    base_lr_adam = None
    base_lr_muon = None

    for group in optimizer.param_groups:
        group_lr = group.get('lr', None)
        if group_lr is None:
            continue

        if group.get('use_muon', False):
            if base_lr_muon is None:
                base_lr_muon = float(group_lr)
        else:
            if base_lr_adam is None:
                base_lr_adam = float(group_lr)

    if base_lr_adam is None:
        base_lr_adam = float(args.lr)
    if base_lr_muon is None:
        base_lr_muon = float(getattr(args, 'muon_lr', base_lr_adam))

    return base_lr_adam, base_lr_muon


def _infer_optimizer_N_rand(optimizer):
    for attr_name in ('N_rand', 'n_rand'):
        if hasattr(optimizer, attr_name):
            try:
                value = getattr(optimizer, attr_name)
                if value is not None:
                    value = int(value)
                    if value > 0:
                        return value
            except (TypeError, ValueError):
                pass

    for group in optimizer.param_groups:
        for key in ('N_rand', 'n_rand'):
            if key in group:
                try:
                    value = int(group[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    defaults = getattr(optimizer, 'defaults', {})
    if isinstance(defaults, dict):
        for key in ('N_rand', 'n_rand'):
            if key in defaults:
                try:
                    value = int(defaults[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    return None


def _resolve_rank_wsd_base_N_rand(args, optimizer):
    if args.rank_wsd_base_N_rand is not None:
        if args.rank_wsd_base_N_rand <= 0:
            raise ValueError('--rank_wsd_base_N_rand must be > 0 when provided.')
        return int(args.rank_wsd_base_N_rand)

    inferred_N_rand = _infer_optimizer_N_rand(optimizer)
    if inferred_N_rand is not None:
        return inferred_N_rand

    return 1


def _safe_setattr(obj, attr_name, value):
    try:
        setattr(obj, attr_name, value)
        return True
    except (AttributeError, TypeError):
        return False


def _apply_N_rand_to_optimizer(optimizer, N_rand):
    N_rand = int(N_rand)

    for setter_name in ('set_N_rand', 'set_n_rand'):
        setter = getattr(optimizer, setter_name, None)
        if callable(setter):
            try:
                setter(N_rand)
            except TypeError:
                pass

    for attr_name in ('N_rand', 'n_rand'):
        if hasattr(optimizer, attr_name):
            _safe_setattr(optimizer, attr_name, N_rand)

    for group in optimizer.param_groups:
        for key in ('N_rand', 'n_rand'):
            if key in group:
                group[key] = N_rand

    defaults = getattr(optimizer, 'defaults', None)
    if isinstance(defaults, dict):
        for key in ('N_rand', 'n_rand'):
            if key in defaults:
                defaults[key] = N_rand


def apply_rank_wsd_scheduler_step(scheduler, optimizer, global_step):
    state = scheduler.step(global_step)

    for group in optimizer.param_groups:
        if group.get('use_muon', False):
            group['lr'] = state['lr_muon']
        else:
            group['lr'] = state['lr_adam']

    _apply_N_rand_to_optimizer(optimizer, state['N_rand'])
    return state


def build_scheduler(args, optimizer):
    scheduler = None

    if args.optimizer == 'lbfgs':
        print("INFO: Schedulers are disabled for L-BFGS optimizer.")
        return scheduler

    if args.scheduler == 'none':
        print("INFO: Scheduler is disabled.")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-6)
        print(f"Using CosineAnnealingLR scheduler with T_max={args.T_max}")
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(f"Using StepLR scheduler with step_size={args.step_size} and gamma={args.gamma}")
    elif args.scheduler == 'rank_wsd':
        if RankAwareWarmupStableLinearScheduler is None:
            raise ImportError(
                "--scheduler rank_wsd requires rank_wsd_schedulers.py either in the same directory "
                "or importable as optims.rank_wsd_schedulers."
            )

        base_lr_adam, base_lr_muon = _infer_rank_wsd_base_lrs(args, optimizer)
        base_N_rand = _resolve_rank_wsd_base_N_rand(args, optimizer)

        scheduler = RankAwareWarmupStableLinearScheduler(
            base_lr_adam=base_lr_adam,
            base_lr_muon=base_lr_muon,
            total_iters=args.epochs,
            base_N_rand=base_N_rand,
            warmup_steps=args.rank_wsd_warmup_steps,
            decay_start_step=args.rank_wsd_decay_start_step,
            min_lr_ratio=args.rank_wsd_min_lr_ratio,
        )
        print(scheduler.describe())

        if args.optimizer not in LOWRANK_LIKE_OPTIMIZERS:
            print(
                "WARNING: rank_wsd is intended for Muon/lowrank-style optimizers. "
                "For this optimizer it will schedule LR only unless the optimizer exposes N_rand/n_rand."
            )
    else:
        raise ValueError(f"Unsupported scheduler type: {args.scheduler}")

    return scheduler


def get_auto_cos_inc_rank_state(optimizer):
    for group in optimizer.param_groups:
        if group.get('use_muon', False):
            return {
                'current_rank': group.get('current_rank'),
                'current_target_rank': group.get('current_target_rank'),
                'rank_floor': group.get('rank'),
                'rank_start': group.get('rank_start'),
                'rank_end': group.get('rank_end'),
                'rank_warmup_steps': group.get('warmup_steps'),
                'auto_rank_start_final': group.get('auto_rank_start_final'),
                'current_method': group.get('current_method'),
            }
    return {}


def make_super_resolution_data(img_hr, scale_factor, device, output_dir):
    """
    SISR path designed to follow the SIREN image-fitting spirit more closely:
    train on LR coordinate/value pairs only, then query the learned continuous field
    on the denser HR grid for evaluation.
    """
    img_hr = modcrop(img_hr, scale_factor)
    h_hr, w_hr, c = img_hr.shape

    img_hr_bchw = img_hr.permute(2, 0, 1).unsqueeze(0).to(device)
    h_lr, w_lr = h_hr // scale_factor, w_hr // scale_factor

    img_lr_bchw = F.interpolate(
        img_hr_bchw,
        size=(h_lr, w_lr),
        mode='bicubic',
        antialias=True,
    ).detach()

    img_lr = img_lr_bchw.squeeze(0).permute(1, 2, 0).cpu()
    save_image(np.clip(img_hr.numpy(), 0, 1), os.path.join(output_dir, 'ground_truth.png'))
    save_image(np.clip(img_lr.numpy(), 0, 1), os.path.join(output_dir, 'low_res_input.png'))
    save_bicubic_baseline(img_lr_bchw, h_hr, w_hr, output_dir)

    coords_lr = get_coordinates(h_lr, w_lr).to(device)
    target_lr = img_lr.view(-1, c).to(device)

    coords_hr = get_coordinates(h_hr, w_hr).to(device)

    return {
        'img_hr': img_hr,
        'img_lr': img_lr,
        'coords_hr': coords_hr,
        'coords_lr': coords_lr,
        'target_lr': target_lr,
        'h_hr': h_hr,
        'w_hr': w_hr,
        'h_lr': h_lr,
        'w_lr': w_lr,
        'c': c,
    }


def compute_chunked_train_loss(model, coords, target, chunk_size):
    total_loss = None
    total_numel = target.numel()

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        pred_chunk = model(coords[start:end])
        target_chunk = target[start:end]
        chunk_loss = F.mse_loss(pred_chunk, target_chunk, reduction='sum')
        scaled_loss = chunk_loss / total_numel
        scaled_loss.backward()

        if total_loss is None:
            total_loss = chunk_loss.detach()
        else:
            total_loss = total_loss + chunk_loss.detach()

    mean_loss = total_loss / total_numel
    return mean_loss


def train_model(args, output_dir):
    """Train the neural field model on the given image with specified parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lpips_model = lpips.LPIPS(net='alex').to(device)

    img = load_image(args.image)
    evolution_dir = os.path.join(output_dir, 'image_evolution')
    is_super_resolution = args.task == 'super_resolution'
    is_inpainting_task = (args.task == 'fitting') and (args.inpainting_ratio < 1.0)

    if is_super_resolution:
        print(f"Performing {args.scale_factor}x super-resolution task.")
        sr_data = make_super_resolution_data(img, args.scale_factor, device, output_dir)

        img_hr = sr_data['img_hr']
        h, w, c = sr_data['h_hr'], sr_data['w_hr'], sr_data['c']
        coords = sr_data['coords_hr']
        train_coords = sr_data['coords_lr']
        train_target = sr_data['target_lr']
        gt_img_np = img_hr.numpy()
    else:
        h, w, c = img.shape
        coords = get_coordinates(h, w)
        target = img.view(-1, c)
        gt_img_np = img.numpy()
        save_image(gt_img_np, os.path.join(output_dir, 'ground_truth.png'))

        if is_inpainting_task:
            print(f"Performing inpainting task. Training on {args.inpainting_ratio * 100:.2f}% of pixels.")
            num_pixels = h * w
            num_train_pixels = int(num_pixels * args.inpainting_ratio)

            indices = torch.randperm(num_pixels)
            train_indices = indices[:num_train_pixels]
            test_indices = indices[num_train_pixels:]

            train_coords = coords[train_indices].to(device)
            train_target = target[train_indices].to(device)

            test_coords = coords[test_indices].to(device)
            test_target = target[test_indices].to(device)

            mask = torch.ones(num_pixels, 1) * 0.2
            mask[train_indices] = 1.0
            mask_img = mask.view(h, w, 1).cpu().numpy()
            save_image(mask_img, os.path.join(output_dir, 'training_mask.png'))
        else:
            print("Performing overfitting task on all pixels.")
            train_coords = coords.to(device)
            train_target = target.to(device)

        coords = coords.to(device)

    model = build_model(args, c)
    print(f"Using model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters.")
    print(model)

    model = model.to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    metrics = {
        'epochs': [], 'full_psnr': [], 'ssim': [], 'lpips': [],
        'train_psnr': [], 'test_psnr': []
    }
    layer_metrics = {}
    log_records = []

    for epoch in range(args.epochs):
        model.train()

        rank_wsd_state = None
        if is_rank_wsd_scheduler(scheduler):
            rank_wsd_state = apply_rank_wsd_scheduler_step(scheduler, optimizer, epoch)

        if args.optimizer == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)

        else:
            optimizer.zero_grad()

            if is_super_resolution and args.sr_train_chunk_size > 0 and train_coords.shape[0] > args.sr_train_chunk_size:
                loss = compute_chunked_train_loss(model, train_coords, train_target, args.sr_train_chunk_size)
            else:
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()

            optimizer.step()

        if scheduler is not None and not is_rank_wsd_scheduler(scheduler):
            scheduler.step()

        if epoch % args.log_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                if is_super_resolution:
                    full_pred = forward_in_chunks(model, coords, args.sr_eval_chunk_size)
                else:
                    full_pred = model(coords)

                pred_img = full_pred.view(h, w, c).cpu().numpy()
                target_img = gt_img_np

                full_psnr_val = psnr(torch.tensor(pred_img), torch.tensor(target_img)).item()
                ssim_val = ssim(target_img, pred_img, channel_axis=-1, data_range=1.0)
                lpips_val = compute_lpips(target_img, pred_img, lpips_model, device)

                if is_super_resolution and args.sr_train_chunk_size > 0 and train_coords.shape[0] > args.sr_train_chunk_size:
                    train_pred = forward_in_chunks(model, train_coords, args.sr_eval_chunk_size)
                else:
                    train_pred = model(train_coords)

                train_psnr_val = psnr(train_pred, train_target).item()

                test_psnr_val = 0.0
                if is_inpainting_task:
                    test_psnr_val = psnr(model(test_coords), test_target).item()

                metrics['epochs'].append(epoch)
                metrics['full_psnr'].append(full_psnr_val)
                metrics['ssim'].append(ssim_val)
                metrics['lpips'].append(lpips_val)
                metrics['train_psnr'].append(train_psnr_val)
                metrics['test_psnr'].append(test_psnr_val)

                if is_inpainting_task:
                    print(
                        f"Epoch {epoch:4d}: Loss={loss.item():.6f}, TrainPSNR={train_psnr_val:.2f}, "
                        f"TestPSNR={test_psnr_val:.2f}, FullPSNR={full_psnr_val:.2f}, "
                        f"SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}"
                    )
                elif is_super_resolution:
                    print(
                        f"Epoch {epoch:4d}: Loss(LR)={loss.item():.6f}, TrainPSNR(LR)={train_psnr_val:.2f}, "
                        f"FullPSNR(HR)={full_psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d}: Loss={loss.item():.6f}, PSNR={full_psnr_val:.2f}, "
                        f"SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}"
                    )

                log_dict = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'full_psnr': full_psnr_val,
                    'ssim': ssim_val,
                    'lpips': lpips_val,
                    'train_psnr': train_psnr_val,
                }

                if args.optimizer in LOWRANK_LIKE_OPTIMIZERS:
                    log_dict['learning_rate_muon'] = optimizer.param_groups[0]['lr']
                    log_dict['learning_rate_aux'] = optimizer.param_groups[1]['lr']
                else:
                    log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

                if args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
                    rank_state = get_auto_cos_inc_rank_state(optimizer)
                    for key, value in rank_state.items():
                        log_dict[f'auto_cos_inc/{key}'] = value

                if rank_wsd_state is not None:
                    log_dict['rank_wsd_phase'] = rank_wsd_state['phase']
                    log_dict['rank_wsd_phase_name'] = rank_wsd_state['phase_name']
                    log_dict['rank_wsd_lr_ratio'] = rank_wsd_state['lr_ratio']
                    log_dict['rank_wsd_N_rand'] = rank_wsd_state['N_rand']

                if is_inpainting_task:
                    log_dict['test_psnr'] = test_psnr_val

                if args.log_image_evolution:
                    os.makedirs(evolution_dir, exist_ok=True)
                    comparison_fig = create_comparison_image(
                        target_img, pred_img, epoch, full_psnr_val, ssim_val, lpips_val
                    )
                    comparison_fig.savefig(
                        os.path.join(evolution_dir, f'comparison_epoch_{epoch:06d}.png'),
                        bbox_inches='tight'
                    )
                    save_image(
                        np.clip(pred_img, 0, 1),
                        os.path.join(evolution_dir, f'reconstruction_epoch_{epoch:06d}.png')
                    )
                    plt.close(comparison_fig)

                if hasattr(model, 'get_detailed_matrix_info'):
                    info = model.get_detailed_matrix_info()
                    for i, layer_info in enumerate(info['layer_infos']):
                        if f'stable_rank_layer_{i}' not in layer_metrics:
                            layer_metrics[f'stable_rank_layer_{i}'] = []
                            layer_metrics[f'effective_rank_layer_{i}'] = []
                            layer_metrics[f'spectral_norm_layer_{i}'] = []
                            layer_metrics[f'condition_number_layer_{i}'] = []

                        stable_rank_val = layer_info.get('stable_rank', 0)
                        effective_rank_val = layer_info.get('effective_rank', 0)
                        spectral_norm_val = layer_info.get('linear_spectral_norm', 0)
                        condition_number_val = layer_info.get('spectral_condition_no', 0)

                        layer_metrics[f'stable_rank_layer_{i}'].append(stable_rank_val)
                        layer_metrics[f'effective_rank_layer_{i}'].append(effective_rank_val)
                        layer_metrics[f'spectral_norm_layer_{i}'].append(spectral_norm_val)
                        layer_metrics[f'condition_number_layer_{i}'].append(condition_number_val)

                        log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                        log_dict[f'effective_rank/layer_{i}'] = effective_rank_val
                        log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                        log_dict[f'condition_number/layer_{i}'] = condition_number_val

                    if 'end_to_end_spectral_bound' in info:
                        if 'end_to_end_bound' not in layer_metrics:
                            layer_metrics['end_to_end_bound'] = []
                        end_to_end_val = info['end_to_end_spectral_bound']
                        layer_metrics['end_to_end_bound'].append(end_to_end_val)
                        log_dict['end_to_end_bound'] = end_to_end_val

                log_records.append(log_dict)

    if is_super_resolution:
        original_img = img_hr
    else:
        original_img = img

    return model, metrics, layer_metrics, log_records, original_img, lpips_model


def plot_metrics_seaborn_separate(metrics, layer_metrics, args, output_dir):
    """Plot training metrics using seaborn and save them locally."""
    sns.set_style("darkgrid")
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    is_inpainting_task = args.task == 'fitting' and args.inpainting_ratio < 1.0

    main_df = pd.DataFrame({
        'Epoch': metrics['epochs'],
        'Full PSNR': metrics['full_psnr'],
        'SSIM': metrics['ssim'],
        'LPIPS': metrics['lpips'],
        'Train PSNR': metrics['train_psnr']
    })
    if is_inpainting_task:
        main_df['Test PSNR'] = metrics['test_psnr']

    plt.figure(figsize=(10, 6))
    if is_inpainting_task:
        psnr_df = main_df.melt(id_vars=['Epoch'], value_vars=['Train PSNR', 'Test PSNR', 'Full PSNR'],
                               var_name='Metric', value_name='PSNR (dB)')
        ax = sns.lineplot(data=psnr_df, x='Epoch', y='PSNR (dB)', hue='Metric')
        ax.set_title('PSNR Over Training (Inpainting)', fontsize=16, fontweight='bold', pad=20)
    else:
        ax = sns.lineplot(data=main_df, x='Epoch', y='Full PSNR')
        if args.task == 'super_resolution':
            ax.set_title('PSNR Over Training (Super-resolution)', fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title('PSNR Over Training (Overfitting)', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'psnr_comparison.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='SSIM')
    ax.set_title('Full Image SSIM Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ssim.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=main_df, x='Epoch', y='LPIPS')
    ax.set_title('Full Image LPIPS Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lpips.png'), bbox_inches='tight')
    plt.close()

    if not layer_metrics:
        print("No layer metrics found to plot.")
        sns.reset_defaults()
        return

    layer_df = pd.DataFrame({'Epoch': metrics['epochs']})
    for key, values in layer_metrics.items():
        layer_df[key] = values

    stable_rank_keys = [k for k in layer_df.columns if 'stable_rank_layer_' in k]
    if stable_rank_keys:
        plt.figure(figsize=(12, 7))
        stable_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=stable_rank_keys, var_name='Layer', value_name='Stable Rank')
        stable_rank_df_melted['Layer'] = stable_rank_df_melted['Layer'].str.replace('stable_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=stable_rank_df_melted, x='Epoch', y='Stable Rank', hue='Layer', linewidth=2)
        ax.set_title('Stable Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'stable_ranks.png'), bbox_inches='tight')
        plt.close()

    effective_rank_keys = [k for k in layer_df.columns if 'effective_rank_layer_' in k]
    if effective_rank_keys:
        plt.figure(figsize=(12, 7))
        effective_rank_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=effective_rank_keys, var_name='Layer', value_name='Effective Rank')
        effective_rank_df_melted['Layer'] = effective_rank_df_melted['Layer'].str.replace('effective_rank_layer_', 'Layer ')
        ax = sns.lineplot(data=effective_rank_df_melted, x='Epoch', y='Effective Rank', hue='Layer', linewidth=2)
        ax.set_title('Effective Rank Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'effective_ranks.png'), bbox_inches='tight')
        plt.close()

    spectral_norm_keys = [k for k in layer_df.columns if 'spectral_norm_layer_' in k]
    if spectral_norm_keys:
        plt.figure(figsize=(12, 7))
        spectral_norm_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=spectral_norm_keys, var_name='Layer', value_name='Spectral Norm')
        spectral_norm_df_melted['Layer'] = spectral_norm_df_melted['Layer'].str.replace('spectral_norm_layer_', 'Layer ')
        ax = sns.lineplot(data=spectral_norm_df_melted, x='Epoch', y='Spectral Norm', hue='Layer', linewidth=2)
        ax.set_title('Spectral Norm Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'spectral_norms.png'), bbox_inches='tight')
        plt.close()

    condition_keys = [k for k in layer_df.columns if 'condition_number_layer_' in k]
    if condition_keys:
        plt.figure(figsize=(12, 7))
        condition_df_melted = layer_df.melt(id_vars=['Epoch'], value_vars=condition_keys, var_name='Layer', value_name='Condition Number')
        condition_df_melted['Layer'] = condition_df_melted['Layer'].str.replace('condition_number_layer_', 'Layer ')
        ax = sns.lineplot(data=condition_df_melted, x='Epoch', y='Condition Number', hue='Layer', linewidth=2)
        ax.set_title('Condition Number Evolution', fontsize=16, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'condition_numbers.png'), bbox_inches='tight')
        plt.close()

    sns.reset_defaults()
    print(f"All plots saved under '{plots_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Train a Neural Field for SISR or Image Overfitting/Inpainting.")

    # General arguments
    parser.add_argument('--image', default='data/kodim01.png', type=str, help='Path to the input image.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of training epochs.')
    parser.add_argument('--log_n_epochs', default=500, type=int, help='Frequency of logging metrics and images.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--log_image_evolution', action='store_true', help='Save intermediate image reconstructions to disk.')

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for Adam, Muon (aux), and L-BFGS (initial step size).')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0, help='Weight decay for pure Adam optimizer.')

    # Muon specific
    parser.add_argument('--muon_weight_decay', type=float, default=0.0, help='Weight decay for Muon optimizer (hidden weights).')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_lr', type=float, default=1e-1, help='Learning rate for the Muon part (hidden weights).')
    parser.add_argument('--muon_momentum', type=float, default=0.95, help='Momentum beta for auto_cos_inc Muon updates.')
    parser.add_argument('--muon_ns_steps', type=int, default=5, help='Newton-Schulz steps for auto_cos_inc Muon updates.')
    parser.add_argument('--muon_no_nesterov', action='store_true', help='Disable Nesterov momentum for auto_cos_inc Muon updates.')
    parser.add_argument('--optimize_first_layer_with_muon', action='store_true', help='For models with embeddings (FFN, PosEnc), also optimize the first linear layer of the MLP with Muon.')

    # auto_cos_inc_rank.py specific rank-growth arguments
    parser.add_argument('--rank', type=int, default=200, help='Floor rank used by auto_cos_inc_rank.py.')
    parser.add_argument('--rank_start', type=int, default=None, help='Initial scheduled rank. Default: --rank.')
    parser.add_argument('--rank_end', type=int, default=250, help='Final scheduled rank for cosine rank growth.')
    parser.add_argument('--rank_warmup_steps', type=int, default=None, help='Steps over which rank grows. Default: rank_wsd decay start if scheduler=rank_wsd, otherwise epochs.')
    parser.add_argument('--rank_oversample', type=int, default=4, help='Randomized low-rank oversampling dimension.')
    parser.add_argument('--lowrank_rescale', action='store_true', help='Enable low-rank update rescaling in auto_cos_inc_rank.py.')
    parser.add_argument('--lowrank_eps', type=float, default=1e-6, help='Numerical epsilon for low-rank Muon update.')
    parser.add_argument('--small_ns_bfloat16', action='store_true', help='Use bfloat16 for small Newton-Schulz path on CUDA.')
    parser.add_argument('--auto_init_rank_start', action='store_true', help='Estimate rank_start from early sketched Frobenius-energy probes.')
    parser.add_argument('--init_probe_steps', type=int, default=8, help='Number of early steps for auto rank_start estimation.')
    parser.add_argument('--init_energy', type=float, default=0.90, help='Energy threshold for auto rank_start estimation.')
    parser.add_argument('--init_round_multiple', type=int, default=8, help='Round auto-estimated rank_start up to this multiple.')
    parser.add_argument('--muon_delta', type=float, default=None,
                    help='If set, project (clip) Muon update ΔW to operator-norm ball of radius delta.')
    parser.add_argument('--muon_op_clip_mode', type=str, default='matrix',
                        choices=['matrix', 'conv'],
                        help='Operator-norm estimation mode: matrix (2D reshape) or conv (conv-operator power-iter).')
    parser.add_argument('--muon_op_clip_iters', type=int, default=2,
                        help='Power-iteration steps for operator-norm estimation.')

    # L-BFGS specific
    parser.add_argument('--lbfgs_max_iter', type=int, default=20, help='Max iterations per epoch for L-BFGS.')
    parser.add_argument('--lbfgs_history_size', type=int, default=100, help='History size for L-BFGS.')

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step', 'rank_wsd', 'rank-wsd'], help='Learning rate scheduler type (not used with L-BFGS).')
    parser.add_argument('--T_max', type=int, default=5000, help='T_max for CosineAnnealingLR scheduler.')
    parser.add_argument('--step_size', type=int, default=500, help='Step size for StepLR scheduler.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for StepLR scheduler.')

    # Rank-WSD scheduler arguments
    parser.add_argument('--rank_wsd_warmup_steps', type=int, default=0, help='Warmup steps for rank_wsd scheduler.')
    parser.add_argument('--rank_wsd_decay_start_step', type=int, default=None, help='Step where rank_wsd starts late linear decay. Default: 0.8 * epochs.')
    parser.add_argument('--rank_wsd_min_lr_ratio', type=float, default=0.1, help='Final LR ratio for rank_wsd scheduler.')
    parser.add_argument('--rank_wsd_base_N_rand', type=int, default=None, help='Base N_rand for rank_wsd. If omitted, infer from optimizer when possible; otherwise use 1.')

    # Model and architecture arguments
    parser.add_argument('--model', choices=['relu_mlp', 'relu_ffn', 'relu_hash', 'gauss_ffn', 'gauss_mlp', 'siren_mlp', 'wire_mlp', 'real_wire', 'relu_pos_enc', 'replicate_mlp', 'finer_mlp', 'fourier_net'], default='relu_mlp')
    parser.add_argument('--num_layers', default=5, type=int, help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', default=300, type=int, help='Dimension of hidden layers.')
    parser.add_argument('--mapping_size', default=128, type=int, help='Mapping size for Fourier Feature mappings.')

    # Model-specific activation arguments
    parser.add_argument('--fourier_sigma', default=10.0, type=float, help='Sigma for Fourier Feature mapping.')
    parser.add_argument('--siren_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for SIREN layers.")
    parser.add_argument('--finer_omega', default='40.0,40.0,40.0,40.0', type=str, help="Omega for FINER layers.")
    parser.add_argument('--finer_init_bias', action="store_true",  help="Initial bias for FINER layers.")
    parser.add_argument('--finer_bias_scale', default=float(1/math.sqrt(2)), type=float, help="Bias scale for FINER layers.")
    parser.add_argument('--gauss_scale', default='0.0236', type=str, help='Scale parameter for Gaussian activation.')
    parser.add_argument('--wire_sigma', default='10.0', type=str, help="Sigma for WIRE layers.")
    parser.add_argument('--wire_omega', default='20.0', type=str, help="Omega for WIRE layers.")
    parser.add_argument('--mfn_fourier_scale', default=256, type=float, help='Scale for MFN_FourierNet model.')

    # HashGrid specific arguments (kept for argument compatibility with overfitting code)
    parser.add_argument('--hash_n_levels', default=16, type=int, help='Number of levels in HashGrid.')
    parser.add_argument('--hash_n_features_per_level', default=2, type=int, help='Number of features per level in HashGrid.')
    parser.add_argument('--hash_log2_hashmap_size', default=15, type=int, help='Log2 of hashmap size for HashGrid.')
    parser.add_argument('--hash_base_resolution', default=16, type=int, help='Base resolution for HashGrid.')
    parser.add_argument('--hash_finest_resolution', default=512, type=int, help='Finest resolution for HashGrid.')

    # Task arguments
    parser.add_argument('--task', default='super_resolution', type=str, choices=['super_resolution', 'fitting'], help='Task to perform.')
    parser.add_argument('--scale_factor', default=4, type=int, help='SISR upscaling factor.')
    parser.add_argument('--inpainting_ratio', default=1.0, type=float, help='Ratio of pixels for training (1.0 for overfitting, <1.0 for inpainting).')
    parser.add_argument('--sr_train_chunk_size', default=65536, type=int, help='Chunk size for LR training forward/backward in SISR. Use <=0 to disable.')
    parser.add_argument('--sr_eval_chunk_size', default=262144, type=int, help='Chunk size for HR evaluation in SISR. Use <=0 to disable.')

    # output folder naming
    parser.add_argument('--folder_name', default=None, type=str)

    args = parser.parse_args()
    args.optimizer = normalize_optimizer_name(args.optimizer)
    if args.scheduler == 'rank-wsd':
        args.scheduler = 'rank_wsd'
    resolve_auto_cos_inc_rank_args(args)
    set_seed(args.seed)

    # Parse model-specific parameters that can be lists
    args.gauss_scale = parse_list(args.gauss_scale, args.num_layers)
    args.siren_omega = parse_list(args.siren_omega, args.num_layers)
    args.finer_omega = parse_list(args.finer_omega, args.num_layers)
    args.wire_sigma = parse_list(args.wire_sigma, args.num_layers)
    args.wire_omega = parse_list(args.wire_omega, args.num_layers)

    if not 0.0 < args.inpainting_ratio <= 1.0:
        raise ValueError("inpainting_ratio must be between 0.0 and 1.0.")
    if args.scale_factor < 1:
        raise ValueError('scale_factor must be >= 1.')

    output_dir = get_output_dir(args)
    save_args(args, output_dir)
    print(f"Results will be saved to: {output_dir}")

    model, metrics, layer_metrics, log_records, original_img, lpips_model = train_model(args, output_dir)

    # Final evaluation and logging
    model.eval()
    with torch.no_grad():
        h, w, c = original_img.shape
        coords = get_coordinates(h, w).to(next(model.parameters()).device)

        if args.task == 'super_resolution':
            final_pred = forward_in_chunks(model, coords, args.sr_eval_chunk_size).view(h, w, c).cpu()
        else:
            final_pred = model(coords).view(h, w, c).cpu()

        final_psnr = psnr(final_pred, original_img).item()
        final_ssim = ssim(original_img.numpy(), final_pred.numpy(), channel_axis=-1, data_range=1.0)
        final_lpips = compute_lpips(original_img.numpy(), final_pred.numpy(),
                                  lpips_model, next(model.parameters()).device)

        print("\n--- Final Results ---")
        print(f"Final Full Image PSNR: {final_psnr:.2f}")
        print(f"Final Full Image SSIM: {final_ssim:.4f}")
        print(f"Final Full Image LPIPS: {final_lpips:.4f}")

        final_comparison_fig = create_comparison_image(
            original_img.numpy(), final_pred.numpy(), args.epochs, final_psnr, final_ssim, final_lpips)

        final_comparison_fig.savefig(os.path.join(output_dir, 'final_comparison.png'), bbox_inches='tight')
        plt.close(final_comparison_fig)
        save_image(np.clip(final_pred.numpy(), 0, 1), os.path.join(output_dir, 'final_reconstruction.png'))
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
        save_readme(output_dir, final_psnr, final_ssim, final_lpips)

    save_metrics_csv(metrics, layer_metrics, log_records, output_dir)
    plot_metrics_seaborn_separate(metrics, layer_metrics, args, output_dir)
    print(f"Training finished and results saved to {output_dir}.")


if __name__ == "__main__":
    main()

"""
python sisr_auto_cos_inc_rank_wsd.py \
  --task super_resolution \
  --optimizer auto_cos_inc \
  --scheduler rank_wsd \
  --epochs 5000 \
  --rank 64 \
  --rank_start 64 \
  --rank_end 256 \
  --rank_warmup_steps 4000 \
  --rank_wsd_decay_start_step 4000 \
  --rank_wsd_min_lr_ratio 0.1 \
  --muon_lr 1e-1 \
  --lr 1e-3
"""