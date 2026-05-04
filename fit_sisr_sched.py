import argparse
import importlib
import inspect
import math
import os
import random
from typing import Dict, List, Optional, Tuple

try:
    import lpips
except ImportError:
    lpips = None
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

try:
    from optims.rank_wsd_schedulers import RankAwareWarmupStableLinearScheduler
except ImportError:
    from rank_wsd_schedulers import RankAwareWarmupStableLinearScheduler

try:
    from optims.muon import SingleDeviceMuonWithAuxAdam
except ImportError:
    from muon import SingleDeviceMuonWithAuxAdam

try:
    from optims.lr_inc import SingleDeviceFinalIncWithAuxAdam
except ImportError:
    from lr_inc import SingleDeviceFinalIncWithAuxAdam

try:
    from optims.lr_sign import SingleDeviceSignWithAuxAdam
except ImportError:
    from lr_sign import SingleDeviceSignWithAuxAdam

try:
    from optims.lr_svd import SingleDeviceLrSVDWithAuxAdam
except ImportError:
    from lr_svd import SingleDeviceLrSVDWithAuxAdam

try:
    from optims.auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam
except ImportError:
    from auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam

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


AUTO_COS_INC_CANONICAL_NAME = "auto_cos_inc"
AUTO_COS_INC_OPTIMIZER_ALIASES = {
    "auto_cos_inc",
    "auto-cos-inc",
    "auto_cos_inc_rank",
    "auto-cos-inc-rank",
}

LRSIGN10_CANONICAL_NAME = "lr_sign10_rsclF"
LRSIGN10_OPTIMIZER_ALIASES = {
    "lr_sign10_rsclF",
    "lr-sign10-rsclF",
    "lr_sign10_rsclf",
    "lr-sign10-rsclf",
}

LOWRANK_OPTIMIZER_NAMES = {
    "lr-inc",
    "lr-sign",
    "lr-svd",
    LRSIGN10_CANONICAL_NAME,
}
LOWRANK_LIKE_OPTIMIZERS = {"muon"} | LOWRANK_OPTIMIZER_NAMES | {AUTO_COS_INC_CANONICAL_NAME}


# =========================================================
# Basic utilities
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(img, dtype=torch.float32)


def parse_list(param_str, num_layers: int) -> List[float]:
    if isinstance(param_str, (int, float)):
        return [float(param_str)] * (num_layers - 1)

    param_str = str(param_str)
    if "," in param_str:
        values = [float(x.strip()) for x in param_str.split(",")]
        if len(values) != num_layers - 1:
            raise ValueError(
                f"Expected {num_layers - 1} comma-separated values, but got {len(values)}: {param_str}"
            )
        return values

    return [float(param_str)] * (num_layers - 1)


def get_coordinates(h: int, w: int) -> torch.Tensor:
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return mse.new_tensor(float("inf"))
    return 20 * torch.log10(mse.new_tensor(max_val) / torch.sqrt(mse))


def compute_lpips(img1, img2, lpips_model, device: torch.device) -> float:
    if lpips_model is None:
        return float("nan")

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    img1 = (img1.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)
    img2 = (img2.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).to(device)

    with torch.no_grad():
        value = lpips_model(img1, img2)
    return float(value.item())


def get_output_dir(args) -> str:
    base_dir = args.folder_name if args.folder_name is not None else "results"
    output_dir = os.path.join(base_dir, args.model, args.optimizer)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_args(args, output_dir: str) -> None:
    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def save_readme(output_dir: str, final_psnr: float, final_ssim: float, final_lpips: float) -> None:
    with open(os.path.join(output_dir, "Readme.txt"), "w", encoding="utf-8") as f:
        f.write("Final Evaluation Metrics\n")
        f.write("========================\n")
        f.write(f"PSNR: {final_psnr:.6f}\n")
        f.write(f"SSIM: {final_ssim:.6f}\n")
        f.write(f"LPIPS: {final_lpips:.6f}\n")


def save_metrics_csv(metrics: Dict, layer_metrics: Dict, log_records: List[Dict], output_dir: str) -> None:
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    if log_records:
        pd.DataFrame(log_records).to_csv(os.path.join(output_dir, "train_log.csv"), index=False)

    if layer_metrics:
        layer_df = pd.DataFrame({"Epoch": metrics["epochs"]})
        for key, values in layer_metrics.items():
            layer_df[key] = values
        layer_df.to_csv(os.path.join(output_dir, "layer_metrics.csv"), index=False)


def save_image(image_array: np.ndarray, save_path: str) -> None:
    image_array = np.clip(image_array, 0, 1)
    if image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = image_array.squeeze(-1)
    image_uint8 = (image_array * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(save_path)


def modcrop(img: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return img

    h, w, _ = img.shape
    h_mod = h - (h % scale)
    w_mod = w - (w % scale)

    if h_mod != h or w_mod != w:
        print(f"Applying modcrop: ({h}, {w}) -> ({h_mod}, {w_mod}) for scale x{scale}.")
        img = img[:h_mod, :w_mod, :]

    return img


def forward_in_chunks(model, coords: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size is None or chunk_size <= 0 or coords.shape[0] <= chunk_size:
        return model(coords)

    outputs = []
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        outputs.append(model(coords[start:end]))
    return torch.cat(outputs, dim=0)


# =========================================================
# Model / optimizer setup
# =========================================================
def normalize_optimizer_name(name: str) -> str:
    if name in AUTO_COS_INC_OPTIMIZER_ALIASES:
        return AUTO_COS_INC_CANONICAL_NAME
    if name in LRSIGN10_OPTIMIZER_ALIASES:
        return LRSIGN10_CANONICAL_NAME
    if name == "lr_sign":
        return "lr-sign"
    if name == "lr_inc":
        return "lr-inc"
    if name == "lr_svd":
        return "lr-svd"
    return name


def resolve_auto_cos_inc_rank_args(args) -> None:
    if args.optimizer != AUTO_COS_INC_CANONICAL_NAME:
        return

    if args.rank <= 0:
        raise ValueError("--rank must be > 0.")

    if args.rank_start is None:
        args.rank_start = int(args.rank)
    if args.rank_end is None:
        args.rank_end = int(args.rank_start)

    args.rank_start = int(args.rank_start)
    args.rank_end = int(args.rank_end)

    if args.rank_start <= 0:
        raise ValueError("--rank_start must be > 0.")
    if args.rank_end <= 0:
        raise ValueError("--rank_end must be > 0.")
    if args.rank_end < args.rank_start:
        raise ValueError("--rank_end must be >= --rank_start.")

    if args.rank_warmup_steps is None:
        if args.scheduler == "rank_wsd" and args.rank_wsd_decay_start_step is not None:
            args.rank_warmup_steps = max(1, int(args.rank_wsd_decay_start_step))
        elif args.scheduler == "rank_wsd":
            args.rank_warmup_steps = max(1, int(round(0.8 * args.epochs)))
        else:
            args.rank_warmup_steps = max(1, int(args.epochs))
    else:
        args.rank_warmup_steps = int(args.rank_warmup_steps)

    if args.rank_warmup_steps <= 0:
        raise ValueError("--rank_warmup_steps must be > 0.")
    if args.rank_oversample < 0:
        raise ValueError("--rank_oversample must be >= 0.")
    if args.muon_ns_steps <= 0:
        raise ValueError("--muon_ns_steps must be > 0.")
    if not (0.0 < args.init_energy <= 1.0):
        raise ValueError("--init_energy must be in (0, 1].")
    if args.init_probe_steps <= 0:
        raise ValueError("--init_probe_steps must be > 0.")
    if args.init_round_multiple <= 0:
        raise ValueError("--init_round_multiple must be > 0.")

    if args.rank_start < args.rank:
        print(
            "WARNING: --rank_start is smaller than --rank. "
            "auto_cos_inc_rank floors the applied rank by --rank."
        )


def build_model(args, c: int):
    if args.model == "relu_ffn":
        return ReluFFN(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            sigma=args.fourier_sigma,
        )
    if args.model == "relu_mlp":
        return ReluMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
        )
    if args.model == "relu_pos_enc":
        return ReluPosEncoding(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
        )
    if args.model == "gauss_ffn":
        return GaussFFN(
            input_dim=2,
            mapping_size=args.mapping_size,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            sigma=args.fourier_sigma,
            a=args.gauss_scale,
        )
    if args.model == "gauss_mlp":
        return GaussMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            a=args.gauss_scale,
        )
    if args.model == "siren_mlp":
        return SirenMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.siren_omega,
        )
    if args.model == "wire_mlp":
        return WireMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.wire_omega,
            sigma=args.wire_sigma,
        )
    if args.model == "real_wire":
        return WireRealMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.wire_omega,
            sigma=args.wire_sigma,
        )
    if args.model == "finer_mlp":
        return FinerMLP(
            input_dim=2,
            hidden_dim=args.hidden_dim,
            output_dim=c,
            num_layers=args.num_layers,
            omega=args.finer_omega,
            init_bias=args.finer_init_bias,
            bias_scale=args.finer_bias_scale,
        )

    raise ValueError(f"Unsupported model type: {args.model}")


def split_params_for_muon_like(args, model) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    muon_params = []
    other_params = []

    first_layer_muon_models = {"relu_ffn", "gauss_ffn", "relu_pos_enc"}
    is_special_model = args.model in first_layer_muon_models

    if not (hasattr(model, "mlp") and isinstance(model.mlp, torch.nn.Sequential)):
        raise ValueError(
            f"Model {args.model} does not have a standard torch.nn.Sequential 'mlp' attribute. "
            "Cannot separate hidden matrix parameters for Muon/low-rank optimizers."
        )

    num_mlp_layers = len(model.mlp)

    for name, param in model.named_parameters():
        is_muon_target = False

        if "mlp" in name and "weight" in name and param.ndim >= 2:
            try:
                layer_idx = int(name.split(".")[1])
                is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                is_first_layer_for_muon = (
                    is_special_model
                    and args.optimize_first_layer_with_muon
                    and layer_idx == 0
                )
                is_muon_target = is_hidden_layer or is_first_layer_for_muon
            except (ValueError, IndexError):
                is_muon_target = False

        if is_muon_target:
            muon_params.append(param)
        else:
            other_params.append(param)

    if not muon_params:
        raise ValueError(
            f"{args.optimizer} was selected for model={args.model}, but no suitable hidden weight "
            "parameters were identified."
        )

    return muon_params, other_params


def load_lr_sign10_optimizer_class():
    module = None
    import_errors = []

    for module_name in ("optims.lr_sign10_rsclF", "lr_sign10_rsclF"):
        try:
            module = importlib.import_module(module_name)
            break
        except ImportError as exc:
            import_errors.append(str(exc))

    if module is None:
        raise ImportError(
            "Could not import lr_sign10_rsclF optimizer. Tried optims.lr_sign10_rsclF and lr_sign10_rsclF. "
            f"Errors: {import_errors}"
        )

    candidate_names = (
        "SingleDeviceSign10RsclFWithAuxAdam",
        "SingleDeviceSign10RscLFWithAuxAdam",
        "SingleDeviceLrSign10RsclFWithAuxAdam",
        "SingleDeviceLRSign10RsclFWithAuxAdam",
        "SingleDeviceSignWithAuxAdam",
        "SingleDeviceLrSignWithAuxAdam",
    )

    for class_name in candidate_names:
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls

    optimizer_classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and issubclass(obj, torch.optim.Optimizer):
            optimizer_classes.append(obj)

    if len(optimizer_classes) == 1:
        return optimizer_classes[0]

    class_list = ", ".join(cls.__name__ for cls in optimizer_classes) or "none"
    raise ImportError(
        "Could not infer optimizer class from lr_sign10_rsclF.py. "
        f"Detected Optimizer subclasses: {class_list}. "
        "Please rename the class to SingleDeviceSign10RsclFWithAuxAdam or add it to candidate_names."
    )


def build_plain_muon_param_groups(args, muon_params, other_params):
    """
    optims/muon.py in the Muon-INR/KellerJordan style asserts that every
    param group has exactly these keys:
        use_muon=True:  params, lr, momentum, weight_decay, use_muon
        use_muon=False: params, lr, betas, eps, weight_decay, use_muon

    Do not pass ns_steps or nesterov to this optimizer unless your local
    optims/muon.py has been modified to accept them.
    """
    if args.muon_ns_steps != 5:
        print(
            "WARNING: pure --optimizer muon uses the ns_steps hard-coded in optims/muon.py. "
            "The --muon_ns_steps value is only forwarded to auto_cos_inc_rank unless you modify optims/muon.py."
        )

    return [
        dict(
            params=muon_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
            use_muon=True,
        ),
        dict(
            params=other_params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.muon_aux_weight_decay,
            use_muon=False,
        ),
    ]


def build_lowrank_param_groups(args, muon_params, other_params):
    """
    Conservative default for lr-sign / lr-sign10-rsclF / lr-inc / lr-svd.
    These files are usually Muon-style optimizers with an auxiliary Adam path,
    so the same minimal key set avoids strict key-assertion failures.
    """
    return build_plain_muon_param_groups(args, muon_params, other_params)


def build_auto_cos_inc_param_groups(args, muon_params, other_params):
    """Param groups for optims/auto_cos_inc_rank.py, which needs rank-growth args."""
    return [
        dict(
            params=muon_params,
            use_muon=True,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            ns_steps=args.muon_ns_steps,
            nesterov=not args.muon_no_nesterov,
            weight_decay=args.muon_weight_decay,
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
        ),
        dict(
            params=other_params,
            use_muon=False,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.muon_aux_weight_decay,
        ),
    ]


def build_optimizer(args, model):
    if args.optimizer == "adam":
        print("INFO: Using Adam optimizer.")
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
        )

    if args.optimizer == "lbfgs":
        print("INFO: Using L-BFGS optimizer.")
        return optim.LBFGS(
            model.parameters(),
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
        )

    if args.optimizer not in LOWRANK_LIKE_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    muon_params, other_params = split_params_for_muon_like(args, model)

    if args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
        param_groups = build_auto_cos_inc_param_groups(args, muon_params, other_params)
        optimizer = SingleDeviceAutoCosIncWithAuxAdam(param_groups)
        optimizer_name = AUTO_COS_INC_CANONICAL_NAME
    elif args.optimizer == "muon":
        param_groups = build_plain_muon_param_groups(args, muon_params, other_params)
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        optimizer_name = "muon"
    elif args.optimizer == "lr-inc":
        param_groups = build_lowrank_param_groups(args, muon_params, other_params)
        optimizer = SingleDeviceFinalIncWithAuxAdam(param_groups)
        optimizer_name = "lr-inc"
    elif args.optimizer == "lr-sign":
        param_groups = build_lowrank_param_groups(args, muon_params, other_params)
        optimizer = SingleDeviceSignWithAuxAdam(param_groups)
        optimizer_name = "lr-sign"
    elif args.optimizer == "lr-svd":
        param_groups = build_lowrank_param_groups(args, muon_params, other_params)
        optimizer = SingleDeviceLrSVDWithAuxAdam(param_groups)
        optimizer_name = "lr-svd"
    elif args.optimizer == LRSIGN10_CANONICAL_NAME:
        param_groups = build_lowrank_param_groups(args, muon_params, other_params)
        LrSign10Optimizer = load_lr_sign10_optimizer_class()
        optimizer = LrSign10Optimizer(param_groups)
        optimizer_name = LRSIGN10_CANONICAL_NAME
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")

    print(
        f"INFO: {optimizer_name} optimizer configured. "
        f"Muon/low-rank params: {len(muon_params)}, Aux params: {len(other_params)}."
    )
    return optimizer


# =========================================================
# Scheduler setup
# =========================================================
def is_rank_wsd_scheduler(scheduler) -> bool:
    return isinstance(scheduler, RankAwareWarmupStableLinearScheduler)


def _infer_rank_wsd_base_lrs(args, optimizer) -> Tuple[float, float]:
    base_lr_adam = None
    base_lr_muon = None

    for group in optimizer.param_groups:
        group_lr = group.get("lr")
        if group_lr is None:
            continue
        if group.get("use_muon", False):
            if base_lr_muon is None:
                base_lr_muon = float(group_lr)
        else:
            if base_lr_adam is None:
                base_lr_adam = float(group_lr)

    if base_lr_adam is None:
        base_lr_adam = float(args.lr)
    if base_lr_muon is None:
        base_lr_muon = float(getattr(args, "muon_lr", base_lr_adam))

    return base_lr_adam, base_lr_muon


def _infer_optimizer_N_rand(optimizer) -> Optional[int]:
    for attr_name in ("N_rand", "n_rand"):
        if hasattr(optimizer, attr_name):
            try:
                value = int(getattr(optimizer, attr_name))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

    for group in optimizer.param_groups:
        for key in ("N_rand", "n_rand"):
            if key in group:
                try:
                    value = int(group[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    defaults = getattr(optimizer, "defaults", {})
    if isinstance(defaults, dict):
        for key in ("N_rand", "n_rand"):
            if key in defaults:
                try:
                    value = int(defaults[key])
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass

    return None


def _resolve_rank_wsd_base_N_rand(args, optimizer) -> int:
    if args.rank_wsd_base_N_rand is not None:
        if args.rank_wsd_base_N_rand <= 0:
            raise ValueError("--rank_wsd_base_N_rand must be > 0 when provided.")
        return int(args.rank_wsd_base_N_rand)

    inferred = _infer_optimizer_N_rand(optimizer)
    return inferred if inferred is not None else 1


def _safe_setattr(obj, attr_name: str, value) -> bool:
    try:
        setattr(obj, attr_name, value)
        return True
    except (AttributeError, TypeError):
        return False


def _apply_N_rand_to_optimizer(optimizer, N_rand: int) -> None:
    N_rand = int(N_rand)

    for setter_name in ("set_N_rand", "set_n_rand"):
        setter = getattr(optimizer, setter_name, None)
        if callable(setter):
            try:
                setter(N_rand)
            except TypeError:
                pass

    for attr_name in ("N_rand", "n_rand"):
        if hasattr(optimizer, attr_name):
            _safe_setattr(optimizer, attr_name, N_rand)

    for group in optimizer.param_groups:
        for key in ("N_rand", "n_rand"):
            if key in group:
                group[key] = N_rand

    defaults = getattr(optimizer, "defaults", None)
    if isinstance(defaults, dict):
        for key in ("N_rand", "n_rand"):
            if key in defaults:
                defaults[key] = N_rand


def apply_rank_wsd_scheduler_step(scheduler, optimizer, global_step: int) -> Dict:
    state = scheduler.step(global_step)

    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            group["lr"] = state["lr_muon"]
        else:
            group["lr"] = state["lr_adam"]

    _apply_N_rand_to_optimizer(optimizer, state["N_rand"])
    return state


def build_scheduler(args, optimizer):
    if args.optimizer == "lbfgs":
        print("INFO: Schedulers are disabled for L-BFGS optimizer.")
        return None

    if args.scheduler == "none":
        print("INFO: Scheduler is disabled.")
        return None

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.cosine_eta_min)
        print(f"Using CosineAnnealingLR scheduler with T_max={args.T_max}, eta_min={args.cosine_eta_min}")
        return scheduler

    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(f"Using StepLR scheduler with step_size={args.step_size}, gamma={args.gamma}")
        return scheduler

    if args.scheduler == "rank_wsd":
        if args.optimizer not in {"lr-sign", AUTO_COS_INC_CANONICAL_NAME}:
            print(
                "WARNING: rank_wsd was requested for optimizer "
                f"{args.optimizer}. The bash grid is intended to use rank_wsd only with "
                "lr-sign and auto_cos_inc_rank. Continuing anyway."
            )

        if args.rank_wsd_decay_start_step is None:
            args.rank_wsd_decay_start_step = int(round(0.8 * args.epochs))

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
        return scheduler

    raise ValueError(f"Unsupported scheduler type: {args.scheduler}")


def get_auto_cos_inc_rank_state(optimizer) -> Dict:
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            return {
                "current_rank": group.get("current_rank"),
                "current_target_rank": group.get("current_target_rank"),
                "rank_floor": group.get("rank"),
                "rank_start": group.get("rank_start"),
                "rank_end": group.get("rank_end"),
                "rank_warmup_steps": group.get("warmup_steps"),
                "auto_rank_start_final": group.get("auto_rank_start_final"),
                "current_method": group.get("current_method"),
            }
    return {}


# =========================================================
# Data and training
# =========================================================
def make_super_resolution_data(img_hr: torch.Tensor, scale_factor: int, device: torch.device) -> Dict:
    img_hr = modcrop(img_hr, scale_factor)
    h_hr, w_hr, c = img_hr.shape

    img_hr_bchw = img_hr.permute(2, 0, 1).unsqueeze(0).to(device)
    h_lr, w_lr = h_hr // scale_factor, w_hr // scale_factor

    img_lr_bchw = F.interpolate(
        img_hr_bchw,
        size=(h_lr, w_lr),
        mode="bicubic",
        antialias=True,
    ).detach()

    img_lr = img_lr_bchw.squeeze(0).permute(1, 2, 0).cpu()
    coords_lr = get_coordinates(h_lr, w_lr).to(device)
    target_lr = img_lr.reshape(-1, c).to(device)
    coords_hr = get_coordinates(h_hr, w_hr).to(device)

    return {
        "img_hr": img_hr,
        "img_lr": img_lr,
        "coords_hr": coords_hr,
        "coords_lr": coords_lr,
        "target_lr": target_lr,
        "h_hr": h_hr,
        "w_hr": w_hr,
        "h_lr": h_lr,
        "w_lr": w_lr,
        "c": c,
    }


def compute_chunked_train_loss(model, coords: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    total_loss = None
    total_numel = target.numel()

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        pred_chunk = model(coords[start:end])
        target_chunk = target[start:end]
        chunk_loss = F.mse_loss(pred_chunk, target_chunk, reduction="sum")
        scaled_loss = chunk_loss / total_numel
        scaled_loss.backward()
        total_loss = chunk_loss.detach() if total_loss is None else total_loss + chunk_loss.detach()

    return total_loss / total_numel


def train_model(args, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lpips_model = None
    if args.skip_lpips:
        print("INFO: --skip_lpips enabled. LPIPS model will not be loaded; LPIPS is written as nan.")
    else:
        if lpips is None:
            raise ImportError("lpips package is required unless --skip_lpips is used.")
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    img = load_image(args.image)
    is_super_resolution = args.task == "super_resolution"
    is_inpainting_task = (args.task == "fitting") and (args.inpainting_ratio < 1.0)

    if is_super_resolution:
        print(f"Performing {args.scale_factor}x super-resolution task.")
        sr_data = make_super_resolution_data(img, args.scale_factor, device)
        img_hr = sr_data["img_hr"]
        h, w, c = sr_data["h_hr"], sr_data["w_hr"], sr_data["c"]
        coords = sr_data["coords_hr"]
        train_coords = sr_data["coords_lr"]
        train_target = sr_data["target_lr"]
        gt_img_np = img_hr.numpy()
    else:
        h, w, c = img.shape
        coords_cpu = get_coordinates(h, w)
        target = img.reshape(-1, c)
        gt_img_np = img.numpy()

        if is_inpainting_task:
            print(f"Performing inpainting task. Training on {args.inpainting_ratio * 100:.2f}% of pixels.")
            num_pixels = h * w
            num_train_pixels = int(num_pixels * args.inpainting_ratio)
            indices = torch.randperm(num_pixels)
            train_indices = indices[:num_train_pixels]
            test_indices = indices[num_train_pixels:]
            train_coords = coords_cpu[train_indices].to(device)
            train_target = target[train_indices].to(device)
            test_coords = coords_cpu[test_indices].to(device)
            test_target = target[test_indices].to(device)
        else:
            print("Performing overfitting task on all pixels.")
            train_coords = coords_cpu.to(device)
            train_target = target.to(device)

        coords = coords_cpu.to(device)

    model = build_model(args, c)
    print(f"Using model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters.")
    model = model.to(device)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    metrics = {
        "epochs": [],
        "full_psnr": [],
        "ssim": [],
        "lpips": [],
        "train_psnr": [],
        "test_psnr": [],
    }
    layer_metrics = {}
    log_records = []

    for epoch in range(args.epochs):
        model.train()

        rank_wsd_state = None
        if is_rank_wsd_scheduler(scheduler):
            rank_wsd_state = apply_rank_wsd_scheduler_step(scheduler, optimizer, epoch)

        if args.optimizer == "lbfgs":
            def closure():
                optimizer.zero_grad()
                pred = model(train_coords)
                loss_closure = F.mse_loss(pred, train_target)
                loss_closure.backward()
                return loss_closure

            optimizer.step(closure)
            with torch.no_grad():
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
        else:
            optimizer.zero_grad()
            if (
                is_super_resolution
                and args.sr_train_chunk_size > 0
                and train_coords.shape[0] > args.sr_train_chunk_size
            ):
                loss = compute_chunked_train_loss(model, train_coords, train_target, args.sr_train_chunk_size)
            else:
                pred = model(train_coords)
                loss = F.mse_loss(pred, train_target)
                loss.backward()
            optimizer.step()

        if scheduler is not None and not is_rank_wsd_scheduler(scheduler):
            scheduler.step()

        should_log = (epoch % args.log_n_epochs == 0) or (epoch == args.epochs - 1)
        if not should_log:
            continue

        model.eval()
        with torch.no_grad():
            if is_super_resolution:
                full_pred = forward_in_chunks(model, coords, args.sr_eval_chunk_size)
            else:
                full_pred = model(coords)

            pred_img = full_pred.view(h, w, c).cpu().numpy()
            target_img = gt_img_np

            full_psnr_val = float(psnr(torch.tensor(pred_img), torch.tensor(target_img)).item())
            ssim_val = float(ssim(target_img, pred_img, channel_axis=-1, data_range=1.0))
            lpips_val = float("nan") if args.skip_lpips else compute_lpips(target_img, pred_img, lpips_model, device)

            if (
                is_super_resolution
                and args.sr_train_chunk_size > 0
                and train_coords.shape[0] > args.sr_train_chunk_size
            ):
                train_pred = forward_in_chunks(model, train_coords, args.sr_eval_chunk_size)
            else:
                train_pred = model(train_coords)
            train_psnr_val = float(psnr(train_pred, train_target).item())

            test_psnr_val = 0.0
            if is_inpainting_task:
                test_psnr_val = float(psnr(model(test_coords), test_target).item())

            metrics["epochs"].append(epoch)
            metrics["full_psnr"].append(full_psnr_val)
            metrics["ssim"].append(ssim_val)
            metrics["lpips"].append(lpips_val)
            metrics["train_psnr"].append(train_psnr_val)
            metrics["test_psnr"].append(test_psnr_val)

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
                "epoch": epoch,
                "loss": float(loss.item()),
                "full_psnr": full_psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "train_psnr": train_psnr_val,
            }

            if args.optimizer in LOWRANK_LIKE_OPTIMIZERS:
                log_dict["learning_rate_muon"] = optimizer.param_groups[0]["lr"]
                log_dict["learning_rate_aux"] = optimizer.param_groups[1]["lr"]
            else:
                log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

            if args.optimizer == AUTO_COS_INC_CANONICAL_NAME:
                for key, value in get_auto_cos_inc_rank_state(optimizer).items():
                    log_dict[f"auto_cos_inc/{key}"] = value

            if rank_wsd_state is not None:
                log_dict["rank_wsd_phase"] = rank_wsd_state["phase"]
                log_dict["rank_wsd_phase_name"] = rank_wsd_state["phase_name"]
                log_dict["rank_wsd_lr_ratio"] = rank_wsd_state["lr_ratio"]
                log_dict["rank_wsd_N_rand"] = rank_wsd_state["N_rand"]

            if is_inpainting_task:
                log_dict["test_psnr"] = test_psnr_val

            if hasattr(model, "get_detailed_matrix_info"):
                info = model.get_detailed_matrix_info()
                for i, layer_info in enumerate(info.get("layer_infos", [])):
                    for metric_name in (
                        "stable_rank_layer",
                        "effective_rank_layer",
                        "spectral_norm_layer",
                        "condition_number_layer",
                    ):
                        layer_metrics.setdefault(f"{metric_name}_{i}", [])

                    stable_rank_val = layer_info.get("stable_rank", 0)
                    effective_rank_val = layer_info.get("effective_rank", 0)
                    spectral_norm_val = layer_info.get("linear_spectral_norm", 0)
                    condition_number_val = layer_info.get("spectral_condition_no", 0)

                    layer_metrics[f"stable_rank_layer_{i}"].append(stable_rank_val)
                    layer_metrics[f"effective_rank_layer_{i}"].append(effective_rank_val)
                    layer_metrics[f"spectral_norm_layer_{i}"].append(spectral_norm_val)
                    layer_metrics[f"condition_number_layer_{i}"].append(condition_number_val)

                    log_dict[f"stable_rank/layer_{i}"] = stable_rank_val
                    log_dict[f"effective_rank/layer_{i}"] = effective_rank_val
                    log_dict[f"spectral_norm/layer_{i}"] = spectral_norm_val
                    log_dict[f"condition_number/layer_{i}"] = condition_number_val

                if "end_to_end_spectral_bound" in info:
                    layer_metrics.setdefault("end_to_end_bound", [])
                    end_to_end_val = info["end_to_end_spectral_bound"]
                    layer_metrics["end_to_end_bound"].append(end_to_end_val)
                    log_dict["end_to_end_bound"] = end_to_end_val

            log_records.append(log_dict)

    original_img = img_hr if is_super_resolution else img
    return model, metrics, layer_metrics, log_records, original_img, lpips_model


# =========================================================
# CLI
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Neural Field for SISR or image fitting/inpainting.")

    # General
    parser.add_argument("--image", default="data/0001.png", type=str, help="Path to input image.")
    parser.add_argument("--epochs", default=5000, type=int, help="Number of training epochs.")
    parser.add_argument("--log_n_epochs", default=500, type=int, help="Metric logging frequency.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--log_image_evolution",
        action="store_true",
        help="Accepted for compatibility only. Image evolution saving is disabled in this grid-ready script.",
    )

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="adam, muon, lr-sign, lr_sign10_rsclF, auto_cos_inc_rank, ...")
    parser.add_argument("--lr", default=1e-3, type=float, help="LR for Adam or auxiliary Adam.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)

    # Muon / low-rank
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_aux_weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_lr", type=float, default=1e-1)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_no_nesterov", action="store_true")
    parser.add_argument("--optimize_first_layer_with_muon", action="store_true")

    # auto_cos_inc_rank
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--rank_start", type=int, default=None)
    parser.add_argument("--rank_end", type=int, default=250)
    parser.add_argument("--rank_warmup_steps", type=int, default=None)
    parser.add_argument("--rank_oversample", type=int, default=4)
    parser.add_argument("--lowrank_rescale", action="store_true")
    parser.add_argument("--lowrank_eps", type=float, default=1e-6)
    parser.add_argument("--small_ns_bfloat16", action="store_true")
    parser.add_argument("--auto_init_rank_start", action="store_true")
    parser.add_argument("--init_probe_steps", type=int, default=8)
    parser.add_argument("--init_energy", type=float, default=0.90)
    parser.add_argument("--init_round_multiple", type=int, default=8)

    # L-BFGS
    parser.add_argument("--lbfgs_max_iter", type=int, default=20)
    parser.add_argument("--lbfgs_history_size", type=int, default=100)

    # Scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step", "rank_wsd", "rank-wsd"],
    )
    parser.add_argument("--T_max", type=int, default=5000)
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6)
    parser.add_argument("--step_size", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.9)

    # Rank-WSD
    parser.add_argument("--rank_wsd_warmup_steps", type=int, default=0)
    parser.add_argument("--rank_wsd_decay_start_step", type=int, default=None)
    parser.add_argument("--rank_wsd_min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--rank_wsd_base_N_rand", type=int, default=None)

    # Models
    parser.add_argument(
        "--model",
        choices=[
            "relu_mlp",
            "relu_ffn",
            "gauss_ffn",
            "gauss_mlp",
            "siren_mlp",
            "wire_mlp",
            "real_wire",
            "relu_pos_enc",
            "finer_mlp",
        ],
        default="relu_mlp",
    )
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--hidden_dim", default=300, type=int)
    parser.add_argument("--mapping_size", default=128, type=int)

    # Activation parameters
    parser.add_argument("--fourier_sigma", default=10.0, type=float)
    parser.add_argument("--siren_omega", default="40.0,40.0,40.0,40.0", type=str)
    parser.add_argument("--finer_omega", default="40.0,40.0,40.0,40.0", type=str)
    parser.add_argument("--finer_init_bias", action="store_true")
    parser.add_argument("--finer_bias_scale", default=float(1 / math.sqrt(2)), type=float)
    parser.add_argument("--gauss_scale", default="0.0236", type=str)
    parser.add_argument("--wire_sigma", default="10.0", type=str)
    parser.add_argument("--wire_omega", default="20.0", type=str)

    # Task
    parser.add_argument("--task", default="super_resolution", type=str, choices=["super_resolution", "fitting"])
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--inpainting_ratio", default=1.0, type=float)
    parser.add_argument("--sr_train_chunk_size", default=65536, type=int)
    parser.add_argument("--sr_eval_chunk_size", default=262144, type=int)

    # Output
    parser.add_argument("--folder_name", default=None, type=str)
    parser.add_argument("--save_model", action="store_true", help="Save model_weights.pth. Default: disabled for large grid runs.")
    parser.add_argument("--save_final_reconstruction", action="store_true", help="Save final_reconstruction.png. Default: disabled.")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS and write nan. Useful for fast debugging only.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.optimizer = normalize_optimizer_name(args.optimizer)
    if args.scheduler == "rank-wsd":
        args.scheduler = "rank_wsd"

    resolve_auto_cos_inc_rank_args(args)
    set_seed(args.seed)

    args.gauss_scale = parse_list(args.gauss_scale, args.num_layers)
    args.siren_omega = parse_list(args.siren_omega, args.num_layers)
    args.finer_omega = parse_list(args.finer_omega, args.num_layers)
    args.wire_sigma = parse_list(args.wire_sigma, args.num_layers)
    args.wire_omega = parse_list(args.wire_omega, args.num_layers)

    if not 0.0 < args.inpainting_ratio <= 1.0:
        raise ValueError("--inpainting_ratio must be in (0, 1].")
    if args.scale_factor < 1:
        raise ValueError("--scale_factor must be >= 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.log_n_epochs <= 0:
        raise ValueError("--log_n_epochs must be > 0.")

    output_dir = get_output_dir(args)
    save_args(args, output_dir)
    print(f"Results will be saved to: {output_dir}")

    model, metrics, layer_metrics, log_records, original_img, lpips_model = train_model(args, output_dir)

    model.eval()
    with torch.no_grad():
        h, w, c = original_img.shape
        device = next(model.parameters()).device
        coords = get_coordinates(h, w).to(device)

        final_pred = forward_in_chunks(model, coords, args.sr_eval_chunk_size).view(h, w, c).cpu()
        final_psnr = float(psnr(final_pred, original_img).item())
        final_ssim = float(ssim(original_img.numpy(), final_pred.numpy(), channel_axis=-1, data_range=1.0))
        final_lpips = float("nan") if args.skip_lpips else compute_lpips(original_img.numpy(), final_pred.numpy(), lpips_model, device)

        print("\n--- Final Results ---")
        print(f"Final Full Image PSNR: {final_psnr:.2f}")
        print(f"Final Full Image SSIM: {final_ssim:.4f}")
        print(f"Final Full Image LPIPS: {final_lpips:.4f}")

        if args.save_final_reconstruction:
            save_image(final_pred.numpy(), os.path.join(output_dir, "final_reconstruction.png"))
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pth"))

        save_readme(output_dir, final_psnr, final_ssim, final_lpips)

    save_metrics_csv(metrics, layer_metrics, log_records, output_dir)
    print(f"Training finished and scalar results saved to {output_dir}.")


if __name__ == "__main__":
    main()
