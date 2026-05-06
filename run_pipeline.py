#!/usr/bin/env python3
"""
WorldFM end-to-end pipeline.

Input:  meta.json  (name, image, K, c2w)
Output: generated images at the specified camera poses.

Steps:
  1. Perspective image  ->  panorama           (modules.panogen)
  2. Panorama           ->  depth/PLY/conditions  (modules.moge_pano + pano_postprocess)
  3. Target pose        ->  condition_render + condition_nearest  (modules.point_renderer + depth_selector)
  4. Condition pair     ->  final generated image  (modules.worldfm_infer)

All inter-step data is passed in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm

WORLDFM_ROOT = Path(__file__).resolve().parent
SUBMODULES = WORLDFM_ROOT / "submodules"
DEFAULT_CFG = OmegaConf.load(str(WORLDFM_ROOT / "default.yaml"))

# ---------------------------------------------------------------------------
# Local modules — moge_pano and panogen use try/except for external deps,
# so importing them here is safe even before setup_external_repos().
# MoGeModel etc. are accessed via moge_pano.<attr> because ensure_moge()
# rebinds the module-level globals at runtime.
# ---------------------------------------------------------------------------
import modules.moge_pano as moge_pano
from modules.moge_pano import (
    ensure_moge,
    select_tier,
    _get_panorama_cameras,
)
from modules.panogen import ensure_hy3dworld, Image2PanoramaDemo
from modules.pano_postprocess import PostProcessResult, postprocess_panorama
from modules.point_renderer import TorchPointCloudRenderer
from modules.depth_selector import (
    build_condition_db_in_memory,
    select_best_condition_index,
)
from modules.worldfm_infer import WorldFMInprocessConfig, WorldFMTriConditionInprocess


# ============================== helpers ======================================

def _load_meta(meta_path: str) -> dict:
    """Load and validate meta.json. Normalises c2w to a list of 4×4 matrices."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    required = ("name", "image", "K", "c2w")
    for k in required:
        if k not in meta:
            raise KeyError(f"meta.json missing required key: {k}")
    c2w = np.asarray(meta["c2w"], dtype=np.float64)
    if c2w.ndim == 2:
        meta["c2w"] = [c2w.tolist()]
    elif c2w.ndim != 3:
        raise ValueError(f"c2w must be (4,4) or (N,4,4), got shape {c2w.shape}")
    return meta


def _log(step: str, msg: str) -> None:
    print(f"[WorldFM][{step}] {msg}", flush=True)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync_cuda()
    return time.perf_counter()


def _safe_rate(count: float, seconds: float) -> float | None:
    return (count / seconds) if seconds > 0 else None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Profiler:
    """Collect structured performance data without printing during generation."""

    filename = "performance.json"

    def __init__(self, *, enabled: bool, output_dir: Path, warmup_frames: int = 0) -> None:
        self.enabled = bool(enabled)
        self.output_path = output_dir / self.filename
        self.warmup_frames = max(0, int(warmup_frames))
        self.events: list[dict] = []
        self.frames: list[dict] = []

    def now(self) -> float:
        return _now() if self.enabled else 0.0

    def elapsed(self, start: float) -> float:
        return (_now() - start) if self.enabled else 0.0

    def record_event(self, name: str, *, duration_sec: float, **fields) -> None:
        if not self.enabled:
            return
        event = {"name": name, "duration_sec": float(duration_sec)}
        event.update(fields)
        self.events.append(event)

    def record_frame(
        self,
        *,
        frame_index: int,
        total_frames: int,
        condition_index: int,
        condition_hits: int,
        condition_samples: int,
        render_select_sec: float,
        inference_sec: float,
        frame_total_sec: float,
        worldfm_profile: dict | None = None,
    ) -> None:
        if not self.enabled:
            return
        frame = {
            "frame_index": int(frame_index),
            "total_frames": int(total_frames),
            "condition": {
                "index": int(condition_index),
                "hits": int(condition_hits),
                "samples": int(condition_samples),
            },
            "timings_sec": {
                "render_select": float(render_select_sec),
                "inference": float(inference_sec),
                "frame_total": float(frame_total_sec),
            },
            "fps": {
                "inference": _safe_rate(1.0, float(inference_sec)),
                "end_to_end": _safe_rate(1.0, float(frame_total_sec)),
            },
        }
        if worldfm_profile:
            frame["worldfm_ms"] = {
                str(key): value
                for key, value in (
                    (key, _float_or_none(value)) for key, value in worldfm_profile.items()
                )
                if value is not None
            }
        self.frames.append(frame)

    def _summary_for(self, frames: list[dict]) -> dict:
        n_frames = len(frames)
        infer_sum = float(sum(frame["timings_sec"]["inference"] for frame in frames))
        total_sum = float(sum(frame["timings_sec"]["frame_total"] for frame in frames))
        return {
            "frames": n_frames,
            "avg_inference_sec": (infer_sum / n_frames) if n_frames else None,
            "avg_frame_total_sec": (total_sum / n_frames) if n_frames else None,
            "inference_fps": _safe_rate(n_frames, infer_sum),
            "end_to_end_fps": _safe_rate(n_frames, total_sum),
        }

    def summary(self) -> dict:
        warmup = max(0, min(self.warmup_frames, max(len(self.frames) - 1, 0)))
        steady_frames = self.frames[warmup:] if warmup else self.frames
        return {
            "all_frames": self._summary_for(self.frames),
            "steady_state": {
                "skipped_warmup_frames": warmup,
                **self._summary_for(steady_frames),
            },
        }

    def write(self) -> Path | None:
        if not self.enabled:
            return None
        payload = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "warmup_frames": self.warmup_frames,
            "events": self.events,
            "frames": self.frames,
            "summary": self.summary(),
        }
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.output_path


def _intermediates_dir(output_dir: Path) -> Path:
    return output_dir / "intermediates"


def _save_postprocess_result(result: PostProcessResult, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(save_dir / "postprocess_arrays.npz"),
        pano_bgr=result.pano_bgr,
        depth=result.depth.astype(np.float32, copy=False),
        ply_xyz=result.ply_xyz.astype(np.float32, copy=False),
        ply_rgb=result.ply_rgb.astype(np.uint8, copy=False),
    )


def _load_postprocess_result(save_dir: Path) -> PostProcessResult:
    arrays_path = save_dir / "postprocess_arrays.npz"
    transforms_path = save_dir / "transforms_condition.json"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing cached arrays: {arrays_path}")
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing cached transforms: {transforms_path}")

    arrays = np.load(str(arrays_path))
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    condition_images = []
    for frame in transforms.get("frames", []):
        img_path = save_dir / str(frame["path"])
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing cached condition image: {img_path}")
        condition_images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    return PostProcessResult(
        pano_bgr=arrays["pano_bgr"],
        depth=arrays["depth"],
        ply_xyz=arrays["ply_xyz"],
        ply_rgb=arrays["ply_rgb"],
        condition_images=condition_images,
        transforms=transforms,
    )


def setup_external_repos(*, hw_path: str = "", moge_path: str = "") -> None:
    """Register external repo paths on sys.path **before** any model imports.

    Must be called once at the very beginning of the pipeline, because
    hy3dworld internally imports moge, so MoGe must be available first.
    Additionally, hy3dworld uses realesrgan and zim_anything, so their
    repos (Real-ESRGAN, ZIM) must also be on sys.path.
    """
    resolved_moge = moge_path or str(SUBMODULES / "MoGe")
    if Path(resolved_moge).exists():
        _log("Init", f"ensure_moge({resolved_moge})")
        ensure_moge(resolved_moge)

    for dep in ("Real-ESRGAN", "ZIM"):
        dep_path = str(SUBMODULES / dep)
        if Path(dep_path).exists() and dep_path not in sys.path:
            sys.path.insert(0, dep_path)
            _log("Init", f"sys.path += {dep_path}")

    resolved_hw = hw_path or str(SUBMODULES / "HunyuanWorld-1.0")
    if Path(resolved_hw).exists():
        _log("Init", f"ensure_hy3dworld({resolved_hw})")
        ensure_hy3dworld(resolved_hw)


# ============================== Step 1 =======================================

def step1_panogen(image_path: str, output_dir: Path, *, cfg=None, save_intermediates: bool = False):
    """Perspective image -> panorama (PIL Image).

    Returns PIL.Image.Image (panorama).
    Requires setup_external_repos() to have been called first.
    """
    pcfg = (cfg or DEFAULT_CFG).panogen
    _log("Step1", f"Generating panorama from {image_path}")

    pano_disk = output_dir / "panorama.png"
    if pano_disk.exists():
        _log("Step1", f"Panorama already exists, loading: {pano_disk}")
        return Image.open(pano_disk).convert("RGB")

    class _Args:
        fp8_attention = bool(pcfg.fp8_attention)
        fp8_gemm = bool(pcfg.fp8_gemm)
        cache = bool(pcfg.cache)

    demo = Image2PanoramaDemo(_Args())

    output_dir.mkdir(parents=True, exist_ok=True)
    pano_img = demo.run(
        prompt="",
        negative_prompt="",
        image_path=str(image_path),
        seed=int(pcfg.seed),
        save_to_disk=False,
        output_path=None,
    )

    _log("Step1", f"Panorama generated: {np.array(pano_img).shape}")
    if save_intermediates:
        pano_img.save(str(pano_disk))
        _log("Step1", f"Panorama saved: {pano_disk}")
    return pano_img


# ============================== Step 2 =======================================

def step2_moge_pipeline(panorama_img, output_dir: Path, *, cfg=None, pretrained: str = "", save_intermediates: bool = False):
    """Panorama image -> depth + PLY arrays + condition images + transforms.

    Returns modules.pano_postprocess.PostProcessResult.
    Requires setup_external_repos() to have been called first.
    """
    mcfg = (cfg or DEFAULT_CFG).moge
    pretrained = pretrained or mcfg.pretrained
    resolution_level = int(mcfg.resolution_level)
    fov_deg = float(mcfg.fov_deg)
    num_views = int(mcfg.num_views)
    merge_max_w = int(mcfg.merge_max_width)
    merge_max_h = int(mcfg.merge_max_height)
    batch_size = int(mcfg.batch_size)

    _log("Step2", "Running MoGe + postprocess")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    image_rgb = np.array(panorama_img)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    tier = select_tier(orig_w)
    tgt_w, tgt_h = tier["width"], tier["height"]
    split_resolution = tier["split_res"]
    _log("Step2", f"tier={tier['name']} ({tgt_w}x{tgt_h}), input={orig_w}x{orig_h}")

    if orig_w != tgt_w or orig_h != tgt_h:
        interp = cv2.INTER_AREA if tgt_w < orig_w else cv2.INTER_LINEAR
        image_rgb = cv2.resize(image_rgb, (tgt_w, tgt_h), interpolation=interp)
    height, width = image_rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = moge_pano.MoGeModel.from_pretrained(pretrained).to(device).eval()
    _log("Step2", f"MoGe model loaded on {device}")

    import utils3d
    extrinsics, intrinsics = _get_panorama_cameras(num_views, fov_deg)
    splitted_images = moge_pano.split_panorama_image(image_rgb, extrinsics, intrinsics, split_resolution)

    splitted_dist, splitted_masks = [], []
    for i in trange(0, len(splitted_images), batch_size, desc="MoGe Infer", leave=False):
        batch = np.stack(splitted_images[i:i + batch_size])
        tensor = torch.tensor(batch / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[i:i + batch_size])))
        fov_x_t = torch.tensor(fov_x, dtype=torch.float32, device=device)
        out = model.infer(tensor, resolution_level=resolution_level, fov_x=fov_x_t, apply_mask=False)
        splitted_dist.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        splitted_masks.extend(list(out["mask"].cpu().numpy()))

    _log("Step2", "Merge panorama depth")
    merging_w = min(merge_max_w, width)
    merging_h = min(merge_max_h, height)
    panorama_depth, panorama_mask = moge_pano.merge_panorama_depth(
        merging_w, merging_h, splitted_dist, splitted_masks, extrinsics, intrinsics,
    )
    panorama_depth = panorama_depth.astype(np.float32)
    panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0

    depth_raw = panorama_depth.copy()
    if panorama_mask.any():
        depth_raw[~panorama_mask] = panorama_depth[panorama_mask].max()
    depth_raw = depth_raw / 100.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pano_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    save_dir = _intermediates_dir(output_dir) if save_intermediates else None
    result = postprocess_panorama(pano_bgr, depth_raw, save_dir=save_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "panorama.png"), pano_bgr)
        np.save(str(save_dir / "moge_depth_raw.npy"), depth_raw.astype(np.float32, copy=False))
        cv2.imwrite(str(save_dir / "moge_mask.png"), (panorama_mask.astype(np.uint8) * 255))
        _save_postprocess_result(result, save_dir)
        _log("Step2", f"Saving intermediates: {save_dir}")
    _log("Step2", f"PLY: {result.ply_xyz.shape[0]:,} points, conditions: {len(result.condition_images)}")
    return result


# ============================== Step 3 =======================================

def step3_init(pp_result, *, cfg=None, render_size: int = 0):
    """Create renderer and condition DB (heavy objects, reusable across frames).

    Returns (renderer, cond_db, rcfg, render_size).
    """
    rcfg = (cfg or DEFAULT_CFG).render
    S = render_size or int(rcfg.render_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = TorchPointCloudRenderer(
        points_xyz=pp_result.ply_xyz,
        points_rgb=pp_result.ply_rgb / 255.0 if pp_result.ply_rgb.dtype == np.uint8 else pp_result.ply_rgb,
        width=S, height=S, device=str(device), mode="fast",
    )

    cond_db = build_condition_db_in_memory(
        condition_images=pp_result.condition_images,
        transforms_dict=pp_result.transforms,
        torch_renderer=renderer,
        device=device,
    )

    return renderer, cond_db, rcfg, S


def step3_render_one(
    renderer,
    cond_db,
    pp_result,
    K: np.ndarray,
    c2w: np.ndarray,
    *,
    rcfg=None,
    render_size: int = 512,
):
    """Render condition pair for a single target pose.

    Returns (render_rgb_u8: torch.Tensor, cond_nearest_resized: np.ndarray, cond_index: int).
    """
    rcfg = rcfg or DEFAULT_CFG.render
    S = render_size

    K_arr = np.asarray(K, dtype=np.float64)
    c2w_arr = np.asarray(c2w, dtype=np.float64)

    out = renderer.render_torch(K_3x3=K_arr, c2w_4x4=c2w_arr, c2w_is_camera_to_world=True)
    rgb_u8 = out.rgb_u8
    depth = out.depth_f32

    idx, hits, samples = select_best_condition_index(
        depth_cur=depth,
        K_cur=K_arr, c2w_cur=c2w_arr,
        cond_db=cond_db,
        sample_grid=int(rcfg.sample_grid),
        center_grid=int(rcfg.center_grid),
        center_frac=float(rcfg.center_frac),
        eps_rel=float(rcfg.eps_rel),
        eps_abs=float(rcfg.eps_abs),
        px_radius=int(rcfg.px_radius),
        max_view_angle_deg=float(rcfg.max_view_angle_deg),
        use_distance_weight=bool(rcfg.use_distance_weight),
        dist_min_m=float(rcfg.dist_min_m),
        dist_max_m=float(rcfg.dist_max_m),
        weight_near=float(rcfg.weight_near),
        weight_far=float(rcfg.weight_far),
    )
    cond_nearest_rgb = pp_result.condition_images[int(idx)]
    cond_nearest_resized = np.array(
        Image.fromarray(cond_nearest_rgb, "RGB").resize((S, S), resample=Image.BILINEAR)
    )

    return rgb_u8, cond_nearest_resized, int(idx), int(hits), int(samples)


# ============================== Step 4 =======================================

def step4_init(*, cfg=None):
    """Load WorldFM inference service (heavy, reusable across frames).

    Returns (svc, wcfg).
    """
    wcfg = (cfg or DEFAULT_CFG).worldfm
    model_path = str(wcfg.model_path)
    vae_path = str(wcfg.vae_path)
    image_size = int(wcfg.image_size)
    step = int(wcfg.step)
    compile_model = bool(wcfg.get("compile_model", False))
    compile_mode = str(wcfg.get("compile_mode", "reduce-overhead"))
    vae_channels_last = bool(wcfg.get("vae_channels_last", True))
    vae_deterministic = bool(wcfg.get("vae_deterministic", True))
    compile_vae = bool(wcfg.get("compile_vae", True))
    compile_vae_mode = str(wcfg.get("compile_vae_mode", "reduce-overhead"))
    disable_vae_slicing = bool(wcfg.get("disable_vae_slicing", True))
    disable_vae_tiling = bool(wcfg.get("disable_vae_tiling", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = f"cuda:{torch.cuda.current_device()}" if device.type == "cuda" else "cpu"

    model_p = Path(model_path).resolve()
    vae_p = Path(vae_path)
    if not vae_p.is_absolute():
        vae_p = (WORLDFM_ROOT / vae_p).resolve()

    svc = WorldFMTriConditionInprocess(
        WorldFMInprocessConfig(
            model_path=str(model_p),
            vae_path=str(vae_p),
            image_size=image_size,
            version=str(wcfg.version),
            disable_cross_attn=True,
            step=(step if step in (1, 2) else 2),
            mid_t=200, cfg_scale=0.0,
            device=device_str,
            weight_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            compile_model=compile_model,
            compile_mode=compile_mode,
            vae_channels_last=vae_channels_last,
            vae_deterministic=vae_deterministic,
            compile_vae=compile_vae,
            compile_vae_mode=compile_vae_mode,
            disable_vae_slicing=disable_vae_slicing,
            disable_vae_tiling=disable_vae_tiling,
        )
    )

    return svc, wcfg


def step4_infer_one(
    svc,
    render_rgb_u8,
    cond_nearest_rgb: np.ndarray | None,
    *,
    wcfg=None,
    cond2_index: int | None = None,
    profile: bool = False,
) -> np.ndarray:
    """Run WorldFM inference for a single frame.

    Returns (H, W, 3) uint8 numpy array (RGB).
    """
    wcfg = wcfg or DEFAULT_CFG.worldfm
    step = int(wcfg.step)
    cfg_scale = float(wcfg.cfg_scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cond2_index is None:
        if cond_nearest_rgb is None:
            raise ValueError("cond_nearest_rgb is required when cond2_index is not provided")
        svc.set_cond2_from_array(cond_nearest_rgb)

    if isinstance(render_rgb_u8, torch.Tensor):
        render_u8 = render_rgb_u8
    else:
        render_u8 = torch.from_numpy(
            np.asarray(render_rgb_u8, dtype=np.uint8)
        ).to(device=device, dtype=torch.uint8)

    if step in (1, 2):
        decoded = svc.infer_from_render_u8(render_u8, cond2_index=cond2_index, profile=profile)
    else:
        decoded = svc.infer_from_render_u8_multistep(
            render_u8, sample_steps=step, cfg_scale=cfg_scale, cond2_index=cond2_index,
        )

    out_u8 = (
        torch.clamp(127.5 * decoded[0] + 128.0, 0, 255)
        .permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    )
    return out_u8


# ============================== main =========================================

def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_CFG
    p = argparse.ArgumentParser(
        description="WorldFM pipeline: perspective image + target poses -> generated images",
    )
    p.add_argument("--config", type=str, default="",
                   help="Override config YAML (merged on top of default.yaml)")
    p.add_argument("--meta", type=str, required=True,
                   help="Path to meta.json (name, image, K, c2w)")
    p.add_argument("--output_dir", type=str, default=d.pipeline.output_dir,
                   help="Base output directory")

    p.add_argument("--hw_path", type=str, default=d.submodules.hw_path,
                   help="HunyuanWorld-1.0 repo path (auto-detect if empty)")
    p.add_argument("--moge_path", type=str, default=d.submodules.moge_path,
                   help="MoGe repo path (auto-detect if empty)")

    p.add_argument("--moge_pretrained", type=str, default=d.moge.pretrained,
                   help="MoGe pretrained model path")
    p.add_argument("--render_size", type=int, default=d.render.render_size,
                   help="Point-cloud render resolution (square)")

    p.add_argument("--model_path", type=str, default=d.worldfm.model_path,
                   help="WorldFM model checkpoint path")
    p.add_argument("--vae_path", type=str, default=d.worldfm.vae_path,
                   help="Path to VAE directory (AutoencoderKL)")
    p.add_argument("--image_size", type=int, default=d.worldfm.image_size,
                   help="WorldFM inference resolution")
    p.add_argument("--step", type=int, default=d.worldfm.step,
                   help="WorldFM inference steps (1 or 2)", choices=[1, 2])
    p.add_argument("--cfg_scale", type=float, default=d.worldfm.cfg_scale,
                   help="CFG scale for multi-step sampling")
    p.add_argument("--compile_worldfm", dest="compile_worldfm", action="store_true",
                   default=bool(d.worldfm.get("compile_model", False)),
                   help="Enable torch.compile for the WorldFM denoiser")
    p.add_argument("--no_compile_worldfm", dest="compile_worldfm", action="store_false",
                   help="Disable torch.compile for the WorldFM denoiser")
    p.add_argument("--compile_mode", type=str, default=str(d.worldfm.get("compile_mode", "reduce-overhead")),
                   help="torch.compile mode for WorldFM denoiser")
    p.add_argument("--vae_channels_last", dest="vae_channels_last", action="store_true",
                   default=bool(d.worldfm.get("vae_channels_last", True)),
                   help="Use channels-last memory format for VAE tensors")
    p.add_argument("--no_vae_channels_last", dest="vae_channels_last", action="store_false",
                   help="Disable VAE channels-last memory format")
    p.add_argument("--vae_deterministic", dest="vae_deterministic", action="store_true",
                   default=bool(d.worldfm.get("vae_deterministic", True)),
                   help="Use VAE latent mode instead of latent sampling")
    p.add_argument("--vae_sample", dest="vae_deterministic", action="store_false",
                   help="Use stochastic VAE latent sampling")
    p.add_argument("--compile_vae", dest="compile_vae", action="store_true",
                   default=bool(d.worldfm.get("compile_vae", True)),
                   help="Enable torch.compile for VAE encode/decode wrappers")
    p.add_argument("--no_compile_vae", dest="compile_vae", action="store_false",
                   help="Disable torch.compile for VAE encode/decode wrappers")
    p.add_argument("--compile_vae_mode", type=str, default=str(d.worldfm.get("compile_vae_mode", "reduce-overhead")),
                   help="torch.compile mode for VAE encode/decode wrappers")
    p.add_argument("--disable_vae_slicing", dest="disable_vae_slicing", action="store_true",
                   default=bool(d.worldfm.get("disable_vae_slicing", True)),
                   help="Disable diffusers VAE slicing for speed")
    p.add_argument("--enable_vae_slicing", dest="disable_vae_slicing", action="store_false",
                   help="Enable diffusers VAE slicing")
    p.add_argument("--disable_vae_tiling", dest="disable_vae_tiling", action="store_true",
                   default=bool(d.worldfm.get("disable_vae_tiling", True)),
                   help="Disable diffusers VAE tiling for speed")
    p.add_argument("--enable_vae_tiling", dest="disable_vae_tiling", action="store_false",
                   help="Enable diffusers VAE tiling")
    p.add_argument("--perf_warmup_frames", type=int, default=3,
                   help="Frames to exclude from steady-state performance summary")
    p.add_argument("--profile_worldfm", action="store_true",
                   help="Write structured performance profile to output_dir/performance_file")
    p.add_argument("--gpu_index", type=int, default=d.pipeline.gpu_index,
                   help="CUDA device index")
    p.add_argument("--save_mode", type=str, default="video",
                   choices=["image", "video"],
                   help="Output format: 'image' saves per-frame PNGs, 'video' saves MP4 (default: video)")
    p.add_argument("--fps", type=int, default=30,
                   help="Video frame rate when --save_mode=video (default: 30)")
    p.add_argument("--save_intermediates", action="store_true",
                   help="Save panorama/depth/point cloud/conditions before WorldFM inference")
    p.add_argument("--reuse_intermediates", action="store_true",
                   help="Load cached intermediates and skip panorama + depth + point-cloud preprocessing")
    p.add_argument("--prepare_only", action="store_true",
                   help="Save/load preprocessing intermediates, then exit before WorldFM inference")
    return p


def _load_config(args) -> OmegaConf:
    """Merge default.yaml <- user config <- CLI overrides."""
    cfg = OmegaConf.create(DEFAULT_CFG)
    if args.config:
        user_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, user_cfg)
    cli_overrides = OmegaConf.create({
        "pipeline": {"output_dir": args.output_dir,
                      "gpu_index": args.gpu_index},
        "submodules": {"hw_path": args.hw_path, "moge_path": args.moge_path},
        "moge": {"pretrained": "Ruicheng/moge-2-vitl-normal" if args.moge_pretrained is None else args.moge_pretrained},
        "render": {"render_size": args.render_size},
        "worldfm": {"model_path": args.model_path,
                     "vae_path": args.vae_path,
                     "image_size": args.image_size, "step": args.step,
                     "cfg_scale": args.cfg_scale,
                     "compile_model": bool(args.compile_worldfm),
                     "compile_mode": args.compile_mode,
                     "vae_channels_last": bool(args.vae_channels_last),
                     "vae_deterministic": bool(args.vae_deterministic),
                     "compile_vae": bool(args.compile_vae),
                     "compile_vae_mode": args.compile_vae_mode,
                     "disable_vae_slicing": bool(args.disable_vae_slicing),
                     "disable_vae_tiling": bool(args.disable_vae_tiling)},
    })
    cfg = OmegaConf.merge(cfg, cli_overrides)
    return cfg


def main() -> int:
    args = build_parser().parse_args()
    cfg = _load_config(args)

    if cfg.pipeline.gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(cfg.pipeline.gpu_index))

    meta_path = Path(args.meta).resolve()
    meta = _load_meta(str(meta_path))
    meta_dir = meta_path.parent

    name = meta["name"]
    image_path = (meta_dir / meta["image"]).resolve()
    K = np.asarray(meta["K"], dtype=np.float64)
    c2w_list = [np.asarray(c, dtype=np.float64) for c in meta["c2w"]]

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    base_output = Path(str(cfg.pipeline.output_dir))
    if not base_output.is_absolute():
        base_output = (WORLDFM_ROOT / base_output).resolve()
    output_dir = base_output / name
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Main", f"name={name}")
    _log("Main", f"image={image_path}")
    _log("Main", f"output_dir={output_dir}")
    _log("Main", f"poses={len(c2w_list)}")

    # ---- Setup external repos (MoGe first, then HunyuanWorld) ----
    setup_external_repos(
        hw_path=str(cfg.submodules.hw_path),
        moge_path=str(cfg.submodules.moge_path),
    )

    if args.reuse_intermediates:
        cache_dir = _intermediates_dir(output_dir)
        _log("Main", f"Loading cached intermediates: {cache_dir}")
        pp_result = _load_postprocess_result(cache_dir)
        _log("Main", f"Cached PLY: {pp_result.ply_xyz.shape[0]:,} points, conditions: {len(pp_result.condition_images)}")
    else:
        # ---- Step 1: Perspective -> Panorama (PIL Image) ----
        panorama_img = step1_panogen(
            image_path=str(image_path),
            output_dir=output_dir,
            cfg=cfg,
        )

        # ---- Step 2: Panorama -> depth/PLY/conditions (in memory) ----
        pp_result = step2_moge_pipeline(
            panorama_img=panorama_img,
            output_dir=output_dir,
            cfg=cfg,
            save_intermediates=bool(args.save_intermediates or args.prepare_only),
        )

    if args.prepare_only:
        _log("Main", f"Prepare-only complete: intermediates in {_intermediates_dir(output_dir)}")
        return 0

    # ---- Step 3 init: renderer + condition DB (once) ----
    _log("Step3", "Initializing renderer and condition DB")
    renderer, cond_db, rcfg, S = step3_init(pp_result, cfg=cfg)

    # ---- Step 4 init: WorldFM service (once) ----
    _log("Step4", "Loading WorldFM inference service")
    svc, wcfg = step4_init(cfg=cfg)
    profiler = Profiler(
        enabled=bool(args.profile_worldfm),
        output_dir=output_dir,
        warmup_frames=int(args.perf_warmup_frames),
    )
    cond_cache_t0 = profiler.now()
    svc.set_cond2_candidates_from_arrays(pp_result.condition_images)
    profiler.record_event(
        "condition_latent_cache",
        duration_sec=profiler.elapsed(cond_cache_t0),
        condition_images=len(pp_result.condition_images),
    )

    # ---- Generate for each target pose ----
    save_mode = args.save_mode
    frames: list[np.ndarray] = []
    progress_bar = tqdm(c2w_list, desc="Generating frames", total=len(c2w_list))
    for i, c2w in enumerate(progress_bar):
        frame_t0 = profiler.now()
        render_u8, _cond_nearest_rgb, cond_idx, cond_hits, cond_samples = step3_render_one(
            renderer, cond_db, pp_result, K, c2w,
            rcfg=rcfg, render_size=S,
        )
        after_render = profiler.now()
        render_sec = after_render - frame_t0 if profiler.enabled else 0.0

        frame = step4_infer_one(
            svc,
            render_u8,
            None,
            wcfg=wcfg,
            cond2_index=cond_idx,
            profile=profiler.enabled,
        )
        after_infer = profiler.now()
        infer_sec = after_infer - after_render if profiler.enabled else 0.0
        total_sec = after_infer - frame_t0 if profiler.enabled else 0.0

        profiler.record_frame(
            frame_index=i + 1,
            total_frames=len(c2w_list),
            condition_index=cond_idx,
            condition_hits=cond_hits,
            condition_samples=cond_samples,
            render_select_sec=render_sec,
            inference_sec=infer_sec,
            frame_total_sec=total_sec,
            worldfm_profile=getattr(svc, "_last_profile", None),
        )

        if save_mode == "image":
            out_name = "output.png" if len(c2w_list) == 1 else f"output_{i:04d}.png"
            out_path = output_dir / out_name
            Image.fromarray(frame, mode="RGB").save(str(out_path))
        else:
            frames.append(frame)
    progress_bar.close()

    # ---- Save video ----
    if save_mode == "video" and frames:
        video_path = output_dir / "output.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        _log("Main", f"Video saved: {video_path} ({len(frames)} frames, {args.fps} fps)")

    performance_path = profiler.write()
    if performance_path is not None:
        _log("Performance", f"Saved structured profile: {performance_path}")

    # ---- Cleanup ----
    del renderer, cond_db, svc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(c2w_list)
    _log("Main", f"Pipeline complete: {n} frames ({'video' if save_mode == 'video' else 'images'}) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
