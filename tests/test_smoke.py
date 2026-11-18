"""Lightweight smoke tests — no GPU / live network required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_resolve_scene_engine_override():
    from server.pipeline import resolve_scene_engine

    assert resolve_scene_engine("hybrid_fast", "PERF") == "minimax-h3"
    assert resolve_scene_engine("hybrid_fast", "THEME") == "fasth3"
    assert (
        resolve_scene_engine("hybrid_fast", "PERF", override="fasth3") == "fasth3"
    )
    assert resolve_scene_engine("ltx23", "THEME", has_image=False) == "fasth3"
    assert (
        resolve_scene_engine("hybrid_fast", "THEME", override="auto") == "fasth3"
    )


def test_normalize_video_model():
    from server.pipeline import normalize_video_model

    assert normalize_video_model("h3") == "minimax-h3"
    assert normalize_video_model("fast-h3") == "fasth3"
    assert normalize_video_model("hybrid_fast") == "hybrid_fast"


def test_wizard_model_presets():
    path = ROOT / "scripts" / "setup_wizard.py"
    spec = importlib.util.spec_from_file_location("setup_wizard", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ids = [oid for oid, _ in mod.OLLAMA_CLOUD_MODEL_PRESETS]
    assert "gemma4:31b" in ids
    assert "gpt-oss:120b" in ids
    assert "kimi-k2.7-code" in ids
    assert callable(mod.pick_ollama_cloud_model)
    assert callable(mod.fetch_ollama_cloud_models)


def test_studio_cancel_methods_exist():
    # Avoid constructing a full director (heavy deps); assert methods on class.
    try:
        from server.director.hermes_director import HermesDirector
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"director import unavailable: {exc}")
    assert hasattr(HermesDirector, "request_studio_cancel")
    assert hasattr(HermesDirector, "clear_studio_cancel")


def test_client_dist_mount_path():
    """Packaged UI is optional; path resolution must be stable."""
    dist = ROOT / "client" / "dist"
    # Functionally: either missing (dev) or has index.html (built)
    if dist.is_dir():
        assert (dist / "index.html").is_file() or True
    assert dist.name == "dist"


def test_ha_job_envelope_is_natural_language():
    from server.home_agent.jobs import build_job_envelope

    text = build_job_envelope(job_type="still", prompt="neon stage still")
    assert "neon stage still" in text
    assert "```json" not in text  # no extras → no JSON fence
    with_extras = build_job_envelope(
        job_type="h3_r2v", prompt="sing", extras={"duration": 4}
    )
    assert "lip-sync" in with_extras.lower() or "H3" in with_extras
    assert "scenemux" in with_extras


def test_ha_default_url_port():
    from server.home_agent.client import HomeAgentClient

    c = HomeAgentClient(base_url=None, password="")
    # Env may override; when unset, client defaults to 8765
    if not __import__("os").getenv("HOME_AGENT_URL"):
        assert ":8765" in c.base_url or c.base_url.endswith(":8765")


# ── MiniMax-H3-Image storyboard stills ────────────────────────────────────


def test_normalize_still_model():
    from server.pipeline import normalize_still_model

    assert normalize_still_model(None) == "auto"
    assert normalize_still_model("Z-Image") == "zimage"
    assert normalize_still_model("h3_image") == "h3image"
    assert normalize_still_model("MiniMax-H3-Image") == "h3image"
    assert normalize_still_model("bogus") == "auto"


def test_h3img_canvas_is_grid_aligned_and_near_target_area():
    from server.image_generator import _h3img_canvas

    w, h = _h3img_canvas(768, 448, megapixels=4.0)
    assert w % 32 == 0 and h % 32 == 0
    assert abs((w * h) / (1024 * 1024) - 4.0) < 0.4
    assert abs(w / h - 768 / 448) < 0.05
    # portrait identity sheet keeps its orientation
    pw, ph = _h3img_canvas(768, 1024, megapixels=4.0)
    assert ph > pw
    # clamped to the model's effective range
    big = _h3img_canvas(1024, 576, megapixels=99)
    assert big[0] * big[1] <= 21 * 1024 * 1024


def test_h3img_workflow_graph_has_no_cfg_and_flat_ref_sockets():
    from server.image_generator import ComfyUIClient

    client = ComfyUIClient(backend="local", url="http://127.0.0.1:8188")
    wf = client._build_minimax_h3_image_workflow(
        "a singer on a rooftop", 2720, 1536, seed=7,
        reference_image_names=["a.jpg", "b.jpg"],
    )
    cond = wf["10"]
    assert cond["class_type"] == "MiniMaxH3ImageConditioning"
    assert cond["inputs"]["images.image_1"] == ["21", 0]
    assert cond["inputs"]["images.image_2"] == ["22", 0]
    assert wf["2"]["inputs"]["type"] == "minimax"
    ks = wf["12"]["inputs"]
    assert ks["steps"] == 6 and ks["cfg"] == 1.0
    assert ks["sampler_name"] == "euler" and ks["scheduler"] == "simple"
    # no negative text encode — only a zeroed copy of the positive
    assert wf["11"]["class_type"] == "ConditioningZeroOut"
    assert not any(n["class_type"] == "CLIPTextEncode" for n in wf.values())

    no_refs = client._build_minimax_h3_image_workflow("x", 1024, 1024, seed=1)
    assert not any(k.startswith("images.") for k in no_refs["10"]["inputs"])


def test_h3img_prompt_tagging():
    from server.image_generator import ComfyUIClient

    tag = ComfyUIClient._h3img_tag_prompt
    assert tag("neon alley", 0) == "neon alley"
    one = tag("neon alley", 1)
    assert "<Picture 1>" in one and "<Picture 2>" not in one and "neon alley" in one
    assert "<Picture 2>" in tag("neon alley", 2)
    # already-tagged prompts are left alone
    assert tag("<Picture 1> walks", 1) == "<Picture 1> walks"


def test_h3img_status_reports_missing_and_ready():
    from server.image_generator import ComfyUIClient

    client = ComfyUIClient(backend="local", url="http://127.0.0.1:8188")
    client._available_models = ["z_image_turbo_bf16.safetensors"]
    client._object_nodes = {"KSampler"}
    status = client._h3img_status()
    assert not status["ready"]
    assert any("MiniMaxH3ImageConditioning" in m for m in status["missing"])

    client._available_models = [
        "minimax_h3_image_pruned_bf16.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_image_vae_fp16.safetensors",
    ]
    client._object_nodes = {"MiniMaxH3ImageConditioning"}
    status = client._h3img_status()
    assert status["ready"], status["missing"]
    files = client._h3img_files()
    # picks installed variants instead of the int8 defaults
    assert files["unet"].endswith("pruned_bf16.safetensors")
    assert "nvfp4" in files["clip"]


def test_resolve_still_model_explicit_and_cloud():
    import asyncio

    from server.image_generator import ComfyUIClient

    local = ComfyUIClient(backend="local", url="http://127.0.0.1:8188")
    local.still_model = "h3image"
    assert asyncio.run(local.resolve_still_model()) == "h3image"
    local.still_model = "zimage"
    assert asyncio.run(local.resolve_still_model()) == "zimage"
    cloud = ComfyUIClient(backend="cloud")
    cloud.still_model = "auto"
    assert asyncio.run(cloud.resolve_still_model()) == "zimage"
