"""
SDTM (Structure-then-Detail Token Merging) for PixArt-Sigma

Adapted from the original SDTM implementation for SD3/MMDiT to work with
PixArt-Sigma's sequential attention architecture.

Key differences from SD3:
- PixArt uses BasicTransformerBlock with ada_norm_single
- Sequential: Self-Attention -> Cross-Attention -> Feed-Forward
- Cross-attention should NOT be merged (different token spaces)

Author: Adapted for PixArt-Sigma
"""

import torch
from typing import Tuple, Callable, Type, Dict, Any, Optional
import math
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def SSM(
    metric: torch.Tensor,
    reduce_num: int = 0,
    threshold: float = 0,
    window_size: Tuple[int, int] = (2, 2),
    no_rand: bool = False,
    generator: torch.Generator = None,
    tore_info: Dict = None
) -> Tuple[Callable, Callable, Callable]:
    """
    Similarity-prioritized Structure Merging (SSM)

    Used in early denoising steps (structure phase) to merge similar tokens
    within windows based on cosine similarity.
    """
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w
        assert num_token_window > 1, "window_size must produce at least 2 tokens (K>1)."

        B, N, D = metric.size()
        base_grid_H = int(math.sqrt(N))
        base_grid_W = base_grid_H
        assert base_grid_H * base_grid_W == N and base_grid_H % ws_h == 0 and base_grid_W % ws_w == 0

        # Reshape metric for window-based processing
        metric = metric.view(-1, base_grid_H, base_grid_W, D).permute(0, 3, 1, 2)
        metric = metric.view(B, D, base_grid_H // ws_h, ws_h, base_grid_W // ws_w, ws_w).permute(0, 2, 4, 1, 3, 5)
        b, gh, gw, c, ps_h, ps_w = metric.shape

        # Flatten window for pairwise operations
        tensor_flattened = metric.reshape(b, gh, gw, c, -1)

        # Compute cosine similarities within each window
        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)
        similarity_map = sims.sum(-1).sum(-1) / ((ps_h * ps_w) * (ps_h * ps_w))
        similarity_map = similarity_map.unsqueeze(1).reshape(similarity_map.size(0), -1)

        # Frequency priority score integration
        ssmscore_map = similarity_map
        if tore_info is not None and "states" in tore_info and tore_info["states"].get("last_independent") is not None:
            li = tore_info["states"]["last_independent"]
            if li.shape[1] == base_grid_H * base_grid_W:
                eps = 1e-6
                li_f = li.to(similarity_map.dtype)
                mean_li = li_f.mean(dim=1, keepdim=True) + eps
                indiv_priority = li_f / mean_li
                indiv_priority_grid = indiv_priority.view(B, base_grid_H, base_grid_W)
                indiv_priority_windows = (
                    indiv_priority_grid
                    .view(B, gh, ws_h, gw, ws_w)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(B, gh, gw, ws_h * ws_w)
                    .mean(-1)
                )
                indiv_priority_flat = indiv_priority_windows.view(B, gh * gw)
                a_s = tore_info.get("args", {}).get("a_s", 0.0)
                ssmscore_map = similarity_map + a_s * indiv_priority_flat

        # Adaptive reduce_num
        if reduce_num is None:
            n_B, n_H = ssmscore_map.shape
            node_mean = torch.tensor(threshold).cuda(sims.device)
            node_mean = node_mean.repeat(1, n_H)
            reduce_num = torch.ge(ssmscore_map, node_mean).sum(dim=1).min()
        else:
            reduce_num = reduce_num // 48 * 16

        # Get top k similar super patches
        _, sim_super_patch_idxs = ssmscore_map.topk(reduce_num, dim=-1)

        # Create mergeable and unmergeable super patches
        tensor = torch.arange(base_grid_H * base_grid_W, device=metric.device).reshape(base_grid_H, base_grid_W)
        tensor = tensor.unsqueeze(0).repeat(B, 1, 1)
        windowed_tensor = tensor.unfold(1, ws_h, stride_h).unfold(2, ws_w, stride_w)
        windowed_tensor = windowed_tensor.reshape(B, -1, num_token_window)

        gathered_tensor = torch.gather(windowed_tensor, 1, sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, num_token_window))

        mask = torch.ones((B, windowed_tensor.shape[1]), dtype=bool, device=metric.device)
        mask_values = torch.zeros_like(sim_super_patch_idxs, dtype=torch.bool, device=metric.device)
        mask.scatter_(1, sim_super_patch_idxs, mask_values)

        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, num_token_window)].reshape(B, -1, num_token_window)
        unm_idx = remaining_tensor.reshape(B, -1).unsqueeze(-1)

        K = num_token_window
        dim_index = K - 1

        if no_rand:
            rand_pos = torch.zeros(B, reduce_num, 1, dtype=torch.long, device=metric.device)
        else:
            if generator is not None:
                rand_pos = torch.randint(K, (B, reduce_num, 1), device=metric.device, generator=generator)
            else:
                rand_pos = torch.randint(K, (B, reduce_num, 1), device=metric.device)

        full = torch.arange(K, device=metric.device)
        matrix = full.unsqueeze(0).expand(K, K)
        mask = torch.eye(K, dtype=torch.bool, device=metric.device)
        src_table = matrix[~mask].view(K, K - 1)

        dst_idx = torch.gather(gathered_tensor, 2, rand_pos).reshape(B, -1).unsqueeze(-1)
        src_pos = src_table[rand_pos.squeeze(-1)]
        src_vals = torch.gather(gathered_tensor, 2, src_pos)
        src_idx = src_vals.reshape(B, reduce_num * dim_index).unsqueeze(-1)
        merge_idx = torch.arange(reduce_num, device=metric.device).repeat_interleave(dim_index).repeat(B, 1).unsqueeze(-1)

        # Update last_independent tracking
        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_idx.squeeze(-1), dst_idx.squeeze(-1)], dim=-1)
        else:
            independent_idx = unm_idx.squeeze(-1)

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        update_last_independent(independent_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            n, t1, c = x.shape
            src = gather(x, dim=-2, index=src_idx.expand(n, reduce_num * dim_index, c))
            dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num * num_token_window), c))
            dst = dst.scatter_reduce(-2, merge_idx.expand(n, reduce_num * dim_index, c), src, reduce=mode)
            x = torch.cat([unm, dst], dim=1)
            return x

        def mprune(x: torch.Tensor, mode="mean") -> torch.Tensor:
            n, t1, c = x.shape
            dst = gather(x, dim=-2, index=dst_idx.expand(n, reduce_num, c))
            unm = gather(x, dim=-2, index=unm_idx.expand(n, t1 - (reduce_num * num_token_window), c))
            x = torch.cat([unm, dst], dim=1)
            return x

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, tu, c = unm.shape
            src = gather(dst, dim=-2, index=merge_idx.expand(B, reduce_num * dim_index, c))

            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            out.scatter_(dim=-2, index=dst_idx.expand(B, reduce_num, c), src=dst)
            out.scatter_(dim=-2, index=unm_idx.expand(B, tu, c), src=unm)
            out.scatter_(dim=-2, index=src_idx.expand(B, reduce_num * dim_index, c), src=src)
            return out

    return merge, mprune, unmerge


def FIDM(
    metric: torch.Tensor,
    reduce_num: int = 0,
    window_size: Tuple[int, int] = (2, 2),
    no_rand: bool = False,
    generator: torch.Generator = None,
    tore_info: Dict = None
) -> Tuple[Callable, Callable, Callable]:
    """
    Fake Inattentive-prioritized Detail Merging (FIDM)

    Used in later denoising steps (detail phase). Uses frequency priority scores
    to identify "inattentive" tokens for merging.
    """
    B, N, C = metric.shape

    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        sx, sy = int(window_size[0]), int(window_size[1])

        h = int(math.sqrt(N))
        w = h
        assert h * w == N and h % sy == 0 and w % sx == 0
        hsy, wsx = h // sy, w // sx

        # Use frequency priority (last_independent) for dst selection
        use_li = (
            tore_info is not None and "states" in tore_info and
            tore_info["states"].get("last_independent") is not None and
            tore_info["states"]["last_independent"].shape[1] == N
        )

        if use_li:
            li = tore_info["states"]["last_independent"]
            li_grid = li.view(B, h, w)
            li_windows = li_grid.view(B, hsy, sy, wsx, sx).permute(0, 1, 3, 2, 4)
            li_flat_win = li_windows.reshape(B, hsy, wsx, sy * sx)
            dst_pos = li_flat_win.argmax(dim=-1, keepdim=True)
        else:
            if no_rand:
                dst_pos = torch.zeros(B, hsy, wsx, 1, device=metric.device, dtype=torch.int64)
            else:
                if generator is not None:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device, generator=generator)
                else:
                    dst_pos = torch.randint(sy * sx, (B, hsy, wsx, 1), device=metric.device)

        # Build sentinel buffer
        idx_buffer_view = torch.zeros(B, hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        neg_one = -torch.ones_like(dst_pos, dtype=idx_buffer_view.dtype)
        idx_buffer_view.scatter_(dim=3, index=dst_pos, src=neg_one)

        idx_buffer_view = idx_buffer_view.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(B, -1, 1).argsort(dim=1)

        del idx_buffer, idx_buffer_view

        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]
        b_idx = rand_idx[:, :num_dst, :]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between src and dst
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        reduce_num = min(a.shape[1], reduce_num)
        reduce_num = reduce_num // 16 * 16

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., reduce_num:, :]
        src_idx = edge_idx[..., :reduce_num, :]
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        a_idx_b = a_idx.expand(B, a_idx.shape[1], 1)
        b_idx_b = b_idx.expand(B, b_idx.shape[1], 1)

        dst_in_x_index = b_idx_b
        unm_in_x_index = gather(a_idx_b, dim=1, index=unm_idx)
        src_in_x_index = gather(a_idx_b, dim=1, index=src_idx)

        if tore_info and tore_info.get("args", {}).get("pseudo_merge", False):
            independent_idx = torch.cat([unm_in_x_index.squeeze(-1), dst_in_x_index.squeeze(-1)], dim=-1)
        else:
            independent_idx = unm_in_x_index.squeeze(-1)

        def update_last_independent(independent_indices: torch.Tensor):
            if tore_info["states"].get("last_independent") is None:
                tore_info["states"]["last_independent"] = torch.zeros(B, N, device=independent_indices.device, dtype=torch.int32)
            last_ind = tore_info["states"]["last_independent"]
            last_ind.add_(1)
            zeros_src = torch.zeros_like(independent_indices, dtype=last_ind.dtype)
            last_ind.scatter_(1, independent_indices, zeros_src)

        update_last_independent(independent_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            a, dst = split(x)
            n, t1, c = a.shape
            unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))
            src = gather(a, dim=-2, index=src_idx.expand(n, reduce_num, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, reduce_num, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)

        def mprune(x: torch.Tensor) -> torch.Tensor:
            a, dst = split(x)
            n, t1, c = a.shape
            unm = gather(a, dim=-2, index=unm_idx.expand(n, t1 - reduce_num, c))
            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape
            src = gather(dst, dim=-2, index=dst_idx.expand(B, reduce_num, c))

            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            out.scatter_(dim=-2, index=dst_in_x_index.expand(B, dst_in_x_index.shape[1], c), src=dst)
            out.scatter_(dim=-2, index=unm_in_x_index.expand(B, unm_in_x_index.shape[1], c), src=unm)
            out.scatter_(dim=-2, index=src_in_x_index.expand(B, src_in_x_index.shape[1], c), src=src)
            return out

    return merge, mprune, unmerge


def isinstance_str(x: object, cls_name: str):
    """Check if x has any class named cls_name in its ancestry."""
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def init_generator(device: torch.device, fallback: torch.Generator = None):
    """Fork the current default random generator for the given device."""
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def compute_ratio(tore_info: Dict[str, Any]) -> float:
    """Compute ratio_current using cosine-shaped descent."""
    args = tore_info.get("args", {})
    states = tore_info.get("states", {})

    ratio = args.get("ratio", 0.5)
    deviation = args.get("deviation", 0.2)
    step_current = states.get("step_current", 0)
    step_count = states.get("step_count", 1)
    layer_current = states.get("layer_current", 0)
    layer_count = states.get("layer_count", 1)
    protect_layers_frequency = args.get("protect_layers_frequency", None)

    def is_protected(idx, total, freq):
        if freq is None or freq < 0 or freq == 0:
            return False
        if idx % freq == 0:
            return True
        if idx == max(total - 1, 0):
            return True
        return False

    if is_protected(layer_current, layer_count, protect_layers_frequency):
        tore_info["states"]["last_independent"] = None
        return 0.0

    if step_count > 1:
        progress = step_current / (step_count - 1)
    else:
        progress = 0.0

    alpha = math.cos(progress * math.pi / 2)
    ratio_current = (ratio - deviation) + (2.0 * deviation) * alpha

    return float(ratio_current)


def compute_merge(x: torch.Tensor, tore_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    """Compute merge/unmerge functions based on current step."""
    w = int(math.sqrt(x.shape[1]))
    h = w
    assert w * h == x.shape[1], "Input must be square"

    ratio_current = compute_ratio(tore_info)
    tore_info["states"]["ratio_current"] = ratio_current

    reduce_num = int(x.shape[1] * ratio_current)
    if reduce_num <= 0:
        return do_nothing, do_nothing, do_nothing, do_nothing

    if tore_info["args"]["generator"] is None:
        tore_info["args"]["generator"] = init_generator(x.device)
    elif tore_info["args"]["generator"].device != x.device:
        tore_info["args"]["generator"] = init_generator(x.device, fallback=tore_info["args"]["generator"])

    use_rand = False if x.shape[0] % 2 == 1 else tore_info["args"]["use_rand"]

    # Choose strategy based on switch_step
    step_current = tore_info["states"].get("step_current", 0)
    switch_step = tore_info["args"].get("switch_step", 20)

    if step_current <= switch_step:
        # Structure phase: use SSM
        m, pm, u = SSM(
            metric=x, reduce_num=reduce_num,
            window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),
            no_rand=not use_rand, generator=tore_info["args"]["generator"],
            tore_info=tore_info
        )
    else:
        # Detail phase: use FIDM
        m, pm, u = FIDM(
            metric=x, reduce_num=reduce_num,
            window_size=(tore_info["args"]["sx"], tore_info["args"]["sy"]),
            no_rand=not use_rand, generator=tore_info["args"]["generator"],
            tore_info=tore_info
        )

    if tore_info["args"]["pseudo_merge"]:
        m_a, u_a = (pm, u) if tore_info["states"]["merge_attn"] else (do_nothing, do_nothing)
        m_m, u_m = (pm, u) if tore_info["states"]["merge_mlp"] else (do_nothing, do_nothing)
    else:
        m_a, u_a = (m, u) if tore_info["states"]["merge_attn"] else (do_nothing, do_nothing)
        m_m, u_m = (m, u) if tore_info["states"]["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_m, u_a, u_m


def remove_patch(model: torch.nn.Module):
    """Remove SDTM patch from a model."""
    model = model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tore_info"):
            if "hooks" in module._tore_info:
                for hook in module._tore_info["hooks"]:
                    hook.remove()
                module._tore_info["hooks"].clear()

        if module.__class__.__name__ == "SDTMBlock_PixArt":
            module.__class__ = module._parent

    return model


def make_SDTM_pipe_pixart(pipe_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Create patched PixArt pipeline class."""

    class PixArtPipeline_SDTM(pipe_class):
        _parent = pipe_class

        def __call__(self, *args, **kwargs):
            self._tore_info["states"]["step_count"] = kwargs.get('num_inference_steps', 20)
            self._tore_info["states"]["step_iter"] = list(range(self._tore_info["states"]["step_count"]))
            self._tore_info["states"]["last_independent"] = None
            output = super().__call__(*args, **kwargs)
            return output

    return PixArtPipeline_SDTM


def make_SDTM_model_pixart(model_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Create patched PixArt transformer model class."""

    class PixArtTransformer_SDTM(model_class):
        _parent = model_class

        def forward(self, *args, **kwargs):
            self._tore_info["states"]["layer_count"] = len(self.transformer_blocks)
            self._tore_info["states"]["step_current"] = self._tore_info["states"]["step_iter"].pop(0)
            self._tore_info["states"]["layer_iter"] = list(range(len(self.transformer_blocks)))

            # Phase selection based on step
            if self._tore_info["states"]["step_current"] <= self._tore_info["args"]["switch_step"]:
                self._tore_info["states"]["merge_attn"] = True
                self._tore_info["states"]["merge_mlp"] = True
            else:
                self._tore_info["states"]["merge_attn"] = False
                self._tore_info["states"]["merge_mlp"] = True

            output = super().forward(*args, **kwargs)
            return output

    return PixArtTransformer_SDTM


def make_SDTM_block_pixart(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Create patched BasicTransformerBlock class for PixArt.

    PixArt uses ada_norm_single with:
    - Self-attention (attn1): Merge before, unmerge after
    - Cross-attention (attn2): NO merge (different token spaces)
    - Feed-forward (ff): Merge before, unmerge after
    """

    class BasicTransformerBlock_SDTM(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            # Update layer counter
            self._tore_info["states"]["layer_current"] = self._tore_info["states"]["layer_iter"].pop(0)

            if cross_attention_kwargs is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                if cross_attention_kwargs.get("scale", None) is not None:
                    cross_attention_kwargs.pop("scale")

            batch_size = hidden_states.shape[0]

            # Check if this is a protected step
            args = self._tore_info.get("args", {})
            states = self._tore_info.get("states", {})
            protect_freq = args.get("protect_steps_frequency", None)
            step_current = states.get("step_current", 0)

            def _is_protected_step(idx, freq):
                if freq is None or freq < 0 or freq == 0:
                    return False
                return idx % freq == 0

            protected_step = _is_protected_step(step_current, protect_freq)

            # === Normalization for ada_norm_single ===
            if self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError(f"Unknown norm_type: {self.norm_type}")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

            if protected_step:
                # === PROTECTED STEP: No merging ===
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                if self.norm_type == "ada_norm_zero":
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.norm_type == "ada_norm_single":
                    attn_output = gate_msa * attn_output

                hidden_states = attn_output + hidden_states

                # Cross-attention (no merge)
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm_single":
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])

                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # Feed-forward
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                elif self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)

                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output

                hidden_states = ff_output + hidden_states

            else:
                # === NORMAL STEP: Apply merging ===
                # Compute merge functions
                m_a, m_m, u_a, u_m = compute_merge(norm_hidden_states, self._tore_info)

                # Self-attention with merge
                norm_hidden_states_merged = m_a(norm_hidden_states)
                attn_output = self.attn1(
                    norm_hidden_states_merged,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                if self.norm_type == "ada_norm_zero":
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.norm_type == "ada_norm_single":
                    attn_output = gate_msa * attn_output

                # Unmerge attention output
                attn_output = u_a(attn_output)
                hidden_states = attn_output + hidden_states

                # Cross-attention (NO merge - different token spaces)
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm_single":
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])

                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # Feed-forward with merge
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                elif self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                # Merge for FF
                norm_hidden_states_merged = m_m(norm_hidden_states)

                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states_merged, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states_merged)

                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output

                # Unmerge FF output
                ff_output = u_m(ff_output)
                hidden_states = ff_output + hidden_states

            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

    return BasicTransformerBlock_SDTM


def apply_SDTM_pixart(
    pipe: torch.nn.Module,
    ratio: float = 0.5,
    deviation: float = 0.2,
    switch_step: int = 10,
    use_rand: bool = True,
    sx: int = 2,
    sy: int = 2,
    a_s: float = 0.05,
    pseudo_merge: bool = False,
    protect_steps_frequency: int = None,
    protect_layers_frequency: int = None,
    merge_attn: bool = True,
    merge_mlp: bool = True,
):
    """
    Apply SDTM to a PixArt-Sigma pipeline.

    Args:
        pipe: PixArtSigmaPipeline instance
        ratio: Base merge ratio (0.0 to 1.0)
        deviation: Deviation for ratio scheduling
        switch_step: Step at which to switch from SSM to FIDM
        use_rand: Whether to use randomness in merging
        sx, sy: Window size for merging
        a_s: Frequency priority weight for SSM
        pseudo_merge: Use pseudo-merge (prune) instead of full merge
        protect_steps_frequency: Protect every Nth step from merging
        protect_layers_frequency: Protect every Nth layer from merging
        merge_attn: Whether to merge self-attention
        merge_mlp: Whether to merge feed-forward

    Returns:
        Patched pipeline
    """
    # Remove any existing patch
    remove_patch(pipe)

    # Patch pipeline
    make_pipe_fn = make_SDTM_pipe_pixart
    pipe.__class__ = make_pipe_fn(pipe.__class__)

    # Initialize tore_info
    pipe._tore_info = {
        "type": "SDTM_PixArt",
        "args": {
            "ratio": ratio,
            "deviation": deviation,
            "switch_step": switch_step,
            "use_rand": use_rand,
            "sx": sx,
            "sy": sy,
            "a_s": a_s,
            "pseudo_merge": pseudo_merge,
            "protect_steps_frequency": protect_steps_frequency,
            "protect_layers_frequency": protect_layers_frequency,
            "generator": None,
        },
        "states": {
            "last_independent": None,
            "ratio_current": ratio,
            "step_count": None,
            "step_iter": None,
            "step_current": None,
            "layer_count": None,
            "layer_iter": None,
            "layer_current": None,
            "merge_attn": merge_attn,
            "merge_mlp": merge_mlp,
        }
    }

    # Patch transformer model
    model = pipe.transformer
    make_model_fn = make_SDTM_model_pixart
    model.__class__ = make_model_fn(model.__class__)
    model._tore_info = pipe._tore_info

    # Patch transformer blocks
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_block_fn = make_SDTM_block_pixart
            module.__class__ = make_block_fn(module.__class__)
            module._tore_info = pipe._tore_info

    return pipe


# === Utility functions for benchmarking ===

def benchmark_pixart_sdtm(
    pipe,
    prompt: str = "A photo of a cat sitting on a windowsill",
    num_inference_steps: int = 20,
    num_warmup: int = 3,
    num_runs: int = 10,
    sdtm_kwargs: dict = None,
):
    """
    Benchmark PixArt-Sigma with and without SDTM.

    Args:
        pipe: PixArtSigmaPipeline instance
        prompt: Test prompt
        num_inference_steps: Number of denoising steps
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        sdtm_kwargs: Arguments for apply_SDTM_pixart

    Returns:
        dict with baseline_ms, sdtm_ms, speedup
    """
    import time

    device = pipe.device

    # Warmup
    for _ in range(num_warmup):
        _ = pipe(prompt, num_inference_steps=num_inference_steps, output_type="latent")

    # Baseline timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pipe(prompt, num_inference_steps=num_inference_steps, output_type="latent")
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - start) / num_runs * 1000

    # Apply SDTM
    if sdtm_kwargs is None:
        sdtm_kwargs = {"ratio": 0.5, "switch_step": 10}
    apply_SDTM_pixart(pipe, **sdtm_kwargs)

    # SDTM warmup
    for _ in range(num_warmup):
        _ = pipe(prompt, num_inference_steps=num_inference_steps, output_type="latent")

    # SDTM timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pipe(prompt, num_inference_steps=num_inference_steps, output_type="latent")
    torch.cuda.synchronize()
    sdtm_ms = (time.perf_counter() - start) / num_runs * 1000

    # Remove patch for fair comparison
    remove_patch(pipe)

    return {
        "baseline_ms": baseline_ms,
        "sdtm_ms": sdtm_ms,
        "speedup": baseline_ms / sdtm_ms,
    }
