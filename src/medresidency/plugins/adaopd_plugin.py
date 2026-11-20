"""ms-swift plugin of AdaOPD (loaded with --external_plugins by the `adaopd` method, and by `opd` with monitor_gap).

ms-swift's GRPO trainer with a teacher server trains on-policy distillation: the advantage of a sampled token y is
log p_teacher(y | image) - log p_student(y). This plugin

1. scores every sampled response a second time with the same teacher and the same prompt without the image, which
   splits the advantage into visual evidence, log p_teacher(y | image) - log p_teacher(y | no image), and the language
   gap, log p_student(y) - log p_teacher(y | no image);
2. measures the language gap of the step on fresh tokens, over all ranks (medresidency/adaopd.py);
3. in control mode, replaces the schedule of ms-swift's CHORD weight mu (the weight of the SFT term) by
   mu = clip((EMA(gap) - tau) / max(gap_0 - tau, 1), 0, 1).

Logged per step: `lang_gap`, `visual_evidence` and, in control mode, `adaopd_mu`.

Environment (set by the launcher from the method configuration): MEDRES_ADAOPD_MODE (control | monitor: log only),
_TAU, _EMA, _GAP0, _TAGS (comma list: measure the gap only on rows of these teachers), _FRESH (0: all tokens),
_SIGNAL (lang | kl: the gap to the teacher with the image), and MEDRES_RESUME_CHECKPOINT for a resumed run.
"""
import copy
import os

import swift.rlhf_trainers.utils as su
import torch
import torch.distributed as dist
from swift.rlhf_trainers import grpo_trainer as gt
from swift.rlhf_trainers.gkd_helpers import (
    assemble_teacher_completion_logprobs,
    build_opsd_samples,
    build_teacher_requests,
    fetch_teacher_parsed_by_routing,
)

from medresidency.adaopd import GapController, fresh_flags

MODE = os.environ.get("MEDRES_ADAOPD_MODE", "control")
TAGS = set(filter(None, os.environ.get("MEDRES_ADAOPD_TAGS", "").split(",")))
FRESH = os.environ.get("MEDRES_ADAOPD_FRESH", "1") == "1"
SIGNAL = os.environ.get("MEDRES_ADAOPD_SIGNAL", "lang")
assert MODE in ("control", "monitor") and SIGNAL in ("lang", "kl"), (MODE, SIGNAL)
GAP = GapController(tau=float(os.environ.get("MEDRES_ADAOPD_TAU", "1.0")), ema=float(os.environ.get("MEDRES_ADAOPD_EMA", "0.9")),
                    gap0=float(os.environ["MEDRES_ADAOPD_GAP0"]) if os.environ.get("MEDRES_ADAOPD_GAP0") else None)


def _fresh_mask(mask, ids):
    """completion positions whose token does not complete a 4-gram already seen in the same response"""
    fresh = torch.zeros_like(mask)
    for b, comp in enumerate(ids):
        if comp and isinstance(comp[0], list):  # GRPOSample.response_token_ids is [[ids]] (one list per turn)
            comp = [x for turn in comp for x in turn]
        pos = mask[b].nonzero(as_tuple=True)[0]
        k = min(len(pos), len(comp))
        if k:
            fresh[b, pos[:k]] = torch.tensor(fresh_flags(comp[:k]), dtype=mask.dtype, device=mask.device)
    return fresh


def _without_image(s):
    c = copy.copy(s)
    c.messages = [dict(m) for m in s.messages]
    for m in c.messages:
        if m["role"] == "user":
            assert isinstance(m["content"], str), type(m["content"])
            m["content"] = m["content"].replace("<image>", "").strip()
    c.images, c.teacher_images, c.teacher_messages = [], None, None
    return c


def _assemble_teacher_api_logps(self, samples, batch_encoded_inputs):
    """GRPOTrainer._assemble_teacher_api_logps with a second, image-free teacher pass."""
    flat_samples, chunk_samples_list = [], []
    for batch_encoded in batch_encoded_inputs:
        chunk = batch_encoded.pop('_chunk_samples', [])
        chunk_samples_list.append(chunk)
        flat_samples.extend(chunk)
    build_opsd_samples(flat_samples)

    def fetch(samps):
        return fetch_teacher_parsed_by_routing(
            samps,
            build_teacher_requests(samps, self.template),
            self.teacher_configs,
            self.teacher_clients,
            gather_fn=self._gather_teacher_requests,
            infer_fn=lambda handle, client: self._infer_teacher_requests(handle, topk=0, teacher_client=client),
            scatter_fn=self._scatter_teacher_parsed,
            is_main_process=self.accelerator.is_main_process,
            tag_key=self.args.teacher_tag_key)

    parsed = fetch(flat_samples)
    parsed_n = fetch([_without_image(s) for s in flat_samples])
    offset, vsum, vcnt = 0, 0.0, 0.0
    for batch_encoded, chunk in zip(batch_encoded_inputs, chunk_samples_list):
        grpo_batch = batch_encoded['grpo_batch']
        mask, n = grpo_batch.completion_mask, len(chunk)
        ids = [s.response_token_ids for s in chunk]
        lp = assemble_teacher_completion_logprobs(parsed[offset:offset + n], mask, mask.device,
                                                  response_token_ids=ids).topk_logprobs[..., 0]
        ln = assemble_teacher_completion_logprobs(parsed_n[offset:offset + n], mask, mask.device,
                                                  response_token_ids=ids).topk_logprobs[..., 0]
        grpo_batch.teacher_per_token_logps = lp  # the OPD term itself is unchanged
        m = mask.bool()
        f = _fresh_mask(m, ids) if FRESH else m.clone()
        if TAGS:
            keep = torch.tensor([s.get_tag(self.args.teacher_tag_key) in TAGS for s in chunk], device=f.device)
            f = f & keep[:, None]
        grpo_batch.gap_ref_logps, grpo_batch.fresh_mask = (lp if SIGNAL == "kl" else ln), f
        vsum += float((lp - ln)[m].sum())
        vcnt += float(m.sum())
        offset += n
    self._metrics['train']['visual_evidence'].append(vsum / max(vcnt, 1.0))


_postprocess_batch_orig = gt.GRPOTrainer._postprocess_batch


def _postprocess_batch(self, samples, batch_encoded_inputs):
    """GRPOTrainer._postprocess_batch, then the language gap of the step and the new mu."""
    _postprocess_batch_orig(self, samples, batch_encoded_inputs)
    t = torch.zeros(2, device=self.accelerator.device)
    for be in batch_encoded_inputs:
        gb = be['grpo_batch']
        f = gb.fresh_mask
        t[0] += ((gb.old_per_token_logps - gb.gap_ref_logps) * f).sum().float()
        t[1] += f.sum().float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    if float(t[1]) == 0:  # no row of the measured teachers in this step: keep mu
        return
    g = float(t[0] / t[1].clamp(min=1.0))
    mode = 'train' if self.model.training else 'eval'
    self._metrics[mode]['lang_gap'].append(g)
    if MODE == "control":
        self._metrics[mode]['adaopd_mu'].append(GAP.update(g))


gt.GRPOTrainer._assemble_teacher_api_logps = _assemble_teacher_api_logps
gt.GRPOTrainer._postprocess_batch = _postprocess_batch
if MODE == "control":
    ckpt = os.environ.get("MEDRES_RESUME_CHECKPOINT", "").rstrip("/")
    log = os.path.join(os.path.dirname(ckpt), "logging.jsonl")
    if "checkpoint-" in ckpt and os.path.exists(log):
        n = GAP.restore(log, int(ckpt.rsplit("checkpoint-", 1)[1]))
        print(f"[adaopd] resumed from {ckpt}: {n} steps replayed, EMA {GAP.ema} first gap {GAP.gap0} mu {GAP.mu}")
    su.mu_schedule_function = lambda *args, **kwargs: GAP.mu  # read by ms-swift's compute_chord_loss
