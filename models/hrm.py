import torch
import torch.nn.functional as F
from torch import nn

from .s2_vanelayer import VaneLayer
from .s3star_audit import S3StarAudit


class HRM(nn.Module):
    """Minimal HRM with optional RSI extensions."""

    def __init__(self, input_size, hi_hidden, lo_hidden, output_size, k=4, *, vanelayer_cfg=None, s3star_cfg=None):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(input_size, hi_hidden)
        self.hi = nn.GRUCell(lo_hidden, hi_hidden)
        self.lo = nn.GRUCell(hi_hidden, lo_hidden)
        self.ln_hi = nn.LayerNorm(hi_hidden)
        self.ln_lo = nn.LayerNorm(lo_hidden)
        self.cond = nn.Linear(hi_hidden, hi_hidden)
        self.join = nn.Linear(hi_hidden + lo_hidden, lo_hidden)
        self.out = nn.Linear(lo_hidden, output_size)

        # RSI extensions
        self.use_vane = bool(vanelayer_cfg and vanelayer_cfg.get("enabled", True))
        self.use_s3star = bool(s3star_cfg and s3star_cfg.get("enabled", True))
        if self.use_vane:
            self.vane = VaneLayer(
                ema_beta=vanelayer_cfg.get("ema_beta", 0.9),
                use_entropy=vanelayer_cfg.get("use_entropy", True),
                use_loss_delta=vanelayer_cfg.get("use_loss_delta", True),
                use_backtrack=vanelayer_cfg.get("use_backtrack", True),
            )
        if self.use_s3star:
 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
            if not self.use_vane:
                raise ValueError("S3StarAudit requires VaneLayer; set vanelayer.enabled=True")
=======
 main
            self.s3star = S3StarAudit(
                tau_high_sigma=s3star_cfg.get("tau_high_sigma", 1.2),
                tau_low_sigma=s3star_cfg.get("tau_low_sigma", 0.6),
                cooldown_steps=s3star_cfg.get("cooldown_steps", 3),
                max_interrupts=s3star_cfg.get("max_interrupts_per_episode", 8),
 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
                hysteresis=s3star_cfg.get("hysteresis", True),
=======
                hysteresis=vanelayer_cfg.get("hysteresis", True) if vanelayer_cfg else True,
 main
            )
            self.mix_reset = vanelayer_cfg.get("mix_reset", 0.35) if vanelayer_cfg else 0.35
            self.force_hi_update = s3star_cfg.get("force_hi_update", True)

 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
    def forward(self, x, lengths=None, labels=None, env_backtrack=None, step_loss_fn=None, debug=False):
        if lengths is not None:
            T = int(lengths.max().item())
            x = x[:T]
        else:
            T = x.size(0)
        B = x.size(1)
=======
    def forward(self, x, lengths=None, labels=None, env_backtrack=None, step_loss_fn=None):
        T, B, _ = x.shape
 main
        h_hi = x.new_zeros(B, self.hi.hidden_size)
        h_lo = x.new_zeros(B, self.lo.hidden_size)
        outputs = []
        buf = []
        if self.use_s3star:
            self.s3star.reset()

        for t in range(T):
 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
            if len(buf) == self.k:
                lo_sum = torch.stack(buf, 0).mean(0)
                buf.clear()
                h_hi = self.hi(lo_sum, h_hi)
                h_hi = self.ln_hi(h_hi)

            xt = F.relu(self.enc(x[t]))

=======
            xt = F.relu(self.enc(x[t]))
            # periodic high-level update
            if t % self.k == 0:
                if buf:
                    lo_sum = torch.stack(buf, 0).mean(0)
                    buf.clear()
                else:
                    lo_sum = h_lo
                h_hi = self.hi(lo_sum, h_hi)
                h_hi = self.ln_hi(h_hi)

 main
            # hi→lo conditioning
            cond = torch.tanh(self.cond(h_hi))
            lo_in = xt + cond
            h_lo = self.lo(lo_in, h_lo)
            h_lo = self.ln_lo(h_lo)

            # readout
            z = torch.tanh(self.join(torch.cat([h_lo, h_hi], dim=-1)))
            logits = self.out(z)
            outputs.append(logits)
 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
            buf.append(h_lo.detach())

            # ----- RSI drift + interrupt -----
            if self.use_vane or self.use_s3star:
                mask = (lengths.to(x.device) > t) if lengths is not None else None
                loss_t = None
                if step_loss_fn is not None and labels is not None:
                    loss_t = step_loss_fn(logits, labels, t)
                    if mask is not None and loss_t.ndim > 0:
                        loss_t = loss_t[mask]
                backtrack_t = env_backtrack[t] if env_backtrack is not None else None
                if backtrack_t is not None and backtrack_t.ndim > 1:
                    backtrack_t = backtrack_t.squeeze(-1)
                if mask is not None:
                    vane_logits = logits[mask]
                    backtrack = backtrack_t[mask] if backtrack_t is not None else None
                else:
                    vane_logits = logits
                    backtrack = backtrack_t

                if self.use_vane:
                    s_valid, ema_mean, ema_std = self.vane.step(vane_logits, loss=loss_t, backtrack=backtrack)
                    if mask is not None:
                        s = logits.new_zeros(B)
                        s[mask] = s_valid
                    else:
                        s = s_valid
=======

            # ----- RSI drift + interrupt -----
            if self.use_vane or self.use_s3star:
                loss_t = None
                if step_loss_fn is not None and labels is not None:
                    loss_t = step_loss_fn(logits, labels, t)
                backtrack = env_backtrack[t] if env_backtrack is not None else None
                if self.use_vane:
                    s, ema_mean, ema_std = self.vane.step(logits, loss=loss_t, backtrack=backtrack)
 main
                else:
                    s, ema_mean, ema_std = None, None, None

                if self.use_s3star and ema_mean is not None:
 codex/add-rsi-vanelayer-and-s3-interrupt-to-hrm
                    mean_s = s[mask].mean() if mask is not None else s.mean()
                    z = (mean_s - ema_mean) / (ema_std + 1e-6)
                    fired = self.s3star.should_interrupt(mean_s, ema_mean, ema_std)
                    if debug:
                        print(
                            f"t={t} s={float(mean_s):.2f} ema={float(ema_mean):.2f} "
                            f"std={float(ema_std):.2f} z={float(z):.2f} fired={fired}"
                        )
                    if fired:
=======
                    if self.s3star.should_interrupt(s.mean(), ema_mean, ema_std):
 main
                        if self.force_hi_update:
                            h_hi = self.hi(h_lo.detach(), h_hi)
                            h_hi = self.ln_hi(h_hi)
                        with torch.no_grad():
                            prior = torch.tanh(self.cond(h_hi))
                            h_lo = (1 - self.mix_reset) * h_lo + self.mix_reset * prior
                        buf.clear()
            # ---------------------------------

        return torch.stack(outputs, 0)
