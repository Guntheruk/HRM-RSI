import torch

class VaneLayer:
    """
    Computes a drift score s_t from per-step stats:
      - entropy jump of logits
      - loss delta (instant vs EMA)
      - backtrack density (optional: env provides)
    Maintains EMA + rolling std for adaptive thresholds.
    """
    def __init__(self, ema_beta=0.9, use_entropy=True, use_loss_delta=True, use_backtrack=True):
        self.beta = ema_beta
        self.use_entropy = use_entropy
        self.use_loss_delta = use_loss_delta
        self.use_backtrack = use_backtrack
        self._ema = None
        self._ema2 = None  # for var
        self._last_loss = None

    @staticmethod
    def _entropy(logits):
        # logits: [B, C]
        probs = logits.softmax(-1).clamp_min(1e-8)
        return -(probs * probs.log()).sum(-1)  # [B]

    def step(self, logits, loss=None, backtrack=None):
        """Return drift score per batch element and updated stats."""
        parts = []
        if self.use_entropy:
            ent = self._entropy(logits).detach()  # [B]
            parts.append(ent)
        if self.use_loss_delta and loss is not None:
            if self._last_loss is None:
                dloss = torch.zeros_like(loss.detach())
            else:
                dloss = (loss.detach() - self._last_loss)
            parts.append(dloss.abs())
            self._last_loss = loss.detach()
        if self.use_backtrack and backtrack is not None:
            parts.append(backtrack.float())

        s = torch.stack(parts, dim=-1).sum(-1) if parts else torch.zeros(logits.size(0), device=logits.device)  # [B]

        # update EMA + variance (per batch mean for thresholds)
        m = s.mean()
        if self._ema is None:
            self._ema, self._ema2 = m, m*m
        else:
            b = self.beta
            self._ema  = b*self._ema  + (1-b)*m
            self._ema2 = b*self._ema2 + (1-b)*(m*m)
        var = (self._ema2 - self._ema*self._ema).clamp_min(1e-9)
        std = var.sqrt()
        return s, self._ema.detach(), std.detach()
