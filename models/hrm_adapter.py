import torch
import torch.nn.functional as F
from torch import nn
from .hrm import HRM


class HRMAdapter(nn.Module):
    """Adapter to plug HRM into the training loop API.

    Exposes ``initial_carry`` and a forward signature returning the
    ``(carry, loss, metrics, preds, all_finish)`` tuple expected by the
    trainer. A per-step loss function and optional backtrack signal can be
    forwarded to the underlying :class:`HRM` core.
    """

    def __init__(self, input_size, hi_hidden, lo_hidden, output_size, k, cfg):
        super().__init__()
        self.core = HRM(
            input_size,
            hi_hidden,
            lo_hidden,
            output_size,
            k,
            vanelayer_cfg=cfg.get("vanelayer"),
            s3star_cfg=cfg.get("s3star"),
        )

    def initial_carry(self, batch):
        """Return dummy carry to satisfy trainer interface."""
        return None

    def forward(
        self,
        carry,
        batch,
        return_keys=(),
        step_loss_fn=None,
        env_backtrack=None,
    ):
        """Forward pass matching the trainer protocol.

        Args:
            carry: Ignored dummy carry.
            batch: Mapping containing ``x`` of shape ``[T,B,D]`` and optional
                ``lengths`` ``[B]`` and ``labels`` ``[T,B]``.
            return_keys: Unused, kept for API compatibility.
            step_loss_fn: Optional callable ``(logits_t, labels, t)`` returning
                per-sample loss.
            env_backtrack: Optional tensor ``[T,B]`` providing backtrack density.

        Returns:
            Tuple ``(carry, loss, metrics, preds, all_finish)``.
        """
        x = batch["x"]
        lengths = batch.get("lengths")
        labels = batch.get("labels")

        logits = self.core(
            x,
            lengths=lengths,
            labels=labels,
            env_backtrack=env_backtrack,
            step_loss_fn=step_loss_fn,
        )

        if labels is not None:
            T, B, C = logits.shape
            if lengths is not None:
                mask = (
                    torch.arange(T, device=x.device)[:, None]
                    < lengths[None, :].to(x.device)
                )
                loss = F.cross_entropy(
                    logits[mask], labels[mask], reduction="mean"
                )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, C),
                    labels.reshape(-1),
                    reduction="mean",
                )
        else:
            loss = torch.tensor(0.0, device=x.device)

        metrics = {"loss": loss.detach()}
        preds = {}
        all_finish = True
        return carry, loss, metrics, preds, all_finish
