class S3StarAudit:
    def __init__(self, tau_high_sigma=1.2, tau_low_sigma=0.6, cooldown_steps=3, max_interrupts=8, hysteresis=True):
        self.tau_hi = tau_high_sigma
        self.tau_lo = tau_low_sigma
        self.cooldown_steps = cooldown_steps
        self.max_interrupts = max_interrupts
        self.hysteresis = hysteresis
        self._cooldown = 0
        self._armed = True
        self._count = 0

    def reset(self):
        self._cooldown = 0
        self._armed = True
        self._count = 0

    def should_interrupt(self, batch_drift_mean, ema_mean, ema_std):
        if self._count >= self.max_interrupts or self._cooldown > 0:
            self._cooldown = max(0, self._cooldown - 1)
            return False
        # adaptive threshold relative to EMA mean/std
        z = (batch_drift_mean - ema_mean) / (ema_std + 1e-6)
        if self.hysteresis:
            if self._armed and z > self.tau_hi:
                self._armed = False
                self._cooldown = self.cooldown_steps
                self._count += 1
                return True
            if (not self._armed) and z < self.tau_lo:
                self._armed = True
            return False
        else:
            if z > self.tau_hi:
                self._cooldown = self.cooldown_steps
                self._count += 1
                return True
            return False
