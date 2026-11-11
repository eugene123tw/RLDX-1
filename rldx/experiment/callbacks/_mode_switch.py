"""Model, processor and RNG state a validation window borrows and hands back."""

from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
import torch

from rldx.runtime.heap import freeze_static_heap


# Four uniform Euler steps without RTC: the production policy's sampler.
_EULER_4 = {
    "num_inference_timesteps": 4,
    "denoising_timesteps": [0.0, 0.25, 0.5, 0.75],
    "sample_timestep_from_beta_dist": False,
}


@contextmanager
def validation_mode(model, processor):
    action_model = model.action_model
    model_training, processor_training = model.training, processor.training
    sampler = {name: getattr(action_model, name) for name in _EULER_4}
    rtc_timing = action_model.rtc_timing
    # Validation draws noise; training must see the same RNG stream with or without it.
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )
    model.eval()
    processor.eval()
    for name, value in _EULER_4.items():
        setattr(action_model, name, value)
    action_model.rtc_timing = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        action_model.rtc_timing = rtc_timing
        for name, value in sampler.items():
            setattr(action_model, name, value)
        if processor_training:
            processor.train()
        model.train(model_training)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3])
            torch.cuda.empty_cache()
        # Training froze its heap once; objects validation left behind join it.
        freeze_static_heap()
