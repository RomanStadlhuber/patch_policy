import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import copy


class EMAModel(nn.Module):
    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        super(EMAModel, self).__init__()
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)

        self.register_buffer("decay", torch.tensor(0.0))
        self.register_buffer("optimization_step", torch.tensor(0, dtype=torch.int64))
        self._step = None

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(float(self.min_value), min(value, float(self.max_value)))

    @torch.no_grad()
    def step(self, new_model):
        # the step count and decay stay Python numbers: a GPU tensor as `alpha`
        # syncs the GPU once per parameter, which idled it for half of each step
        if self._step is None:
            # read once, after a resume has loaded the buffer
            self._step = int(self.optimization_step)
        decay = self.get_decay(self._step)
        self.decay.fill_(decay)

        copy_dst, copy_src, avg_dst, avg_src = [], [], [], []
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    copy_dst.append(ema_param)
                    copy_src.append(param.data.to(dtype=ema_param.dtype))
                else:
                    avg_dst.append(ema_param)
                    avg_src.append(param.data.to(dtype=ema_param.dtype))
        # one kernel launch per op for all parameters, instead of one per parameter
        if copy_dst:
            torch._foreach_copy_(copy_dst, copy_src)
        if avg_dst:
            torch._foreach_mul_(avg_dst, decay)
            torch._foreach_add_(avg_dst, avg_src, alpha=1 - decay)
        self._step += 1
        self.optimization_step.fill_(self._step)
