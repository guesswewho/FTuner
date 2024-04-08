import tvm

import torch
import transformers

import logging
import os

logger = logging.getLogger(__name__)

from ..shared.fixture import ModelFixture


# PyTorch
torch.backends.cudnn.benchmark = True
# check the availability of cuDNN
if "CuDNN" in torch.__config__.show():
    logger.info("cuDNN is enabled in pytorch")
else:
    logger.warning("cuDNN is not enabled in pytorch!")


class PyTorchGPTFixture(ModelFixture):
    def __init__(self, B, T):
        self.input_data_torch = torch.randint(30000, (B, T), dtype=torch.int64)
        self.input_data_np = self.input_data_torch.detach().numpy()

        model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
        self.model = model.eval()
        self.scripted_model = torch.jit.trace(model, [self.input_data_torch], strict=False)

    @property
    def name(self):
        return 'GPT'

    @property
    def input_name(self):
        return 'input_ids'
