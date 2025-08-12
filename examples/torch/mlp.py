# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch.nn as nn
import allo
from allo.ir.types import float32, Fixed
import numpy as np

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
allo_t4p_dir = os.path.dirname(os.path.dirname(cur_dir))
llvm_build_dir = os.path.join(allo_t4p_dir, "externals/llvm-project/build")
os.environ["LLVM_BUILD_DIR"] = llvm_build_dir


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32)  # 8*16 * 32*16
        self.linear2 = torch.nn.Linear(32, 10)

    def forward(self, data):
        out = self.linear1(data)
        out = self.linear2(out)
        out = F.relu(out)
        return out


model = MLP()
model.eval()
example_inputs = [torch.rand(8, 16)]
golden = model(*example_inputs)
np_input = example_inputs[0].detach().numpy()

# llvm_mod = allo.frontend.from_pytorch(
#     model,
#     example_inputs=example_inputs,
#     verbose=True,
#     weights_as_args=True,
#     op_dtypes={
#         "inputs": float32,
#         # "linear1": [float32, Fixed(64, 30), float32],  # X, W, O for first linear
#         # "linear2": [float32, Fixed(64, 30), float32],  # X, W, O for second linear
#         "linear1": [float32, Fixed(16, 10), float32],  # X, W, O for first linear
#         "linear2": [float32, Fixed(16, 10), float32],  # X, W, O for second linear
#         "relu": float32,
#         "outputs": float32,  # optional outputs annotation
#     },
# )
# res = llvm_mod(np_input)
# torch.testing.assert_close(res, golden.detach().numpy(), atol=1e-2, rtol=1e-2)
# print("Passed!")


# VITIS HLS
mode = "sw_emu"
os.environ["XDEVICE"] = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
os.environ["XCL_EMULATION_MODE"] = mode

vitis_mod = allo.frontend.from_pytorch(
    model,
    example_inputs=example_inputs,
    target="vitis_hls",
    mode=mode,
    project="test_fixed.prj",
    verbose=False,
    weights_as_args=False,
    op_dtypes={
        "inputs": float32,
        "linear1": [float32, Fixed(16, 10), float32],  # X, W, O for first linear
        "linear2": [float32, Fixed(16, 10), float32],  # X, W, O for second linear
        "relu": float32,
        "outputs": float32,  # optional outputs annotation
    },
)

allo_out = np.zeros((8, 16), dtype=np.float32)
vitis_mod(np_input, allo_out)
np.testing.assert_allclose(allo_out, golden.detach().numpy(), rtol=1e-2, atol=1e-2)
print("Passed!")
