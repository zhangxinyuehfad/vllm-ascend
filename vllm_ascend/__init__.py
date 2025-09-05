#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

# Supported SOC_VERSION codes
ASCEND_A2_SOC_VERSION = range(220, 226)
ASCEND_A3_SOC_VERSION = range(250, 256)
ASCEND_310P_SOC_VERSION = [202]

__ascend_soc_version__ = None  

def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    global __ascend_soc_version__
    import torch_npu  # type: ignore
    raw = torch_npu.npu.get_soc_version()
    __ascend_soc_version__ = (
    "A2"   if raw in ASCEND_A2_SOC_VERSION   else
    "A3"   if raw in ASCEND_A3_SOC_VERSION   else
    "310P" if raw in ASCEND_310P_SOC_VERSION else
    "UNDEFINED"
    )
    from .models import register_model
    register_model()

