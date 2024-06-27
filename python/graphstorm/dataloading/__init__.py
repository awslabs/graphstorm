"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Initial to import dataloading and dataset classes
"""
from .dataloading import GSgnnLinkPredictionDataLoader
from .dataloading import (GSgnnLPJointNegDataLoader,
                          GSgnnLPLocalUniformNegDataLoader,
                          GSgnnLPLocalJointNegDataLoader,
                          GSgnnLPInBatchJointNegDataLoader)
from .dataloading import GSgnnAllEtypeLPJointNegDataLoader
from .dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from .dataloading import GSgnnEdgeDataLoader
from .dataloading import GSgnnNodeDataLoader, GSgnnNodeSemiSupDataLoader
from .dataloading import (GSgnnLinkPredictionTestDataLoader,
                          GSgnnLinkPredictionJointTestDataLoader,
                          GSgnnLinkPredictionPredefinedTestDataLoader)
from .dataloading import (FastGSgnnLinkPredictionDataLoader,
                          FastGSgnnLPLocalJointNegDataLoader,
                          FastGSgnnLPJointNegDataLoader,
                          FastGSgnnLPLocalUniformNegDataLoader)
from .dataloading import (GSgnnEdgeDataLoaderBase,
                          GSgnnLinkPredictionDataLoaderBase,
                          GSgnnNodeDataLoaderBase)
from .dataloading import GSgnnMultiTaskDataLoader

from .dataset import GSgnnData

from .dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                          BUILTIN_LP_JOINT_NEG_SAMPLER,
                          BUILTIN_LP_INBATCH_JOINT_NEG_SAMPLER,
                          BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER,
                          BUILTIN_LP_LOCALJOINT_NEG_SAMPLER,
                          BUILTIN_LP_FIXED_NEG_SAMPLER)
from .dataloading import BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER
from .dataloading import BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER
from .dataloading import (BUILTIN_FAST_LP_UNIFORM_NEG_SAMPLER,
                          BUILTIN_FAST_LP_JOINT_NEG_SAMPLER,
                          BUILTIN_FAST_LP_LOCALUNIFORM_NEG_SAMPLER,
                          BUILTIN_FAST_LP_LOCALJOINT_NEG_SAMPLER)

from .dataloading import (DistillDataloaderGenerator,
                          DistillDataManager)

from .sampler import DistributedFileSampler
