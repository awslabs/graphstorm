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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.
"""

import os
import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Analyze profiling results.")
    # dataset and file arguments
    argparser.add_argument("--profile-path", type=str, required=True,
                           help="The path of the folder with profiling results.")
    args = argparser.parse_args()
    all_profiles = {}
    profile_files = os.listdir(args.profile_path)
    for profile_file in profile_files:
        profile_file = os.path.join(args.profile_path, profile_file)
        df = pd.read_csv(profile_file)
        arrs = {name: np.array(df[name]) for name in df}
        for name in arrs:
            if name not in all_profiles:
                all_profiles[name] = [arrs[name]]
            else:
                all_profiles[name].append(arrs[name])

    for name in all_profiles:
        assert len(all_profiles[name]) == len(profile_files)
        for i in range(len(all_profiles[name])):
            assert len(all_profiles[name][i]) == len(all_profiles[name][0])
        profile = np.stack(all_profiles[name], axis=1)
        print(f"There are {len(profile)} recordings for {name}.")
        avgs = []
        variances = []
        for i in range(len(profile)):
            avgs.append(np.average(profile[i]))
            variances.append(np.var(profile[i]))
        print(name, "average: {:.3f}, variance: {:.3f}".format(np.average(avgs),
                                                               np.average(variances)))
