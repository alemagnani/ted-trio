"""LIGER `run.py` with preprocessing replaced by existence checks (Instruments / Yelp TIGER).

The paper's Instruments/Yelp TIGER rows were trained with an on-box copy of LIGER's
`run.py` that was byte-identical except that `preprocessing()` (which can only download
Amazon-2014 Beauty/Toys/Sports) was replaced by assertions that the pre-staged files
exist (stage them with baselines/tiger/stage_expand_inputs.py). That copy was not
archived; this file is a faithful RECONSTRUCTION that achieves the same effect without
editing vendor code: it swaps `preprocessing` in LIGER's data_process module and then runs
LIGER's own, unmodified `run.py` as __main__ (Hydra CLI overrides pass straight through).

Copy this file into the LIGER clone root and run it from there, e.g.:

  python run_expand.py dataset=amazon dataset.name=Musical_Instruments \
      dataset.content_model=sentence-t5-xl seed=42 device_id=0 method=base \
      test_method=tiger experiment_id=close2_tiger logging.mode=disabled
  (dataset.name=Yelp for Yelp; seeds 42/43/44)
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import ID_generation.preprocessing.data_process as _dp  # noqa: E402


def _preprocessing_exists_only(config):
    name = config["name"]
    proc = config["processed_data_path"]
    features = "_".join(config["features_needed"])
    data_file = os.path.join(proc, f"{name}.txt")
    id2meta_file = os.path.join(proc, f"{name}_{features}_{config['prompt_format']}_id2meta.json")
    for f in (data_file, id2meta_file):
        assert os.path.exists(f), f"missing pre-staged file {f} (run stage_expand_inputs.py)"
    return data_file, id2meta_file, None  # item2attributes is unused by run.main


_dp.preprocessing = _preprocessing_exists_only

if __name__ == "__main__":
    # Execute LIGER's run.py itself as __main__: its `from ...data_process import
    # preprocessing` then binds the patched function, and Hydra resolves
    # config_path="configs" relative to run.py exactly as for `python run.py`.
    # (Importing `run` as a module instead makes Hydra look for a Python package
    # named `configs` and fail with MissingConfigException.)
    import runpy

    sys.argv[0] = os.path.join(os.getcwd(), "run.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
