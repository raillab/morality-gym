import copy
import os
from typing import Dict, Any, Optional, List

from morality_gym.utils.common import copy_to_dict


# Compute kwargs from default kwargs, kwargs and overrides
def compute_kwargs(
        default_kwargs: Dict[str, Any],
        kwargs: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
        deepcopy_vals: bool = True
):
    out_kwargs = copy.deepcopy(default_kwargs)
    copy_to_dict(from_dict=kwargs, to_dict=out_kwargs, deepcopy_vals=deepcopy_vals)
    if overrides is not None:
        copy_to_dict(from_dict=overrides, to_dict=out_kwargs, deepcopy_vals=deepcopy_vals)
    return out_kwargs


