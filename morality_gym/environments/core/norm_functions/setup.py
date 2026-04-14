from typing import Optional, List, Dict, Any

from morality_gym.environments.core.norm_functions.outcome import make_fn as make_outcome_fn
from morality_gym.environments.core.norm_functions.utility import make_fn as make_utility_fn
from morality_gym.utils.common import copy_to_dict


def create_norm_fn(
        norm_type: str,
        func_name: Optional[str] = None,
        **kwargs
):
    if norm_type not in ["action", "causal", "outcome", "utility"]:
        raise ValueError(f"norm_type {norm_type} not recognized. Supported values = ['action', 'causal', "
                         f"'outcome', 'utility']")

    if norm_type == "action":
        raise NotImplementedError
    elif norm_type == "causal":
        raise NotImplementedError
    elif norm_type == "outcome":
        return make_outcome_fn(**kwargs)
    else:  # I.e. utility
        if func_name is None:
            raise ValueError(f"func_name must be specified for utility norm_type")
        return make_utility_fn(func_name, **kwargs)

        # return UTILITY_REGISTRY[func_name]


def create_multi_norm_funcs(
        norm_type: str,
        func_names: List[str] = None,
        func_kwargs: Dict[str, Dict[str, Any]] = None,
):
    if norm_type != "utility":
        raise ValueError(f"norm_type {norm_type} not recognized. Supported values = ['utility']")

    if func_names is None:
        raise ValueError(f"func_names must be specified for utility norm_type")

    if func_kwargs is None:
        func_kwargs = {}

    norm_fns = {}
    for name in func_names:
        if name in func_kwargs:
            kwargs = func_kwargs[name]
        else:
            kwargs = {}
        norm_fns[name] = create_norm_fn(norm_type, name, **kwargs)

    def _comb_norm_fn(_event):
        _d = {}
        for _name, _fn in norm_fns.items():
            _d[_name] = _fn(_event)
        # Flatten dict
        _flat_d = {}
        for _k, _v in _d.items():
            if not isinstance(_v, dict):
                raise TypeError(f"_v is not dict. type(_v) = {type(_v)}")
            copy_to_dict(from_dict=_v, to_dict=_flat_d)

        return _flat_d

    return _comb_norm_fn