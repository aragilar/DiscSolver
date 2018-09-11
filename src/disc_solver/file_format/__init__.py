"""
Defines common data structures and how they are to be written to files
"""

from h5preserve import new_registry_list

from . import _dumpers # noqa
from . import _loaders # noqa
from ._containers import ( # noqa
    Solution, SolutionInput, ConfigInput, Problems, InternalData,
    Run, InitialConditions,
)
from ._old_dict_loading import (
    dict_as_group_registry as _dict_as_group_registry
)
from ._utils import ds_registry as _ds_registry, get_fields

registries = new_registry_list(_ds_registry, _dict_as_group_registry)

CONFIG_FIELDS = get_fields(ConfigInput)
