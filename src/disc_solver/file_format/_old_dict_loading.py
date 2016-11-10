"""
Old registry for dictionaries (since it's been dropped from h5preserve
"""
from h5preserve import Registry, GroupContainer

dict_as_group_registry = Registry("Python: dict as group")


@dict_as_group_registry.dumper(dict, "dict", version=None)
def _dict_dumper(d):
    # pylint: disable=missing-docstring
    return GroupContainer(**d)


@dict_as_group_registry.loader("dict", version=None)
def _dict_loader(group):
    # pylint: disable=missing-docstring
    new_dict = {}
    new_dict.update(group)
    if group.attrs:
        new_dict["attrs"] = group.attrs
    return new_dict
