"""
Wrapper to get version to avoid cyclic imports
"""
from ._version_ import get_versions
version = get_versions()['version']
