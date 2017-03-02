# -*- coding: utf-8 -*-
"""
Compatibility module containing code for older versions of python
"""

try:
    from os import fspath  # pylint: disable=unused-import,wrong-import-order
except ImportError:
    def fspath(path):
        """Return the string representation of the path.

        If str or bytes is passed in, it is returned unchanged.
        """
        if isinstance(path, (str, bytes)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            return path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            try:
                import pathlib
            except ImportError:
                pass
            else:
                if isinstance(path, pathlib.PurePath):
                    return str(path)

            raise TypeError(
                "expected str, bytes or os.PathLike object, not " +
                path_type.__name__
            )
