# -*- coding: utf-8 -*-
"""
Handle setting precision of floating point numbers.

Defines the floating point type to use `FLOAT_TYPE`, and conversion function
`float_type`.
"""

import numpy as _np

FLOAT_TYPE = _np.binary128
float_type = FLOAT_TYPE
FLOAT_TYPE_INFO = _np.finfo(FLOAT_TYPE)
