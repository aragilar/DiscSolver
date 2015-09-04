"""
Wrapper around h5py and hdf5 to manage different versions of output files from
Disc Solver
"""

from h5schemaesqe import HDF5File, NamedtupleNamespace

from .._version import version as ds_version
from .version4 import version_4_schema

FILE_TYPE = "Disc Solver Solution File"
SCHEMAS = {
    "4": version_4_schema,
}

LATEST_VERSION = max(SCHEMAS)

LATEST_SCHEMA = SCHEMAS[LATEST_VERSION]
LATEST_NAMESPACE = NamedtupleNamespace(LATEST_SCHEMA)


def wrap_hdf5_file(file, version=None, new=False):
    """
    Wrap HDF5 file to use a schema
    """
    if version == "latest":
        version = LATEST_VERSION
    elif version is not None and version not in SCHEMAS:
        raise RuntimeError("Requested version does not exist")

    if new:
        if version is None:
            RuntimeError("Version required")
        write_metadata(file, version)
    else:
        file_version = get_metadata(file)
        if version is None:
            version = file_version
        elif version != file_version:
            raise RuntimeError("Requested version does not match file version")

    if version == LATEST_VERSION:
        schema = LATEST_SCHEMA
        namespace = LATEST_NAMESPACE
    else:
        schema = SCHEMAS[version]
        namespace = NamedtupleNamespace(schema)
    return HDF5File(schema, namespace, file=file)


def write_metadata(file, version):
    """
    Write metadata to file
    """
    file.attrs["version"] = version
    file.attrs["filetype"] = FILE_TYPE
    file.attrs["generator_version"] = ds_version


def get_metadata(file):
    """
    Get metadata out of HDF5 file
    """
    if file.attrs.get("filetype") != FILE_TYPE:
        raise RuntimeError("Incorrect file type")
    version = file.attrs.get("version")
    if version is None or version not in SCHEMAS:
        raise RuntimeError("Unknown version of file")
    return version
