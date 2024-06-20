"""
Script to update the device capabilities. This script should be called when devices are added/removed from the Smartthings app, and will update the databset with the available devices and their capabilities.

Example
python bin/update_smartthings.py
"""
import os
from pathlib import Path

from sage.smartthings.db import DeviceCapabilityDb
from sage.smartthings.docmanager import DocManager
from sage.base import BaseConfig, GlobalConfig

BaseConfig.global_config = GlobalConfig()
# update device capabilities in db
db = DeviceCapabilityDb(db_name="real_device_capabilities")
db.populate(BaseConfig(None))

json_cache_path = Path(os.getenv("SMARTHOME_ROOT")).joinpath(
    "external_api_docs/cached_real_docmanager.json"
)

# update cache of docmanager
dm = DocManager("real_device_capabilities")
dm.init()
dm.to_json(json_cache_path=json_cache_path)