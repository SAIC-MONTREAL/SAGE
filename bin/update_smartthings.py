"""
Script to update the device capabilities. This script should be called when devices are added/removed from the Smartthings app, and will update the databset with the available devices and their capabilities.

Example
python bin/update_smartthings.py
"""
from smartthings.db import DeviceCapabilityDb
from smartthings.docmanager import DocManager

# update device capabilities in db
db = DeviceCapabilityDb()
db.populate()

# update cache of docmanager
dm: DocManager = DocManager(update_cache=True)
