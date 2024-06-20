"""
Class for interfacing with mongoDB storing smartthings state.

If you want to use this, you need to have pymongo installed:
pip install pymongo
"""
import datetime
import os
import time
from typing import Optional

import requests
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

from sage.base import BaseConfig


mongo_url = f"mongodb://{os.getenv('MONGODB_SERVER_URL')}"


class TvScheduleDb:
    """
    DB to store TV programming information.

    Programming can be added using the update_schedule function, and retrieved using whats_on.
    The notion of provider_string allows multiple TV sources to coexist in the DB (for example,
    cable TV and free samsung TV programming.)
    """

    db_name = "tv_schedule"

    def __init__(self):
        self.client = MongoClient(mongo_url)
        self.db = self.client[self.db_name]

    def _init_collection(self, provider_string: str) -> None:
        """
        Create the collection.

        If it already exists, do nothing.
        """
        try:
            self.db.create_collection(
                provider_string,
                timeseries={
                    "timeField": "end_ts",
                    "metaField": "channel_number",
                    "granularity": "minutes",
                },
            )
        except CollectionInvalid:
            pass

    def update_schedule(self, provider_string: str, schedule: list[dict]) -> None:
        """
        Update the shedule

        Args:
            provider_string
            schedule: list of shows, each show must have at least an end_ts timestamp
                and a channel_number param.
        """
        self._init_collection(provider_string)
        self.db[provider_string].insert_many(schedule, ordered=False)

    def whats_on(
        self,
        provider_string: str,
        now_utc: Optional[datetime.datetime] = None,
        n_per_channel: int = 1,
    ) -> list[list[dict]]:
        """
        What's scheduled for this provider?

        Args:
            provider_string
            now_utc: UTC time for which to check the schedule. If None, will default to the current time.
            n_per_channel: How many shows ahead to look. For each channel, will return
                the min(n_per_channel, max number available) next shows.

        Returns:
            outer list is channel, inner list is show
        """
        now_utc = now_utc or datetime.datetime.utcnow()

        aggregation_stages = [
            {"$sort": {"end_ts": -1}},
            {"$match": {"end_ts": {"$gt": now_utc}}},
            {
                "$group": {
                    "_id": "$channel_number",
                    "doc": {"$firstN": {"input": "$$ROOT", "n": n_per_channel}},
                }
            },
        ]
        result = list(self.db[provider_string].aggregate(aggregation_stages))

        return [r["doc"] for r in result]


class DeviceCapabilityDb:
    """
    Store device capabilities.

    Useful for automating interaction with the API. In the current implementation, these details
    are fused with online documentation in the DocManager class to support SAGE's device interaction.

    To update this when adding new devices, call the populate function.
    """

    def __init__(self, db_name: str):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]

    def get_device_capabilities(self, device_id: str) -> list[dict]:
        """
        Get all capabilities for a device.

        If a device has multiple components with the same capability, a separate
        item will appear for each in the output.
        """
        capabilities = list(
            self.db["device_capabilities"].find({"device_id": device_id})
        )
        capabilities.sort(key=lambda x: x["retrieval_ts"])

        return capabilities[-1]["capabilities"]

    def _init_collection(self):
        """
        Create the collection.

        If it already exists, do nothing.
        """
        try:
            self.db.create_collection(
                "device_capabilities",
            )
        except CollectionInvalid:
            pass

    def populate(self, config: BaseConfig) -> None:
        """
        Auto-populate the DB.

        You only need to run this when devices are updated.
        """
        self._init_collection()
        headers = {
            "Authorization": "Bearer %s" % config.global_config.smartthings_token
        }
        response = requests.get(
            "https://api.smartthings.com/v1/devices", headers=headers
        )
        devices = response.json()["items"]

        for device in devices:
            device_id = device["deviceId"]
            components = device["components"]
            device_capabilities = {
                "device_id": device_id,
                "retrieval_ts": datetime.datetime.utcnow(),
                "capabilities": [],
            }
            print("doing device", device_id)

            for component in components:
                component_id = component["id"]

                for capability in component["capabilities"]:
                    capability_id = capability["id"]
                    capability_version = capability["version"]
                    url = "https://api.smartthings.com/v1/capabilities/%s/%s" % (
                        capability_id,
                        capability_version,
                    )
                    cap_info = requests.get(url, headers=headers).json()
                    device_capabilities["capabilities"].append(
                        {
                            "component_id": component_id,
                            "capability_id": capability_id,
                            "capability_version": capability_version,
                            "capability_info": cap_info,
                        }
                    )
                    # avoid hitting rate limits
                    print("did capability", capability_id)
                    time.sleep(5)
            self.db["device_capabilities"].insert_one(device_capabilities)
