import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from typing import Union

import requests

from sage.smartthings.db import DeviceCapabilityDb


def to_ordered_dict_recur(input: Union[list, dict]) -> OrderedDict:
    """
    Convert JSON objects (i.e. object made of lists and dictionaries) to
    use OrderedDict with sorted key instead of regular dictionaries.

    This will make the ordering of the dictionary order consistent across multiple
    runs of the code, leading to a much higher cache hit rate.
    """

    if isinstance(input, dict):
        return OrderedDict(
            sorted([(k, to_ordered_dict_recur(v)) for k, v in input.items()])
        )
    elif isinstance(input, list):
        return [to_ordered_dict_recur(v) for v in input]
    else:
        return input


def dump_ordered_dict_recur(input: OrderedDict) -> list:
    """Convert OrderedDict to JSON-serializable object"""

    if isinstance(input, OrderedDict):
        return ["odict"] + [(k, dump_ordered_dict_recur(v)) for k, v in input.items()]
    elif isinstance(input, list):
        return [dump_ordered_dict_recur(v) for v in input]
    else:
        return input


def load_ordered_dict_recur(input: list) -> OrderedDict:
    """Convert output of dump_ordered_dict_recur to OrderedDict"""

    if isinstance(input, list):
        if input and input[0] == "odict":
            return OrderedDict([(k, load_ordered_dict_recur(v)) for k, v in input[1:]])
        else:
            return [load_ordered_dict_recur(v) for v in input]

    return input


class DocManager:
    """
    Class to manage all the different sources of device / capability documentation,
    with a focus on enabling agents.

    Capability docs can come from two sources:
    1. Online documentation, which tends to be pretty decent. These are preferred.
    2. If no online documentation exists, we can extract some info about the capability from the API.
        This is lacking in descriptions of what anything means, but at least has names for functions,
        arguments, and attributes.

    We are interested in maintaining the best possible description for each capability, as well
    as keeping track of which capabilities are relevant to which devices and components. This information
    is then used to construct prompts for the agent.

    In this class, the presence of the word "info" in a variable name indicates that it is some kind of
    structured format, while "docs" indicate a string.

    There is lots of sorting to be found within this class, the aim of this is to return consistent
    strings each time the class is called, leading to more cache hits and thus faster and cheaper testing.
    """

    def __init__(self, capability_db_name: str):
        self.db = DeviceCapabilityDb(db_name=capability_db_name)



    def init(self):
        with open(
            Path(os.getenv("SMARTHOME_ROOT")).joinpath(
                "external_api_docs/smartthings_capabilities.json"
            )
        ) as f:
            self.online_info = json.load(f)

        # beware that these might change over time a bit, as devices are connected/disconnected
        # also, we don't handle photobot till it is becomes more mature
        smartthings_token = os.getenv("SMARTTHINGS_API_TOKEN")

        headers = {"Authorization": "Bearer %s" % smartthings_token}
        response = requests.get(
            "https://api.smartthings.com/v1/devices", headers=headers
        )
        devices = response.json()["items"]
        self.default_devices = sorted([device["deviceId"] for device in devices])
        self.device_names = {device["deviceId"]: device["name"] for device in devices}

        self.device_capabilities = {
            d: sorted(
                self.db.get_device_capabilities(d), key=lambda x: x["component_id"]
            )
            for d in self.default_devices
        }
        self.device_capabilities = to_ordered_dict_recur(self.device_capabilities)
        self.capability_info_from_devices = {
            cap_info["capability_id"]: cap_info
            for cap_infos in self.device_capabilities.values()
            for cap_info in cap_infos
        }

    def to_json(self, json_cache_path: Path):
        """
        Save self to disk as JSON.
        """
        obj = {
            "default_devices": self.default_devices,
            "device_names": self.device_names,
            "device_capabilities": dump_ordered_dict_recur(self.device_capabilities),
            "capability_info_from_devices": self.capability_info_from_devices,
            "online_info": self.online_info,
            "capability_db_name": self.db.db.name,
        }
        with open(json_cache_path, "w") as f:
            json.dump(obj, f)

    @staticmethod
    def from_json(json_cache_path: Path):
        """
        Create DocManager from serialized JSON.
        """
        with open(json_cache_path, "r") as f:
            obj = json.load(f)
        dm = DocManager('tmp')
        dm.default_devices = obj["default_devices"]
        dm.device_names = obj["device_names"]
        dm.device_capabilities = load_ordered_dict_recur(obj["device_capabilities"])
        dm.capability_info_from_devices = obj["capability_info_from_devices"]
        dm.online_info = obj["online_info"]
        dm.db = DeviceCapabilityDb(db_name=obj["capability_db_name"])

        return dm

    @staticmethod
    def build_capability_docstring_from_online_info(cap: dict) -> str:
        """
        Build up a full capability docstring for capabilities which are nicely documented online.
        """

        return json.dumps(cap, sort_keys=True)

    @staticmethod
    def build_capability_docstring_without_online_info(cap: dict) -> str:
        """
        Build up a full capability docstring for capabilities which are NOT documented online.
        """

        return json.dumps(cap["capability_info"], sort_keys=True)

    def has_refresh_capability(self, device_id: str) -> bool:
        """
        Check if device has refresh capability

        If it does, you should use this capability before reading attributes, or they might
        not be properly updated when you read them.
        """

        return bool(
            [
                c
                for c in self.device_capabilities[device_id]
                if c["capability_id"] == "refresh"
            ]
        )

    def find_online_info(self, capability_id: str) -> Union[dict, None]:
        """
        Try to find nice online documentation for a capability.
        """
        info = [x for x in self.online_info if x["id"] == capability_id]

        if info:
            return info[0]

        return None

    def capability_docs(self, capability_id: str) -> dict:
        """
        Get best docs for a single capability.
        """

        info = self.find_online_info(capability_id)
        cap = self.capability_info_from_devices[capability_id]
        name = cap["capability_info"]["name"]

        if info:
            desc = info["i18n"]["description"]
            one_liner = f"{capability_id} ({desc})"
            remaining_docs = self.build_capability_docstring_from_online_info(info)

        else:
            one_liner = capability_id
            remaining_docs = self.build_capability_docstring_without_online_info(cap)

        return {
            "id": capability_id,
            "name": name,
            "one_liner": one_liner,
            "docs": remaining_docs,
        }

    def capability_summary_for_devices(
        self, devices: Optional[list[str]] = None
    ) -> tuple[str, str]:
        """
        Create a string which contains one-liner summaries for all capabilities of devices,
        and a second string which specifies which devices have which capabilities.
        """
        devices = devices or self.default_devices
        all_capabilities = set()
        device_strings = []

        for device_id in devices:
            device_capabilities = [
                c["capability_id"] for c in self.device_capabilities[device_id]
            ]

            all_capabilities.update(device_capabilities)
            device_strings.append(
                "%s (%s): %s"
                % (
                    device_id,
                    self.device_names[device_id],
                    ",".join(device_capabilities),
                )
            )

        capability_docs = OrderedDict(
            (cap_id, self.capability_docs(cap_id))
            for cap_id in sorted(all_capabilities)
        )
        one_liners_string = "\n".join(
            [doc["one_liner"] for doc in capability_docs.values()]
        )
        device_capability_string = "\n".join(device_strings)

        return one_liners_string, device_capability_string

    def device_capability_details(self, device_id: str, capability_id: str) -> str:
        """
        Create a string which gives full details about the capability for the device.

        The device is relevant because the capability may be available on multiple components.
        """
        device_name = self.device_names[device_id]
        capability_docs = self.capability_docs(capability_id)
        one_liner = capability_docs["one_liner"]
        details = capability_docs["docs"]
        components = ", ".join(
            [
                cap["component_id"]
                for cap in self.device_capabilities[device_id]
                if cap["capability_id"] == capability_id
            ]
        )

        return f"-Device: {device_name} ({device_id}) \n - The API documentation for the capability {one_liner} \n {details} \n - The components for this capability: \n {components}"
