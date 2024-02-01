"""
Drop-in replacement for standard requests module which intercepts traffic to smartthings API.

This allows for automated testing. When testing is performed, this module is used instead of
the standard requests module, implementing all device state changes on an a dictionary representing
the state of all of the devices, rather than on the real devices themselves. The current state is
logged in a database so that it may be retrieved and tested for correctness. Each request is also
logged, but currently these logs are not used in validation logic.
"""
import os

import requests
from typing import Union
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

mongo_url = f"mongodb://{os.getenv('MONGODB_SERVER_URL')}"

test_id = ["-1"]


def set_test_id(new_test_id: str):
    """
    Sets the global test id.
    """
    test_id[0] = new_test_id


def replace_requests_with_fake_requests(code: str, test_id: str) -> str:
    """
    Replace requests with fake requests in a code string.
    """

    return code.replace(
        "import requests",
        'import testing.fake_requests as requests; requests.set_test_id("%s")'
        % test_id,
    )


class TestLogsDb:
    """
    Store test logs.

    Useful to use a DB here instead of a file because we have multiple
    processes simultaneously dumping stuff.

    We also use this to track device state, since it works well with multiple processes.
    """

    db_name = "test_logs"

    def __init__(self):
        self.client = MongoClient(mongo_url)
        self.db = self.client[self.db_name]
        self._init_collection()

    def _init_collection(self):
        """
        Initialize collection if it does not exist already
        """
        try:
            self.db.create_collection(
                "test_logs",
            )
            self.db["test_logs"].create_index("test_id")
            self.db.create_collection(
                "device_state",
            )
            self.db["device_state"].create_index("test_id")
        except CollectionInvalid:
            pass

    def add_test_log(self, test_id: str, log: dict):
        """
        Add a single test log to the db.
        """
        if test_id == "-1":
            raise ValueError("You forgot to set the log id")
        doc = {"test_id": test_id, "log": log}
        self.db["test_logs"].insert_one(doc)

    def get_test_logs(self, test_id: str) -> list[dict]:
        """
        Retrieve all logs for a given test.
        """
        if test_id == "-1":
            raise ValueError("You forgot to set the log id")

        return list(self.db["test_logs"].find({"test_id": test_id}))

    def set_device_state(self, test_id: str, device_state: dict):
        """
        Update state of all devices for a single test
        """
        self.db["device_state"].find_one_and_replace(
            {"test_id": test_id},
            {"test_id": test_id, "device_state": device_state},
            upsert=True,
        )

    def get_device_state(self, test_id: str) -> dict:
        """
        Get state of all devices for a single test.
        """
        return self.db["device_state"].find_one({"test_id": test_id})["device_state"]


db = TestLogsDb()


class FakeResponse:
    """
    Mimics the requests.Response object.
    """

    def __init__(self, json_content, status_code=200):
        self.json_content = json_content
        self.status_code = status_code

    def json(self):
        return self.json_content


def request(method: str, url: str, **kwargs) -> Union[FakeResponse, requests.Response]:
    """
    Used in place of requests.request

    Implements all interactions with the device state. Whenever a device state change is made,
    it is written into the database. If the request is not to the smartthings API, use the real
    requests module to complete it.
    """
    db.add_test_log(test_id[0], {"method": method, "url": url, "kwargs": kwargs})
    # only intercept requrests to smartthings API, let all others through

    if "api.smartthings.com" not in url:
        return requests.request(method, url, **kwargs)
    device_state = db.get_device_state(test_id[0])

    if method == "get":
        url_bits = url.split("/")
        # you can get the status of the entire device instead of a specific component
        # and the code written by the LLM does that, so we need to support it as well.
        # The URL is shorter when this is the case.
        device_id = url_bits[5]

        if len(url_bits) < 8:
            out = {"components": device_state[device_id]}

            return FakeResponse(out)

        component = url_bits[7]
        capability = url_bits[9]

        if device_id not in device_state:
            return FakeResponse(["no such device"])

        if component not in device_state[device_id]:
            return FakeResponse(["no such component"])

        if capability not in device_state[device_id][component]:
            return FakeResponse(["no such capability"])

        try:
            out = device_state[device_id][component][capability]
        except Exception as e:
            out = str(e)

        return FakeResponse(out)

    elif method == "post":
        device_id = url.split("/")[5]  # might be brittle
        update_state = False
        try:
            for com in kwargs["json"]["commands"]:
                component, capability, command, args = (
                    com["component"],
                    com["capability"],
                    com["command"],
                    com["arguments"],
                )
                # turn on/off

                if component not in device_state[device_id].keys():
                    raise ValueError(f"The component {component} is not supported.")

                if capability == "switch":
                    if command in ("on", "off"):
                        device_state[device_id][component][capability]["switch"][
                            "value"
                        ] = command
                        update_state = True
                    else:
                        raise ValueError("Invalid command: %s" % command)
                # change brightness
                elif capability == "switchLevel":
                    if command == "setLevel":
                        if isinstance(args[0], int):
                            device_state[device_id][component][capability]["level"][
                                "value"
                            ] = args[0]
                        else:
                            raise ValueError(
                                "The switchLevel command expects an integer for the level argument."
                            )

                    update_state = True

                elif capability == "colorTemperature":
                    if command == "setColorTemperature":
                        device_state[device_id][component][capability][
                            "colorTemperature"
                        ]["value"] = args[0]

                    update_state = True
                elif capability == "colorControl":
                    if command == "setHue":
                        if args[0] <= 100:
                            device_state[device_id][component][capability]["hue"][
                                "value"
                            ] = args[0]
                            update_state = True
                        else:
                            raise ValueError(
                                ["The hue value should be in percentage between 0-100"]
                            )
                    elif command == "setSaturation":
                        device_state[device_id][component][capability]["saturation"][
                            "value"
                        ] = args[0]
                        update_state = True

                    elif command == "setColor":
                        if args[0]["hue"] <= 100:
                            device_state[device_id][component][capability]["hue"][
                                "value"
                            ] = args[0]["hue"]
                            device_state[device_id][component][capability][
                                "saturation"
                            ]["value"] = args[0]["saturation"]
                            update_state = True
                        else:
                            raise ValueError(
                                ["The hue value should be in percentage between 0-100"]
                            )
                # change TV channel
                elif capability == "tvChannel":
                    if command == "setTvChannel":
                        # TODO check if the arguments is the right type
                        device_state[device_id][component][capability]["tvChannel"][
                            "value"
                        ] = args[0]
                        update_state = True
                    else:
                        raise ValueError("Invalid command or value: %s" % command)

                # change TV audio
                elif capability == "audioVolume":
                    if command == "setVolume":
                        device_state[device_id][component][capability]["volume"][
                            "value"
                        ] = args[0]
                        update_state = True
                    elif command == "volumeDown":
                        device_state[device_id][component][capability]["volume"][
                            "value"
                        ] -= 5
                        update_state = True
                    elif command == "volumeUp":
                        device_state[device_id][component][capability]["volume"][
                            "value"
                        ] += 5
                        update_state = True
                    else:
                        raise ValueError(
                            f"Invalid command: {command} for the capability {capability}"
                        )
                # change dishwasher mode
                elif capability == "refresh":
                    if command == "refresh":
                        pass
                elif capability == "samsungce.dishwasherWashingCourse":
                    if command == "setWashingCourse":
                        device_state[device_id][component][
                            "samsungce.dishwasherWashingCourse"
                        ]["washingCourse"]["value"] = args[0]
                        update_state = True

                elif capability == "execute":
                    if command == "start":
                        device_state[device_id][component]["dishwasherOperatingState"][
                            "machineState"
                        ]["value"] = "run"
                        update_state = True
                    else:
                        raise ValueError("Invalid command or value: %s" % command)
                elif capability == "custom.thermostatSetpointControl":
                    if command == "setSetpoint":
                        device_state[device_id][component]["temperatureMeasurement"][
                            "temperature"
                        ]["value"] = args[0]
                        update_state = True
                    else:
                        raise ValueError("Invalid command or value: %s" % command)
                elif capability == "dishwasherOperatingState":
                    if command == "setMachineState":
                        device_state[device_id][component]["dishwasherOperatingState"][
                            "machineState"
                        ]["value"] = args[0]
                        update_state = True
                    else:
                        raise ValueError("Invalid command: %s" % command)

                elif capability == "thermostatCoolingSetpoint":
                    if component == "main":
                        raise ValueError(
                            "The main component does not allow temperature reading or control"
                        )

                    if command == "setCoolingSetpoint":
                        device_state[device_id][component]["thermostatCoolingSetpoint"][
                            "coolingSetpoint"
                        ]["value"] = args[0]
                        device_state[device_id][component]["temperatureMeasurement"][
                            "temperature"
                        ]["value"] = args[0]
                        update_state = True
                else:
                    return FakeResponse(
                        ["capability not supported yet"], status_code=500
                    )
        except Exception as e:
            return FakeResponse(["An error occurred: " + str(e)], status_code=500)

        if update_state:
            db.set_device_state(test_id[0], device_state)

        return FakeResponse(["successfully executed command"])
    else:
        raise ValueError("Unsupported method %s" % method)


# mimic requests convenience functions
def get(url, params=None, **kwargs):
    return request("get", url, params=params, **kwargs)


def options(url, **kwargs):
    return request("options", url, **kwargs)


def head(url, **kwargs):
    return request("head", url, **kwargs)


def post(url, data=None, json=None, **kwargs):
    return request("post", url, data=data, json=json, **kwargs)
