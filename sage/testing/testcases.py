"""Testcases"""
import json
import os
import re
import sys
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from io import StringIO
from typing import Any
from typing import Callable

import numpy as np
from dateutil import parser
from langchain.schema.messages import HumanMessage
from langchain.utilities import OpenWeatherMapAPIWrapper

from sage.misc_tools.gcloud_auth import gcloud_authenticate
from sage.misc_tools.google_suite import GoogleCalendarListEventsTool
from sage.testing.fake_requests import db
from sage.testing.testing_utils import listen
from sage.testing.testing_utils import manual_gmail_search
from sage.testing.testing_utils import pretty_print_email
from sage.testing.testing_utils import setup


tv_id = "8e20883f-c444-4edf-86bf-64e74c1d70e2"
frame_tv_id = "415228e4-456c-44b1-8602-70f98c67fba8"
tvs = [tv_id, frame_tv_id]
nightstand_light = "22e11820-64fa-4bed-ad8f-f98ef66de298"
fireplace_light = "571f0327-6507-4674-b9cf-fe68f7d9522d"
dining_table_light = "c4b4a179-6f75-48ed-a6b3-95080be6afc6"
tv_light = "363f714d-e4fe-4052-a78c-f8c97542e709"
lights = [nightstand_light, fireplace_light, dining_table_light, tv_light]
fridge = "51f02f33-4b43-11bf-2a6d-e7b5cf5be0ee"
dishwasher = "6c645f61-0b82-235f-e476-8a6afc0e73dc"
switch = "2214d9f0-4404-4bf6-a0d0-aaee20458d66"


TEST_REGISTER = {
    "device_resolution": set(),
    "personalization": set(),
    "persistence": set(),
    "intent_resolution": set(),
    "command_chaining": set(),
    "simple": set(),
    "test_set": set(),
    "human_interaction": set(),
    "google": set(),
}


def get_test_challenges(test_name: str):
    """
    Given a test name returns a list of challenges the test poses
    """
    challeges = []

    for key, setvals in TEST_REGISTER.items():
        if test_name in setvals:
            challeges.append(key)

    return challeges


def register(names: list[str]):
    "Add the desired testcases to the test register"

    def wrapper(f):
        for name in names:
            TEST_REGISTER[name].add(f)

        return f

    return wrapper


def get_tests(
    test_class_list: list[Callable], combination="intersection"
) -> list[Callable]:
    """Selects the testcases to run"""
    tests = TEST_REGISTER[test_class_list[0]]

    for test_class in test_class_list[1:]:
        if combination == "union":
            tests = tests.union(TEST_REGISTER[test_class])

        if combination == "intersection":
            tests = tests.intersection(TEST_REGISTER[test_class])

    return list(tests)


def check_status_on(device_state: dict[str, Any], device_id_list: list[str]) -> bool:
    """
    Returns True if any of the listed devices are on. Else False
    """

    for device in device_id_list:
        if device_state[device]["main"]["switch"]["switch"]["value"] == "on":
            return True

    return False


@register(["device_resolution"])
def turn_on_tv(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "on"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: turn on the TV"
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "TV was not turned on"


@register(["device_resolution"])
def get_current_channel(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"

    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "42"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: what channel is playing on the TV?"
    answer = coordinator.execute(user_command)
    assert (
        "42" in answer["output"].lower()
    ), f"The proper channel number is missing. The response : {answer['output']}"


@register(["device_resolution"])
def turn_on_bedside_light(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: turn on the light by the bed"
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The bedroom light is not turned on."


@register(["device_resolution"])
def check_dishwasher_state(device_state, config):
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "prewash"

    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "run"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "prewash"
    device_state[dishwasher]["main"]["switch"]["switch"]["value"] = "on"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Dmitriy: what is the current phase of the dish washing cycle?"
    result = coordinator.execute(user_command)

    assert (
        "prewash" in result["output"].lower()
    ), f"The correct state was not reported. The response : {result['output']}"


@register(["device_resolution", "intent_resolution"])
def dim_fireplace_lamp(device_state, config):
    device_state[fireplace_light]["main"]["switchLevel"]["level"]["value"] = 90

    device_state[fireplace_light]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    user_command = (
        "Abhisek : dim the lights by the fire place to a third of the current value"
    )

    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[fireplace_light]["main"]["switchLevel"]["level"]["value"] == 30
    ), "The fireplace light was not set to the appropriate brightness"


@register(["device_resolution"])
def lower_tv_volume(device_state, config):
    device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] = 50
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : Lower the volume of the TV by the light"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] < 50
    ), "The volume of the TV was not lowered."


@register(["device_resolution"])
def check_freezer_temp(device_state, config):
    device_state[fridge]["freezer"]["temperatureMeasurement"]["temperature"][
        "value"
    ] = -17
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek : what is the current temperature of the freezer?"
    result = coordinator.execute(command)

    assert (
        "-17" in result["output"].lower()
    ), "Correct freezer temperature (-17) was not reported"


@register(["device_resolution"])
def is_main_tv_on(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : is the TV by the credenza on?"
    result = coordinator.execute(command)
    assert (
        "no" in result["output"].lower() or "off" in result["output"].lower()
    ), f"The response was supposed to say that the TV is off. The response : {result['output']}"


# trick test


def microwave_some_popcorn(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: microwave some popcorn"
    result = coordinator.execute(user_command)

    ways_to_say_no = ["I'm sorry", "cannot", "can't"]
    assert len([x for x in ways_to_say_no if x in result]) > 0


def turn_up_the_heat(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: crank up the heat in here! I'm freezing my hands off."
    result = coordinator.execute(user_command)
    ways_to_say_no = ["I'm sorry", "cannot", "can't"]
    assert len([x for x in ways_to_say_no if x in result]) > 0


#### bit vague
@register(["device_resolution", "intent_resolution"])
def play_something_for_kids(device_state, config):
    device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek : play something for the kids on the TV by the plant"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        str(device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "3"
    ), "TV channel number was not set to 3 (PBS Kids)"


@register(["device_resolution", "intent_resolution"])
def put_on_something_funny(device_state, config):
    device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Dmitriy : play something funny on the TV by the plant"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert str(
        device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"]
    ) in [
        "3",
        "2",
    ], "TV channel number was not set to 3 (Sesame Street) or 2 (Big Bang Theory)"


@register(["device_resolution", "intent_resolution"])
def setup_lights_for_dinner(device_state, config):
    device_state[dining_table_light]["main"]["switch"]["switch"]["value"] = "off"
    device_state[dining_table_light]["main"]["switchLevel"]["level"]["value"] = 0

    for light in [fireplace_light, nightstand_light, tv_light]:
        device_state[light]["main"]["switch"]["switch"]["value"] = "off"

    command = "Dmitriy : set up lights for dinner"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(command)
    new_device_state = db.get_device_state(test_id)

    # NOTE: check if other lights were not turned on.
    # Otherwise the testcase will pass if the LLM turn on other lights by mistake

    for device_id in lights:
        if device_id == dining_table_light:
            assert (
                new_device_state[dining_table_light]["main"]["switch"]["switch"][
                    "value"
                ]
                == "on"
                or new_device_state[dining_table_light]["main"]["switchLevel"]["level"][
                    "value"
                ]
                > 0
            ), "Dining table light was not turned on"
        else:
            assert (
                new_device_state[device_id] == device_state[device_id]
            ), f"{device_id} state changed without asking."


@register(["device_resolution", "intent_resolution", "personalization"])
def make_it_cozy_in_bedroom(device_state, config):
    device_state[nightstand_light]["main"]["colorTemperature"]["colorTemperature"][
        "value"
    ] = 5000
    device_state[nightstand_light]["main"]["switch"]["switch"]["value"] = "off"
    device_state[nightstand_light]["main"]["switchLevel"]["level"]["value"] = 100

    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek : Set the lights in the bedroom to a cozy setting"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    # cozy means brightness < 50 and warm
    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "Nightstand light was not turned on"
    assert (
        device_state[nightstand_light]["main"]["switchLevel"]["level"]["value"] < 50
        or device_state[nightstand_light]["main"]["colorTemperature"][
            "colorTemperature"
        ]["value"]
        < 3000
    ), "Nightstand light was not dimmed below 50\% or 3000K"


@register(["device_resolution", "intent_resolution", "personalization"])
def match_the_lights_to_weather(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = (
        "Dmitriy : set the light over the dining table to match my weather preference"
    )
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    weather_report = OpenWeatherMapAPIWrapper().run("quebec city, Canada")
    cloud_cover = int(weather_report.split("\n")[-1].split(":")[1].strip()[0:-1])
    assert device_state[dining_table_light]["main"]["switch"]["switch"]["value"] == "on"

    if cloud_cover < 50:
        # sunny

        assert (
            13
            <= device_state[dining_table_light]["main"]["colorControl"]["hue"]["value"]
            <= 17
        ), "Dining table light was not set to yellow even though it is sunny"
    else:
        # cloudy
        assert (
            device_state[dining_table_light]["main"]["colorControl"]["hue"]["value"]
            > 65
            and device_state[dining_table_light]["main"]["colorControl"]["hue"]["value"]
            < 68
        ), "Dining table light was not set to blue even though it is cloudy"


@register(["device_resolution", "command_chaining", "intent_resolution"])
def turn_off_all_lights(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Dmitriy : darken the entire house"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    for device_id in lights:
        assert (
            device_state[device_id]["main"]["switch"]["switch"]["value"] == "off"
        ), f"Light {device_id} was not turned off"


@register(["device_resolution", "command_chaining"])
def turn_off_light_dim(device_state, config):
    for device_id in [nightstand_light, fireplace_light]:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
        device_state[device_id]["main"]["switchLevel"]["level"]["value"] = 20

    for device_id in [tv_light, dining_table_light]:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
        device_state[device_id]["main"]["switchLevel"]["level"]["value"] = 90

    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Dmitriy : turn off all the lights that are dim"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    for device_id in [nightstand_light, fireplace_light]:
        assert (
            device_state[device_id]["main"]["switch"]["switch"]["value"] == "off"
        ), f"Light {device_id} was turned off even though it was not dim"

    for device_id in [tv_light, dining_table_light]:
        assert (
            device_state[device_id]["main"]["switch"]["switch"]["value"] == "on"
        ), f"Light {device_id} was not turned on off even though it was dim"


@register(["device_resolution", "intent_resolution"])
def getting_call_tv_too_loud(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] = 75
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : I am getting a call, adjust the volume of the TV"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "TV was not turned on"
    assert (
        device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] < 75
    ), "TV volume was not turned down"


@register(["intent_resolution"])
def dishes_dirty_set_appropriate_mode(device_state, config):
    device_state[dishwasher]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : Dishes are too greasy, set an appropriate mode in the dishwasher."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert device_state[dishwasher]["main"]["samsungce.dishwasherWashingCourse"][
        "washingCourse"
    ]["value"].lower() in (
        "heavy",
        "intensive",
    ), "Dishwasher was not set to 'heavy' cycle"


@register(["personalization", "device_resolution", "intent_resolution"])
def put_something_informative(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : Put something informative on the tv by the plant."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] == "on"
    assert str(
        device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"]
    ) in [
        "9",
        "4",
        "1",
        "10",
    ], "TV channel number was not set to the news, National Geographic, or Jeopardy"


@register(
    ["device_resolution", "personalization", "command_chaining", "intent_resolution"]
)
def change_light_colors_conditioned_on_favourite_team(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek : Change the lights of the house to represent my favourite hockey team. Use the lights by the TV, the dining room and the fireplace."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    #  HUE:  0 = red, 120 green, 240 blue
    light_vals = [(0, 100), (66.67, 100), (0, 0)]  # red, blue, white

    for device_id in [fireplace_light, dining_table_light, tv_light]:
        assert (
            device_state[device_id]["main"]["switch"]["switch"]["value"] == "on"
        ), f"Device {device_id} was not turned on."
        hue = device_state[device_id]["main"]["colorControl"]["hue"]["value"]

        sat = device_state[device_id]["main"]["colorControl"]["saturation"]["value"]

        for light_val in light_vals:

            if (light_val[0] - 1 < hue < light_val[0] + 1) and sat == light_val[1]:
                # NOTE modifying light_vals inside loop maybe be dangerous
                light_vals.remove(light_val)
                print(f"device {device_id} ({hue}, {sat}) match {light_val})")

                continue

    assert (
        len(light_vals) == 0
    ), "Light hues were not set to correct colors, red, blue, white"


@register(["device_resolution", "personalization", "intent_resolution"])
def set_bedroom_light_for_sleeping(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : I am going to sleep. Change the bedroom light accordingly."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "Nightstand light was not turned on"

    assert (
        device_state[nightstand_light]["main"]["colorControl"]["hue"]["value"] == 0
    ), f"Nightstand light was not set to red (hue 0), given hue: {device_state[nightstand_light]['main']['colorControl']['hue']['value']}"


@register(["device_resolution", "command_chaining"])
def turn_off_tvs_turn_on_fireplace_light(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"

    for device_id in tvs:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Dmitriy : turn off all the TVs and switch on the fireplace light"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    for device_id in tvs:
        assert (
            device_state[device_id]["main"]["switch"]["switch"]["value"] == "off"
        ), "TV {device_id} was not turned off"
    assert (
        device_state[fireplace_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "Fireplace light was not turned on"
    assert not check_status_on(
        device_state, [nightstand_light, dining_table_light, tv_light]
    ), "Other lights, that weren't supposed to be turned on, have been turned on."


##persistent commands
@register(["persistence"])
def frige_door_light(device_state, config):
    device_state[fridge]["main"]["contactSensor"]["contact"]["value"] = "closed"
    device_state[dining_table_light]["main"]["switch"]["switch"]["value"] = "off"
    device_state[dining_table_light]["main"]["switchLevel"]["level"]["value"] = 0

    test_id, coordinator = setup(device_state, config.coordinator_config)

    command = (
        "abhisek : turn on the light in the dining room when the I open the fridge"
    )
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    device_state[fridge]["main"]["contactSensor"]["contact"]["value"] = "open"
    db.set_device_state(test_id, device_state)

    trigger_command = listen(config)
    coordinator.execute(trigger_command[0] + " : " + trigger_command[1])
    device_state = db.get_device_state(test_id)
    assert (
        device_state[dining_table_light]["main"]["switch"]["switch"]["value"] == "on"
        or device_state[dining_table_light]["main"]["switchLevel"]["level"]["value"] > 0
    ), "The dining room table was not turned on"


@register(["device_resolution", "persistence"])
def turn_on_bedroom_light_dishwasherstate(device_state, config):
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "prewash"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "run"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "prewash"

    device_state[nightstand_light]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "abhisek : turn on light by the nightstand when the dishwasher is done"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "off"
    ), "The nightstand light should have stayed off. Not triggered yet."

    # dishwasher stops
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "finish"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "stop"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "finish"
    db.set_device_state(test_id, device_state)
    trigger_command = listen(config)
    coordinator.execute(trigger_command[0] + " : " + trigger_command[1])
    device_state = db.get_device_state(test_id)

    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The night stand light should have been turned on."

    assert not check_status_on(
        device_state, [tv_light, fireplace_light, dining_table_light]
    ), "Other lights, that weren't supposed to be turned, have been turned on."


@register(["device_resolution", "persistence"])
def increase_volume_with_dishwasher_on(device_state, config):
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "unknown"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "stop"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "none"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingPercentage"][
        "dishwasherOperatingPercentage"
    ]["value"] = 0

    device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] = 50

    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "abhisek : increase the volume of the TV by the credenza whenever the dishwasher is running"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] == 50
    ), "The audio volume should not have been changed yet!"

    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "spin"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "run"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "spin"
    db.set_device_state(test_id, device_state)

    trigger_command = listen(config)
    coordinator.execute(trigger_command[0] + ":" + trigger_command[1])
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] > 50
    ), "The audio volume should have been increased."


@register(["device_resolution", "personalization", "intent_resolution"])
def put_the_game_on_amal(device_state, config):
    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    command = "amal : put the game on the tv by the credenza"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."

    assert (
        str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "7"
    ), "The channel was not properly set."


@register(["device_resolution", "personalization", "intent_resolution"])
def put_the_game_on_abhisek(device_state, config):
    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"

    command = "abhisek : put the game on the tv by the credenza"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."

    assert str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) in [
        "6",
        "7",
    ], "The channel was not properly set."


@register(["device_resolution", "intent_resolution", "personalization"])
def long_day_unwind(device_state, config):
    device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "dmitriy : its been a long, tiring day. Can you play something light and entertaining on the TV by the plant"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert str(
        device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"]
    ) in [
        "2",
        "10",
    ], "The channel was not properly set."
    assert (
        device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."


@register(["device_resolution", "command_chaining", "intent_resolution"])
def switch_off_everything(device_state, config):
    for device in [fireplace_light, dining_table_light, frame_tv_id, tv_id]:
        device_state[device]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek : Heading off to work. Turn off all the non essential devices."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    for device in [frame_tv_id, tv_id, fireplace_light, dining_table_light]:
        assert (
            device_state[device]["main"]["switch"]["switch"]["value"] == "off"
        ), "The device was not turned off."


@register(["device_resolution", "intent_resolution"])
def room_too_bright(device_state, config):
    device_state[dining_table_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[dining_table_light]["main"]["switchLevel"]["level"]["value"] = 100
    test_id, coordinator = setup(device_state, config.coordinator_config)

    command = "Amal : It is too bright in the dining room."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        device_state[dining_table_light]["main"]["switch"]["switch"]["value"] == "off"
        or device_state[dining_table_light]["main"]["switchLevel"]["level"]["value"]
        < 100
    ), "The dining table light has not been set to the proper brightness."


@register(["device_resolution", "intent_resolution"])
def set_christmassy_lights_by_fireplace(device_state, config):
    device_state[fireplace_light]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    command = "Dmitriy : Setup a christmassy mood by the fireplace."
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[fireplace_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "the fireplace light was not turned on"

    assert (
        device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] == 0
        or 33
        < device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"]
        < 34
        or 66
        < device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"]
        < 67
    ), "The light setting does not look christmassy enough to me."


def dishwasher_notification_with_tv_in_the_mix(device_state, config):
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "prewash"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "run"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "prewash"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    command = "Dmitriy : Notify me when the dishwasher is done, but if I am watching TV, only notify me when I turn the TV off"
    coordinator.execute(command)

    device_state = db.get_device_state(test_id)
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["dishwasherJobState"][
        "value"
    ] = "unknown"
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "stop"
    device_state[dishwasher]["main"]["custom.dishwasherOperatingProgress"][
        "dishwasherOperatingProgress"
    ]["value"] = "none"
    db.set_device_state(test_id, device_state)
    user_command = listen(config, timeout=10)
    assert user_command is None, "There shouldnt be any trigger commands at this stage."
    device_state = db.get_device_state(test_id)
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    db.set_device_state(test_id, device_state)
    trigger_command = listen(config, timeout=10)

    res2 = coordinator.execute(trigger_command[0] + ":" + trigger_command[1])
    completion_phrases = ["done", "did", "notify"]
    assert_flag = False

    for phrase in completion_phrases:
        if phrase in res2["output"].lower():
            assert_flag = True
    assert (
        assert_flag
    ), f"I presume the task did not succeed. The response:  {res2['output']}"


######### demos for video ###########


@register(["device_resolution", "persistence"])
def tv_off_lights_on_persist(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"

    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = (
        "Amal: when the TV  by the credenza turns off turn on the light by the bed"
    )
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)
    # lights should not have been messed with yet

    for light in lights:
        assert (
            device_state[light]["main"]["switch"]["switch"]["value"] == "off"
        ), "The light have stayed off at this point of the execution"

    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    db.set_device_state(test_id, device_state)
    trigger_command = listen(config)
    coordinator.execute(trigger_command[0] + ":" + trigger_command[1])
    device_state = db.get_device_state(test_id)
    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The bedroom light should have been turned on."
    assert not check_status_on(
        device_state, [tv_light, dining_table_light, fireplace_light]
    ), "Other lights, that weren't supposed to be turned on, have been turned on."


@register(["device_resolution", "intent_resolution", "command_chaining"])
def switch_to_other_tv(device_state, config):
    # setup initial conditions
    cur_channel = "7"
    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = cur_channel
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    user_command = "Amal: move this channel to the other TV and turn this one off"
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)
    # assert on/off
    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "off"
        and device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The switch values of the TVs are not proper."

    # assert the channel
    assert (
        str(device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"])
        == cur_channel
    ), "The proper channel has not been set."


@register(
    ["device_resolution", "personalization", "command_chaining", "intent_resolution"]
)
def put_the_game_on_dim_the_lights(device_state, config):
    # setup initial conditions
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[tv_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[tv_light]["main"]["switchLevel"]["level"]["value"] = 100
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = (
        "Amal: put the game on the tv by the credenza and dim the lights by the TV"
    )

    coordinator.execute(user_command)

    device_state = db.get_device_state(test_id)
    # assert TV on
    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."
    # assert channel
    assert (
        str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "7"
    ), "The proper channel has not been set."

    # assert ligth dimming
    assert (
        device_state[tv_light]["main"]["switchLevel"]["level"]["value"] < 100
    ), "The brightness level has not been reduced."


@register(["personalization"])
def memory_weather_test(device_state, config):
    whitelist = set("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Abhisek : I am going to visit my mom. Should I bring an umbrella?"
    result = coordinator.execute(user_command)
    weather_report = OpenWeatherMapAPIWrapper().run("quebec city, Canada")
    rain_val = re.findall(r"\{(.*?)\}", weather_report.split("\n")[-3])[0]
    ans = "".join(filter(whitelist.__contains__, result["output"].lower()))

    if len(rain_val) == 0:
        assert "no" in ans, "Failed to judge the need for umbrella"
    else:
        assert "yes" in ans, "Failed to judge the need for umbrella"


def fail():
    raise ValueError("This test likes to fail")


@register(["personalization", "google"])
def create_calendar_event(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(
        "Amal : create a new event in my calendar - build a spaceship tomorrow at 4pm"
    )

    # allow time for google calendar to update so the sent email will be retrievable
    time.sleep(15)

    gcal_api_resource = gcloud_authenticate(app="calendar")
    timenow = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    gcaltool = GoogleCalendarListEventsTool(api_resource=gcal_api_resource)
    events = gcaltool._run(timeMin=timenow)

    spaceship_found = False
    right_date = False

    tomorrow = ""

    for event in events:
        if (
            "spaceship" in event["summary"].lower()
            or "spaceship" in event["description"]
        ):
            spaceship_found = True

            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            if tomorrow in event["start"]:
                right_date = True

                # delete the event to not influence future testing now that we know we found it
                gcal_api_resource.events().delete(
                    calendarId="primary", eventId=event["id"]
                ).execute()

                break

    assert spaceship_found, "No event found contained 'spaceship'"
    assert (
        right_date
    ), f"The event containing 'spaceship' did not have the date of tomorrow ({tomorrow})"


@register(["personalization", "intent_resolution", "google"])
def watch_with_mom(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    result = coordinator.execute(
        "Amal : What am I scheduled to do with my mom on Saturday?"
    )
    assert (
        "casablanca".lower() in result["output"].lower()
    ), "Response did not contain 'casablanca'"


@register(["personalization", "intent_resolution", "google"])
def next_week_look_like(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    result = coordinator.execute("Dmitriy : what does next week look like?")

    # judge output using another llm
    messages = [
        HumanMessage(
            content=f'Does the following sentence summarize upcoming events in a calendar including watching Casablanca, and cooking dinner? (Respond \'yes\' or \'no\'): "{result["output"]}"'
        )
    ]
    ai_message = config.evaluator_llm(messages)
    print(
        f"Does the AI evaluator think this test succeeded? AI message: {ai_message.content}"
    )
    assert (
        "yes" in ai_message.content.lower()
    ), "The AI evaluator did not think that the system summarized the expected upcoming events in the calendar."


@register(["personalization", "command_chaining", "google"])
def summarize_and_email(device_state, config):
    gmail_api_resource = gcloud_authenticate(app="gmail")

    # most recently recieved email
    last_email = manual_gmail_search(
        api_resource=gmail_api_resource, query="to:me", maxResults=1
    )[0]

    # Main LLM test:
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(
        "Amal : Summarise the last email I received. Send the summary to Adam Sigal via email."
    )

    # allow time for gmail to update so the sent email will be retrievable
    time.sleep(15)

    # Manually verify the LLM's sent email and its contents:
    sent_email = manual_gmail_search(
        api_resource=gmail_api_resource,
        query=f"in:sent after:{datetime.now().strftime('%Y-%m-%d')}",
        maxResults=1,
    )
    assert (
        len(sent_email) > 0
    ), "No emails were sent today, i.e. the system did not send an email."
    # there should only be one result
    sent_email = sent_email[0]

    datetime_object = parser.parse(sent_email["date"])
    diff = datetime.now(timezone.utc) - datetime_object
    assert (
        diff.seconds <= 300
    ), "No email was sent within the last 5 minutes; the system did not send an email as was desired."

    # After the above asserts, we can be confident we found the right email,
    # now we can delete the message to avoid influencing future test runs
    gmail_api_resource.users().messages().delete(
        userId="me", id=sent_email["id"]
    ).execute()

    # judge the summary using another llm
    eval_prompt = pretty_print_email([last_email, sent_email])
    eval_prompt += (
        "\nDoes email 2 appear to be a summary of email 1? (Respond 'yes' or 'no')"
    )
    print(eval_prompt, "\n")

    messages = [HumanMessage(content=eval_prompt)]
    ai_message = config.evaluator_llm(messages)
    print(
        f"Does the AI evaluator think this test succeeded? AI message: {ai_message.content}"
    )
    assert (
        "yes" in ai_message.content.lower()
    ), "The AI evaluator did not think that the system summarized the most recently received email in the email which was sent."
    assert (
        sent_email["recipient"] == "sigalsamsung@gmail.com"
    ), "Recipient of email was not 'sigalsamsung@gmail.com'" 


@register(["personalization", "command_chaining", "google"])
def schedule_dad(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(
        "Amal : If my father is not scheduled to visit next week, compose an email draft inviting him to come build a spaceship.",
    )

    # allow time for gmail to update so the sent email will be retrievable
    time.sleep(15)

    # Manually verify the desired email and its contents:
    gmail_api_resource = gcloud_authenticate(app="gmail")
    res = manual_gmail_search(
        api_resource=gmail_api_resource,
        query=f"in:drafts after:{datetime.now().strftime('%Y-%m-%d')}",
        maxResults=1,
    )

    assert (
        len(res) > 0
    ), "No email drafts were created today, i.e. the system did not create an email draft."
    # there should only be one result
    res = res[0]

    assert (
        "spaceship".lower() in res["subject"].lower()
        or "spaceship".lower() in res["snippet"].lower()
    ), "'spaceship' was not in either the email's subject nor body."
    assert (
        res["recipient"] == "dad@example.com"
    ), "Recipient of email was not 'dad@example.com'"

    datetime_object = parser.parse(res["date"])
    diff = datetime.now(timezone.utc) - datetime_object
    assert (
        diff.seconds <= 300
    ), "No email draft was composed within the last 5 minutes; the system did not compose a draft as was desired."

    # delete the message to avoid influencing future test runs, since we are now sure we found the right email
    gmail_api_resource.users().messages().delete(userId="me", id=res["id"]).execute()


@register(["device_resolution", "personalization", "command_chaining", "google"])
def mother_natgeo(device_state, config):
    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal : if my mother is scheduled to visit this week, turn on national geographic on the tv by the credenza"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)

    assert (
        str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "9"
    ), "TV is not set to the proper channel"

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."


@register(["device_resolution", "command_chaining"])
def same_light_as_tv(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"

    test_id, coordinator = setup(device_state, config.coordinator_config)

    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"

    for light in lights:
        device_state[light]["main"]["switch"]["switch"]["value"] = "off"

    db.set_device_state(test_id, device_state)

    command = "Abhisek : Turn on the light"
    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[tv_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV Light has not been turned on."
    assert not check_status_on(
        device_state, [nightstand_light, fireplace_light, dining_table_light]
    ), "Other lights, that weren't supposed to be turned on, have been turned on."


@register(
    [
        "device_resolution",
        "personalization",
        "command_chaining",
        "intent_resolution",
        "google"
    ]
)
def what_did_i_miss(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)
    res3 = coordinator.execute("Abhisek : what did I miss?")

    # manually get last 2 emails
    gmail_api_resource = gcloud_authenticate(app="gmail")
    messages_info = manual_gmail_search(gmail_api_resource, "to:me", maxResults=2)

    # judge output using another llm
    eval_prompt = "Here are the user's two most recent emails:"
    eval_prompt += pretty_print_email(messages_info)
    eval_prompt += f'\nDoes the following sentence appear to summarize the two most recently received emails presented above? (Respond \'yes\' or \'no\'): "{res3["output"]}"'

    messages = [HumanMessage(content=eval_prompt)]
    ai_message = config.evaluator_llm(messages)
    print(
        f"Does the AI evaluator think this test succeeded? AI message: {ai_message.content}"
    )
    assert (
        "yes" in ai_message.content.lower()
    ), "The AI evaluator did not think that the system summarized the two most recently received emails."


@register(
    [
        "device_resolution",
        "personalization",
        "command_chaining",
        "intent_resolution",
        "google"
    ]
)
def mothers_email(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)

    res3 = coordinator.execute("Abhisek : what is my mother's email address?")
    assert (
        "mom@example.com" in res3["output"]
    ), f"The email was not retrieved. The response :{res3['output']}"


@register(["simple"])
def turn_on_frame_tv(device_state, config):
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "42"

    test_id, coordinator = setup(device_state, config.coordinator_config)
    user_command = "Amal: turn on the Frame TV to Channel 5"
    coordinator.execute(user_command)
    device_state = db.get_device_state(test_id)
    assert (
        device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "TV not switched on"
    assert (
        str(device_state[frame_tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "5"
    ), "TV not set to right Channel"


@register(["simple"])
def read_all_lights(device_state, config):
    # IC: turn all lights on

    for light in lights:
        device_state[light]["main"]["switch"]["switch"]["value"] = "on"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    result = coordinator.execute("Amal : Are all the lights on?")

    # judge output using another llm
    messages = [
        HumanMessage(
            content=f'Does the following sentence acknowledge that all of the lights are on? (Respond \'yes\' or \'no\'): "{result["output"]}"'
        )
    ]
    ai_message = config.evaluator_llm(messages)
    print(
        f"Does the AI evaluator think this test succeeded? AI message: {ai_message.content}"
    )
    assert (
        "yes" in ai_message.content.lower()
    ), "The AI evaluator did not think that the system acknowledged that all of the lights are on."


@register(["simple"])
def is_fridge_open(device_state, config):
    # set door open
    device_state[fridge]["main"]["contactSensor"]["contact"]["value"] = "open"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    result = coordinator.execute("Amal : Is the fridge door open?")

    # judge output using another llm
    messages = [
        HumanMessage(
            content=f'Does the following sentence acknowledge that the fridge door is open? (Respond \'yes\' or \'no\'): "{result["output"]}"'
        )
    ]
    ai_message = config.evaluator_llm(messages)
    print(
        f"Does the AI evaluator think this test succeeded? AI message: {ai_message.content}"
    )
    assert (
        "yes" in ai_message.content.lower()
    ), "The AI evaluator did not think that the system acknowledged that the fridge door is open."


@register(["simple"])
def start_dishwasher(device_state, config):
    # set IC
    device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
        "value"
    ] = "stop"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute("Abhisek : Start the dishwasher")
    device_state = db.get_device_state(test_id)
    assert (
        device_state[dishwasher]["main"]["dishwasherOperatingState"]["machineState"][
            "value"
        ]
        == "run"
    ), "Device is not turned on"


@register(["simple"])
def control_fridge_temp(device_state, config):
    # set IC
    device_state[fridge]["cooler"]["temperatureMeasurement"]["temperature"]["value"] = 2

    test_id, coordinator = setup(device_state, config.coordinator_config)
    coordinator.execute(
        "Abhisek : Change the fridge internal temperature to 5 degrees Celsius"
    )
    device_state = db.get_device_state(test_id)
    assert (
        device_state[fridge]["cooler"]["temperatureMeasurement"]["temperature"]["value"]
        == 5
    ), "Fridge temperature value is not correct"


@register(["simple"])
def turn_on_all_lights(device_state, config):
    # set IC

    for id in lights:
        device_state[id]["main"]["switch"]["switch"]["value"] = "off"

    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute("Abhisek : Turn on all the lights.")
    device_state = db.get_device_state(test_id)

    for id in lights:
        assert (
            device_state[id]["main"]["switch"]["switch"]["value"] == "on"
        ), f"Device {id} is not turned on"


@register(["simple"])
def query_fridge_temp(device_state, config):
    # IC
    device_state[fridge]["cooler"]["temperatureMeasurement"]["temperature"][
        "value"
    ] = "11"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    result = coordinator.execute("Abhisek : What is the fridge temperature?")
    device_state = db.get_device_state(test_id)
    assert "11" in result["output"], "the temperature value is wrong"


### testcases with human interaction


def get_interaction_question(run_logpath):
    with open(os.path.join(run_logpath, "experiment.log"), "r") as f:
        log_text = f.read()
    # find human_interaction_tool string
    human_interaction_tool_str = re.search("Action: human_interaction_tool", log_text)
    relevant_text = log_text[human_interaction_tool_str.span()[1] :]
    human_interaction_action_inp = re.search(r"\{.*\}", relevant_text)
    span = human_interaction_action_inp.span()
    interaction_action_inp = relevant_text[span[0] : span[1]]
    llm_utterance = json.loads(interaction_action_inp)

    return llm_utterance["query"]


@register(["device_resolution", "intent_resolution", "human_interaction"])
def set_bedroom_light_fav_color(device_state, config):
    device_state[nightstand_light]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    # hijack the IO stream
    f1 = sys.stdin
    human_interaction_resp = "My favorite color is blue"
    f = StringIO(f"Amal : {human_interaction_resp}")
    sys.stdin = f

    coordinator.execute("Amal : Set bedroom light to my favourite color")
    # changing things back in order
    f.close()
    sys.stdin = f1
    # read the data
    llm_query = get_interaction_question(
        config.coordinator_config.global_config.logpath
    )
    messages = [
        HumanMessage(
            content=f"Check if the following question answer pair makes sense. (Respond 'yes' or 'no'). \n\n Question: {llm_query} \n Answer: {human_interaction_resp}"
        )
    ]
    llm_resp = config.evaluator_llm(messages)
    assert (
        "yes" in llm_resp.content.lower()
    ), f"This question: {llm_query} to this answer {human_interaction_resp} does not makes sense."

    device_state = db.get_device_state(test_id)
    assert (
        device_state[nightstand_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The nightstand light has not been turned on first."
    assert (
        device_state[nightstand_light]["main"]["colorControl"]["hue"]["value"] > 65.2
        and device_state[nightstand_light]["main"]["colorControl"]["hue"]["value"] < 68
    ), "Nightstand  light was not set to my favourite color, blue."


@register(
    ["device_resolution", "personalization", "intent_resolution", "human_interaction"]
)
def put_the_game_on_dmitriy(device_state, config):
    device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"] = "0"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    command = "dmitriy : put the game on the tv by the credenza"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    # hijack the IO stream
    f1 = sys.stdin
    human_interaction_resp = "The game of basketball"
    f = StringIO(f"Dmitriy : {human_interaction_resp}")
    sys.stdin = f

    coordinator.execute(command)
    device_state = db.get_device_state(test_id)
    f.close()
    sys.stdin = f1
    llm_query = get_interaction_question(
        config.coordinator_config.global_config.logpath
    )
    messages = [
        HumanMessage(
            content=f"Check if the following question answer pair makes sense. (Respond 'yes' or 'no'). \n\n Question: {llm_query} \n Answer: {human_interaction_resp}"
        )
    ]
    llm_resp = config.evaluator_llm(messages)
    assert (
        "yes" in llm_resp.content.lower()
    ), f"This question: {llm_query} to this answer {human_interaction_resp} does not makes sense."

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "on"
    ), "The TV has not been turned on."

    assert (
        str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "7"
    ), "The channel was not properly set."


@register(["device_resolution", "intent_resolution", "human_interaction"])
def redonkulous_living_room(device_state, config):
    device_state[fireplace_light]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    # hijack the IO stream
    f1 = sys.stdin
    human_interaction_resp = "just turn the light by the fireplace to red."
    f = StringIO(f"Dmitriy : {human_interaction_resp}")
    sys.stdin = f

    coordinator.execute("Dmitriy : Make the living room look redonkulous!")
    # changing things back in order
    f.close()
    sys.stdin = f1
    device_state = db.get_device_state(test_id)

    llm_query = get_interaction_question(
        config.coordinator_config.global_config.logpath
    )
    messages = [
        HumanMessage(
            content=f"Check if the following question answer pair makes sense. (Respond 'yes' or 'no'). \n\n Question: {llm_query} \n Answer: {human_interaction_resp}"
        )
    ]
    llm_resp = config.evaluator_llm(messages)
    assert (
        "yes" in llm_resp.content.lower()
    ), f"This question: {llm_query} to this answer {human_interaction_resp} does not makes sense."
    assert (
        device_state[fireplace_light]["main"]["switch"]["switch"]["value"] == "on"
    ), "The fireplace light has not been turned on first."
    assert (
        device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] < 1.3
    ), "The living room is not turned red."


@register(["device_resolution", "intent_resolution", "human_interaction"])
def turn_it_off(device_state, config):
    # turn everything on

    for light in lights:
        device_state[light]["main"]["switch"]["switch"]["value"] = "on"

    for tv in tvs:
        device_state[tv]["main"]["switch"]["switch"]["value"] = "on"

    test_id, coordinator = setup(device_state, config.coordinator_config)

    # hijack the IO stream
    f1 = sys.stdin
    human_interaction_resp = "turn off the TV by the plant"
    f = StringIO(f"Abhisek : {human_interaction_resp}")
    sys.stdin = f

    coordinator.execute("Abhisek : Turn it off.")

    # changing things back in order
    f.close()
    sys.stdin = f1

    llm_query = get_interaction_question(
        config.coordinator_config.global_config.logpath
    )
    messages = [
        HumanMessage(
            content=f"Check if the following question answer pair makes sense. (Respond 'yes' or 'no'). \n\n Question: {llm_query} \n Answer: {human_interaction_resp}"
        )
    ]
    llm_resp = config.evaluator_llm(messages)
    device_state = db.get_device_state(test_id)

    assert (
        "yes" in llm_resp.content.lower()
    ), f"This question: {llm_query} to this answer {human_interaction_resp} does not makes sense."

    assert (
        device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] == "off"
    ), "The TV by the plant is still on."

    flag = True

    for light in lights:
        if device_state[light]["main"]["switch"]["switch"]["value"] == "off":
            flag = False

    if device_state[tv_id]["main"]["switch"]["switch"]["value"] == "off":
        flag = False

    assert flag, "Other devices have been turned off too!"


# How many lights do I have?
@register(["simple", "test_set"])
def how_many_lights(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)

    result = coordinator.execute("Abhisek : How many lights do I have?")
    answer = result["output"]
    assert (
        "4" in answer or "four" in answer.lower()
    ), f"There are 4 lights. The response : {answer}"


# Can I change the color of the light by the fireplace?
@register(["device_resolution", "test_set"])
def can_i_change_light_color(device_state, config):
    test_id, coordinator = setup(device_state, config.coordinator_config)

    result = coordinator.execute(
        "Abhisek: Can I change the color of the light by the fireplace? Respond with a yes or a no."
    )
    answer = result["output"]
    assert (
        "yes" in answer.lower()
    ), f"The light by the fireplace can be turned on. The response : {answer}"


@register(["intent_resolution", "test_set"])
def freezer_too_cold(device_state, config):
    # there are two, thermostatCoolingSetpoint, and custom.thermostatSetpointControl
    # TODO: figure out which one is actually the right one, or just say both work
    freezer_temp_orig = float(
        device_state[fridge]["freezer"]["thermostatCoolingSetpoint"]["coolingSetpoint"][
            "value"
        ]
    )
    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute(
        "Abhisek: I think my freezer is set too cold, all my food is freezer burned."
    )
    device_state = db.get_device_state(test_id)
    new_freezer_temp = float(
        device_state[fridge]["freezer"]["thermostatCoolingSetpoint"]["coolingSetpoint"][
            "value"
        ]
    )
    assert (
        new_freezer_temp > freezer_temp_orig
    ), f"Failed to increase freezer set point. Original temp: {freezer_temp_orig}, new temp: {new_freezer_temp}"


# this one is like, impossible
@register(["persistence", "test_set"])
def notify_when_show_on(device_state, config):
    device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] = 50
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek: Let me know if anyone in the house watches Jeopardy without me by turning the light by the fireplace red."
    coordinator.execute(command)

    device_state = db.get_device_state(test_id)
    # to make it a bit easier, we'll set both TVs to channel 10 (where jeopardy is playing) and turn both on

    for device_id in tvs:
        device_state[device_id]["main"]["tvChannel"]["tvChannel"]["value"] = "10"
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
    db.set_device_state(test_id, device_state)

    trigger_command = listen(config)
    coordinator.execute(trigger_command)
    device_state = db.get_device_state(test_id)
    new_hue = device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"]

    assert (
        str(new_hue) == "0"
    ), f"Hue of fireplace light should have been set to zero (red), actual hue is {new_hue}"


@register(["intent_resolution", "test_set"])
def st_patrick_lights(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["colorControl"]["hue"]["value"] = "10"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Abhisek: Set up the lights for St. Patrick's day."
    coordinator.execute(command)
    # at least one should be green
    device_state = db.get_device_state(test_id)
    new_hues = [
        int(device_state[device_id]["main"]["colorControl"]["hue"]["value"])
        for device_id in lights
    ]
    # green is 120 degrees, which should correspond to 33% hue
    is_green = [(h > 30 and h < 36) for h in new_hues]
    assert True in is_green, (
        "At least one light should have turned green (hue between 30 and 36), but hues were: %s"
        % new_hues
    )


@register(["persistence", "test_set"])
def prank_husband(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[fridge]["main"]["contactSensor"]["contact"]["value"] = "closed"
    test_id, coordinator = setup(device_state, config.coordinator_config)
    command = "Amal: I want you to help me prank my husband. The next time someone opens the fridge, turn all the lights in the house off."
    coordinator.execute(command)

    device_state = db.get_device_state(test_id)
    device_state[fridge]["main"]["contactSensor"]["contact"]["value"] = "open"
    db.set_device_state(test_id, device_state)

    trigger_command = listen(config)
    coordinator.execute(trigger_command)
    device_state = db.get_device_state(test_id)
    light_states = [
        device_state[device_id]["main"]["switch"]["switch"]["value"]
        for device_id in lights
    ]
    assert "on" not in light_states, (
        "All lights should have been turned off, final states were: %s" % light_states
    )


@register(["device_resolution", "test_set", "command_chaining"])
def dont_turn_lights_that_are_on_blue(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"

    device_state[tv_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[fireplace_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[tv_light]["main"]["colorControl"]["hue"]["value"] = 0
    device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] = 0
    # freezer not below -10, so don't do anything
    device_state[fridge]["freezer"]["temperatureMeasurement"]["temperature"][
        "value"
    ] = -7

    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute(
        "Dmitriy: If the freezer is below minus 10 degrees, turn all the lights that are currently on blue."
    )
    device_state = db.get_device_state(test_id)

    assert (device_state[tv_light]["main"]["colorControl"]["hue"]["value"] == 0) and (
        device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] == 0
    ), "the hues of the lights should not have been modified as the freezer temp is above -10 degrees"


@register(["device_resolution", "test_set", "command_chaining"])
def do_turn_lights_that_are_on_blue(device_state, config):
    for device_id in lights:
        device_state[device_id]["main"]["switch"]["switch"]["value"] = "off"

    device_state[tv_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[fireplace_light]["main"]["switch"]["switch"]["value"] = "on"
    device_state[tv_light]["main"]["colorControl"]["hue"]["value"] = 0
    device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"] = 0
    # freezer is below -10, so don't do anything
    device_state[fridge]["freezer"]["temperatureMeasurement"]["temperature"][
        "value"
    ] = -17

    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute(
        "Dmitriy: If the freezer is below minus 10 degrees, turn all the lights that are currently on blue."
    )
    device_state = db.get_device_state(test_id)

    correct_value = 66
    # correct_value = 240
    assert np.isclose(
        int(device_state[tv_light]["main"]["colorControl"]["hue"]["value"]),
        correct_value,
        atol=5,
    ) and np.isclose(
        int(device_state[fireplace_light]["main"]["colorControl"]["hue"]["value"]),
        correct_value,
        atol=5,
    ), "the hues of the lights been modified as the freezer temp is below -10 degrees"


# Dmitriy's memory says he doesn't like hockey (it's not true)
@register(["personalization", "device_resolution", "test_set"])
def play_sports_not_hockey(device_state, config):
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "off"
    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute("Dmitriy: Put some sports on the TV by the credenza.")
    device_state = db.get_device_state(test_id)

    assert (
        device_state[tv_id]["main"]["switch"]["switch"]["value"] == "main"
    ), "credenza tv was not turned on"
    assert (
        str(device_state[tv_id]["main"]["tvChannel"]["tvChannel"]["value"]) == "7"
    ), "should have played basketball (the only non-hockey game currently on) but didn't"


@register(["device_resolution", "intent_resolution", "test_set"])
def crank_the_tv_thats_playing(device_state, config):
    device_state[frame_tv_id]["main"]["switch"]["switch"]["value"] = "off"
    device_state[tv_id]["main"]["switch"]["switch"]["value"] = "on"
    device_state[tv_id]["main"]["audioVolume"]["volume"]["value"] = 40
    test_id, coordinator = setup(device_state, config.coordinator_config)

    coordinator.execute(
        "Dmitriy: I love the song that's playing on TV by the credenza, crank it!"
    )
    device_state = db.get_device_state(test_id)

    assert (
        int(device_state[tv_id]["main"]["audioVolume"]["volume"]["value"]) > 40
    ), "should have increased the volume on the frame TV"
