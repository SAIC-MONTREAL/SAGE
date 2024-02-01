"""Functions to help visualize the agent's decision."""
import json
import os
import pickle
import textwrap
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Type

import cv2
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import tyro

from sage.base import BaseConfig
from sage.utils.logging_utils import extract_texts

ROOT = os.getenv("SMARTHOME_ROOT", default=None)

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"].join(
    [
        r"\usepackage{dashbox}",
        r"\setmainfont{xcolor}",
    ]
)


def image_resize(
    image: np.ndarray, width=None, height=None, inter=cv2.INTER_AREA
) -> np.ndarray:
    """Resize image proportionally (maintain aspect ratio)"""
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image

    if width is None and height is None:
        return image

    # check to see if the width is None

    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


@dataclass
class VisualizerConfig(BaseConfig):
    """Base Visualizer configuration"""

    _target: Type = field(default_factory=lambda: AgentVisualizer)

    logpath: str = (
        "oct_20_2pm_seabuscuit_run/2023-10-20 14:01:43.757528/put_the_game_on_abhisek"
    )
    figsize_textbox: tuple[int] = (12, 9)
    fontsize_textbox: int = 48
    foldername_textbox: str = "img"
    figname_textbox: str = "textbox.png"
    icon_width: int = 4
    graph_fontsize: int = 40
    folderpath_icons: str = "assets/icons"
    folderpath_graph: str = "graph/decision_tree"
    folderpath_transparent_icons: str = "assets/transparent_icons"
    graph_width: int = 4096
    graph_height: int = 2304


class AgentVisualizer:
    """Build and render decision tree from log experiment file"""

    def __init__(self, config=VisualizerConfig):
        self.config = config
        assert (
            config.logpath is not None
        ), "You need to specify a logpath in VisualizerConfig"
        self.logpath = os.path.join(ROOT, "logs", config.logpath)
        tool_file = os.path.join(self.logpath, "tools.pickle")
        assert os.path.exists(
            tool_file
        ), f"File tools.pickle not found in logs path {self.logpath}"
        with open(tool_file, "rb") as file:
            self.tools_list = pickle.load(file)

    def get_final_answer(self):
        """Extract feedback from Agent in log file when task is completed successfully"""
        with open(self.logpath + "/viz.log", "r") as f:
            text = f.read()

        return extract_texts(
            text=text, start_string="Final Answer: ", end_string="\x1b[0m\n[CHAIN END]"
        )

    def get_actions_from_log(self):
        """Read all actions (tools) from log file"""
        with open(self.logpath + "/viz.log", "r") as f:
            text = f.read()
        actions = extract_texts(
            text=text, start_string="Action: ", end_string="Action Input: "
        )
        arguments = extract_texts(
            text=text, start_string="Action Input: ", end_string="\x1b[0m\n"
        )

        return actions, arguments

    def get_tool_options(self, tool: list[str]):
        """Get available tools to agent (at given level of hierchy)"""

        for tool_list in self.tools_list:
            if tool in tool_list:
                return tool_list

        return [tool]

    def plot_text_box(
        self,
        tool: str = "device_disambiguation",
        arguments: str = "{arguments: 3}",
    ):
        """Generate right handed text box that prints tool chosen and chosen arguments"""

        #        fig = plt.figure(figsize=self.config.figsize_textbox)
        fig, ax = plt.subplots(figsize=self.config.figsize_textbox, tight_layout=True)

        text_kwargs = dict(
            ha="center",
            va="center",
            fontsize=self.config.fontsize_textbox,
            color="black",
            wrap=True,
        )

        # if data is a dictionary, use pretty display
        try:
            data = json.loads(arguments)

            def dict2str(data):
                for key in data:
                    data[key] = textwrap.fill(f"{data[key]}", 30)
                    arguments = json.dumps(data, indent=4, sort_keys=True)
                    arguments = arguments.replace("{", "")
                    arguments = arguments.replace("}", "")
                    arguments = arguments.replace("\\n", "\n")

                    if arguments[0:1] == "\n":
                        arguments = arguments[1:]

                return arguments

            if isinstance(data, dict):
                arguments = dict2str(data)
            elif isinstance(
                data, list
            ):  # sometimes, arguments are list of dictionaries
                arguments = ""

                for _data in data:
                    arguments += dict2str(_data)

            argument_string = f"{arguments}"
        except Exception:
            argument_string = textwrap.fill(f"{arguments}", 30)
        ax.text(
            0.5,
            0.5,
            r"{\textbf{Tool}}"
            + f"\n {tool}"
            + "\n \n"
            + r"\textbf{Arguments}"
            + "\n"
            + argument_string,
            **text_kwargs,
        )
        ax.axis("off")
        foldername = os.path.join(self.logpath, self.config.foldername_textbox)

        if not os.path.exists(foldername):
            os.makedirs(foldername)
        plt.savefig(
            os.path.join(
                self.logpath,
                self.config.foldername_textbox,
                self.config.figname_textbox,
            )
        )

    def build_graph(self, tools: list[str]):
        """Generate left handed graph that generates a visualization of the decision making process"""
        graph = gv.Digraph(format="png")
        graph.attr("node", fontsize=str(self.config.graph_fontsize))
        counter1 = 0

        if len(tools) > 0:
            last_name = tools[0]

        for counter, tool in enumerate(tools):
            candidate_tools = self.get_tool_options(tool)

            for candidate in candidate_tools:
                color = "#00000033"

                if candidate == tool:
                    next_name = candidate + str(counter1)
                    color = "blue"

                if candidate in ["Start", "Completed"]:
                    graph.node(
                        candidate + str(counter1),
                        label=candidate,
                        color="red",
                    )
                else:
                    if color == "blue":
                        img_file = os.path.join(
                            ROOT, self.config.folderpath_icons, candidate + ".png"
                        )

                        if not os.path.exists(img_file):
                            img_file = os.path.join(
                                ROOT, self.config.folderpath_icons, "empty.png"
                            )
                        img = cv2.imread(img_file)
                        height = self.config.icon_width * (img.shape[0] / img.shape[1])
                        graph.node(
                            candidate + str(counter1),
                            label="",
                            color="transparent",
                            image=img_file,
                            width=str(self.config.icon_width),
                            height=str(height),
                            fixedsize="true",
                        )
                    else:
                        img_file = os.path.join(
                            ROOT,
                            self.config.folderpath_transparent_icons,
                            candidate + ".png",
                        )

                        if not os.path.exists(img_file):
                            img_file = os.path.join(
                                ROOT,
                                self.config.folderpath_transparent_icons,
                                "empty.png",
                            )

                        img = cv2.imread(img_file)
                        height = self.config.icon_width * (img.shape[0] / img.shape[1])
                        graph.node(
                            candidate + str(counter1),
                            label="",
                            color="transparent",
                            image=img_file,
                            width=str(self.config.icon_width),
                            height=str(height),
                            fixedsize="true",
                        )

                if counter > 0:
                    graph.edge(last_name, candidate + str(counter1), color=color)
                counter1 += 1
            last_name = next_name
        graph.render(filename=os.path.join(self.logpath, self.config.folderpath_graph))

        return graph

    def simulate_log(
        self,
    ):
        """Live visualization of the log file as a decision tree"""
        tools, arguments = self.get_actions_from_log()
        final_list = self.get_final_answer()
        arguments = ["None"] + arguments
        tools = ["Start"] + tools

        if len(final_list) > 0:
            tools = tools + ["Completed"]
            arguments = arguments + [final_list[-1]]
        tool_list = []
        argument_list = []

        for tool, argument in zip(tools, arguments):
            tool_list.append(tool)
            argument_list.append(argument)
            self.build_graph(tool_list)
            self.plot_text_box(tool_list[-1], argument_list[-1])
            self.show_graph(wait_key=0)

    def visualize_log(
        self,
    ):
        """Live visualization of the log file as a decision tree"""
        tools, arguments = self.get_actions_from_log()
        final_list = self.get_final_answer()
        arguments = ["None"] + arguments
        tools = ["Start"] + tools

        if len(final_list) > 0:
            tools = tools + ["Completed"]
            arguments = arguments + [final_list[-1]]
        self.build_graph(tools)
        self.plot_text_box(tools[-1], arguments[-1])
        self.show_graph()

    def show_graph(
        self,
        wait_key: int = 1,
    ):
        """Combine left and right images into 1 and display"""
        img1 = cv2.imread(f"{self.logpath}/graph/decision_tree.png")
        img2 = cv2.imread(f"{self.logpath}/img/textbox.png")
        left_graph_width = int(self.config.graph_width * (2 / 3))
        fig_width = self.config.graph_width - left_graph_width
        img1_large = 255 * np.ones((self.config.graph_height, left_graph_width, 3))
        img2_large = 255 * np.ones((self.config.graph_height, fig_width, 3))

        def center_img(img_large: np.ndarray, img_small: np.ndarray):
            """Insert smaller image into larger canvas"""

            if img_small.shape[0] > img_large.shape[0]:
                img_small = image_resize(img_small, height=img_large.shape[0])

            if img_small.shape[1] > img_large.shape[1]:
                img_small = image_resize(img_small, width=img_large.shape[1])
            x_start, y_start = int(img_large.shape[0] / 2) - int(
                img_small.shape[0] / 2
            ), int(img_large.shape[1] / 2) - int(img_small.shape[1] / 2)
            img_large[
                x_start : x_start + img_small.shape[0],
                y_start : y_start + img_small.shape[1],
                :,
            ] = img_small

            return img_large.astype("uint8")

        if img1 is not None:
            img1_large = center_img(img1_large, img1)

        if img2 is not None:
            img2_large = center_img(img2_large, img2)
        img = np.concatenate((img1_large, img2_large), axis=1)
        cv2.imshow("viz", img)
        cv2.waitKey(wait_key)


def visualize_agent(logpath=None, frequency: int = 5) -> None:
    """Wrapper function to visualize graph in a loop from given logs path at given loop frequency"""
    viz_config = tyro.cli(VisualizerConfig)

    if logpath is not None:
        viz_config.logpath = logpath
    visualizer = viz_config.instantiate()

    while True:
        visualizer.visualize_log()
        time.sleep(1.0 / frequency)


def simulate_agent(logpath=None) -> None:
    """Wrapper function to visualize graph in a loop from given logs path at given loop frequency"""
    viz_config = tyro.cli(VisualizerConfig)

    if logpath is not None:
        viz_config.logpath = logpath
    visualizer = viz_config.instantiate()
    visualizer.simulate_log()


if __name__ == "__main__":
    simulate_agent()
