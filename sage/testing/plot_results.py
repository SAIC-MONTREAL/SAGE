from smartthings.docmanager import DocManager


if __name__ == "__main__":
    # Libraries
    import os
    import numpy as np
    from typing import Type, Dict
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import pi
    from testing.testcases import TEST_REGISTER
    import matplotlib.pyplot as plt
    import seaborn as sns

    def compute_overall_results(data: pd.DataFrame) -> float:
        # Compute average success in overall experiment
        result_list = data.values[data.index.get_loc("result")]
        types_list = data.values[data.index.get_loc("types")]
        return (result_list.tolist().count("success") / len(result_list)) * 100

    def compute_categorical_success(data: pd.DataFrame) -> Dict:
        # compute average success conditioned on task category
        results = {}
        for category in [
            "device_resolution",
            "personalization",
            "persistence",
            "intent_resolution",
            "command_chaining",
        ]:
            category_indices = [category in data[item]["types"] for item in data.keys()]
            results_array = data.values[data.index.get_loc("result")][category_indices]
            if results_array.shape[0] == 0:
                results[category] = 0
            else:
                results[category] = (
                    results_array.tolist().count("success")
                    / len(results_array.tolist())
                    * 100
                )
        #
        return results

    def plot_overall_success(results):
        # Set a Seaborn style (optional but enhances aesthetics)
        sns.set(style="whitegrid")
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        # Create the bar plot
        categories = ["one_prompt", "sasha", "sage"]
        values = [results[key]["overall"] for key in categories]
        sns.barplot(x=categories, y=values, palette="viridis", ax=ax)
        # Customize the plot (optional)
        ax.set_xlabel("Methods", fontsize=14)
        ax.set_ylabel("Success Rate", fontsize=14)
        # ax.set_title('Beautiful Bar Plot', fontsize=16)
        ax.grid(axis="x", linestyle="--", alpha=0.6)  # Show the plot plt.show()
        plt.savefig("overall_results.pdf", format="pdf")
        plt.show()

    def plot_categorical_success(results_dict: Dict):
        # Define the methods, metrics, and performance data
        methods = ["one_prompt", "sasha", "sage"]
        metrics = [
            "device_resolution",
            "personalization",
            "persistence",
            "intent_resolution",
            "command_chaining",
        ]
        performance_list = []
        for method in methods:
            performance_list.append(
                [results_dict[method]["categories"][item] for item in metrics]
            )
        performance_data = np.array(performance_list)
        # Set the figure size and style
        plt.figure(figsize=(10, 6))
        plt.style.use("ggplot")
        # Set the bar width and spacing
        bar_width = 0.2
        bar_spacing = np.arange(len(metrics))
        # Create a bar for each method
        for i, method in enumerate(methods):
            plt.bar(
                bar_spacing + i * bar_width,
                performance_data[i],
                bar_width,
                label=method,
            )
        # Customize the chart
        plt.xlabel("Query Type")
        plt.ylabel("Success Rate")
        #        plt.title('Performance of Different Methods Over Metrics')
        plt.xticks(bar_spacing + bar_width, metrics)
        plt.legend()
        # Show the chart
        plt.tight_layout()
        plt.savefig("categorical_results.pdf", format="pdf")
        plt.show()

    def plot_radar_chart(results_dict: Dict):
        # plot radar chart to show categorical results
        # Create the radar chart
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        colors = ["b", "r", "k"]
        for key, color in zip(results_dict, colors):
            results = results_dict[key]["categories"]
            categories = [*results.keys()]
            values = [*results.values()]
            num_categories = len(categories)
            angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
            ax.fill(angles, values, color, alpha=0.2)
            angles2 = angles + angles[:1]
            values2 = values + values[:1]
            ax.plot(
                angles2,
                values2,
                linewidth=1,
                linestyle="solid",
                color=color,
                marker="o",
                markersize=6,
                markerfacecolor="b",
            )
            ax.set_xticks(angles)
            ax.set_xticklabels(categories)
            ax.set_rmax(80)
            ax.set_rticks([0, 25, 50, 75])  # Less radial ticks
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)
        plt.savefig("categorical_results.pdf", format="pdf")
        plt.show()

    def relabel_results(json_file):
        import json

        file = open(json_file, "rb")
        data = json.load(file)
        successes = []
        for key in data.keys():
            if data[key]["result"] == "failure":
                print(data[key]["case"], data[key]["result"])
                successes.append(key)


    ROOT = os.getenv("SMARTHOME_ROOT", default=None)
    results = {}
    for key in ["sage"]:
        results[key] = {}
        data = pd.read_json(
            os.path.join(
                "/Users/f.hogan/src/smarthome-llms/logs/test_logs",
                key + "_results.json",
            )
        )
        results[key]["categories"] = compute_categorical_success(data)
        results[key]["overall"] = compute_overall_results(data)
        results[key]["data"] = data
        relabel_results(
            os.path.join(
                "/Users/f.hogan/src/smarthome-llms/logs/test_logs",
                key + "_results.json",
            )
        )
#    plot_categorical_success(results)
#    plot_overall_success(results)
