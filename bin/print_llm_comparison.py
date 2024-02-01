"""
A script to write the results of different runs
"""
import sys
import glob
from utils.common import read_json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

TEST_CATEGORIES = [
    "device_resolution",
    "personalization",
    "persistence",
    "intent_resolution",
    "command_chaining",
    "simple",
]


def to_dataframe(results_json):
    """Convert a json to a dataframe"""
    column_names = ["case", "result", "error", "runtime"]
    column_names.extend(TEST_CATEGORIES)
    results_df = pd.DataFrame(columns=column_names)

    for key, value in results_json.items():
        types = value.pop("types")
        type_dict = {t: True if t in types else False for t in TEST_CATEGORIES}

        value.update(type_dict)
        results_df.loc[len(results_df)] = value
    results_df["result"].replace(["failure", "success"], [0, 1], inplace=True)

    return results_df


def read_results(path):
    """Read the latest json file in a folder"""
    filenames = sorted(
        glob.glob(f"{path}/*.json"), key=lambda x: x.split("/")[-1].split(".")[0]
    )

    results = read_json(filenames[-1])

    return results


def compute_success_rate_per_category(df):
    """Compute the success rate per test case category"""
    result_per_category = (
        df.where(df.device_resolution).groupby(["result"], as_index=False).case.count()
    )

    result_per_category.rename(columns={"case": "device_resolution"}, inplace=True)

    for cat in TEST_CATEGORIES[1:]:

        cat_df = df.where(df[cat]).groupby(["result"], as_index=False).case.count()
        result_per_category[cat] = cat_df["case"]

    result_per_category = result_per_category.div(
        result_per_category.sum(axis=0), axis=1
    )

    return result_per_category[result_per_category.result == 0.0]


def compute_success_rate(df):
    """COmpute the success rate"""
    fails = df[df.result == 0]

    return (len(df) - len(fails)) / len(df)


if __name__ == "__main__":
    paths = sys.argv[1:]

    success_rate = pd.DataFrame(columns=["LLM", "success_rate", "runtime"])
    success_rate_categories = pd.DataFrame(columns=TEST_CATEGORIES + ["LLM"])

    for path in paths:
        llm_name = path.split("/")[1].lower()
        results = read_results(path)
        results_df = to_dataframe(results)
        mean_runtime = np.round(results_df[results_df.result == 1].runtime.mean(), 2)
        sr = compute_success_rate(results_df)
        success_rate.loc[len(success_rate)] = {
            "LLM": llm_name,
            "success_rate": sr,
            "runtime": mean_runtime,
        }

        sr_per_cat = compute_success_rate_per_category(results_df)
        sr_per_cat["LLM"] = llm_name
        success_rate_categories.loc[len(success_rate_categories)] = sr_per_cat.values[
            0
        ][1:]

# Further aggregate results per LLM
success_rate = success_rate.groupby(["LLM"], as_index=False).mean()
success_rate_categories = success_rate_categories.groupby(
    ["LLM"], as_index=False
).mean()

print(success_rate)
print(success_rate_categories)


# plot success rate
success_rate["LLM"] = success_rate["LLM"].str.upper()
palette = dict(zip(success_rate.LLM.unique(), sns.color_palette("tab10")))
order = success_rate.sort_values("success_rate", ascending=False).LLM

ax = sns.barplot(
    x="LLM", y="success_rate", data=success_rate, order=order, palette=palette
)

for patch in ax.patches:
    h, w, x = patch.get_height(), patch.get_width(), patch.get_x()
    xy = (x + w / 2, h)
    text = f"{np.round(patch.get_height(), 2) * 100} %"
    ax.annotate(
        text=text,
        xy=xy,
        ha="center",
        va="center",
        size=12,
        xytext=(0, 8),
        textcoords="offset points",
    )

plt.show()
