"""
Run this script to summarize GPT usage
"""

from utils.our_openai import OpenAiUsageDb
import pandas as pd


def compute_cost(row):
    if "gpt-4" in row["model"]:
        input_token_cost = 0.03 / 1000
        output_token_cost = 0.06 / 1000
        if "32k" in row["model"]:
            input_token_cost = 0.06 / 1000
            output_token_cost = 0.12 / 1000
    elif "3.5" in row["model"]:
        input_token_cost = 0.0015 / 1000
        output_token_cost = 0.002 / 1000
        if "16k" in row["model"]:
            input_token_cost = 0.003 / 1000
            output_token_cost = 0.004 / 1000
    else:
        raise ValueError("invalid model %s" % row["model"])
    if row["type"] == "input":
        return row["tokens_used"] * input_token_cost
    elif row["type"] == "output":
        return row["tokens_used"] * output_token_cost
    else:
        raise ValueError("invalid type %s" % row["type"])


def extract_user(row):
    user_ids = [
        "d.rivkin",
        "f.hogan",
        "abhisek.k",
        "amal.feriani",
        "adam.sigal",
        "greg.dudek",
    ]
    for user in user_ids:
        if user in row["caller"]:
            return user
    return "mystery person"


db = OpenAiUsageDb()

all_usage = db.get_all_usage()
df = pd.DataFrame(all_usage)
df["cost"] = df.apply(compute_cost, axis=1)
df["user"] = df.apply(extract_user, axis=1)
df["year"] = df["log_ts"].dt.year
df["month"] = df["log_ts"].dt.month
print("total usage since we started tracking: $USD %s" % df["cost"].sum())
print("")

last_3_months = df.groupby(["year", "month"])["cost"].sum().sort_index()[-3:]
print("total usage last 3 months:")
print(last_3_months)
print("")


this_year = df["year"].max()
this_month = df[df["year"] == this_year]["month"].max()
this_month_usage = df[(df["year"] == this_year) & (df["month"] == this_month)]
this_month_by_user = this_month_usage.groupby(["user"])["cost"].sum()
print("total usage this month by user:")
print(this_month_by_user)
