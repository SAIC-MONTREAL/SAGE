"""
Plot manual analysis
"""

import json
from pathlib import Path

import pandas as pd


base_path = Path("logs/manual_analysis")

# temporary until fixed
# bad_annotations = [
# '{',
# # 'Missing information about montreal canadiens.',
# 'n/a',
# '  "action_input": {',
# '    "max_results": 1'
# ]

bad_annotations = []

results = []
for path in base_path.rglob("*.json"):
    with open(path, "r") as f:
        x = json.load(f)
    for trial, trial_analysis in x.items():
        for case_id, case_analysis in trial_analysis.items():
            results.append(
                {
                    "condition": path.stem,
                    "trial": trial,
                    "case_id": case_id,
                    "fc": case_analysis["failiure_category"],
                }
            )

df = pd.DataFrame(results)
# TODO: drop once they are all fixed
df = df[~df["fc"].isin(bad_annotations)]

df["fc"] = df["fc"].str.lower()
df["fc"] = df["fc"].str.strip(" ")
# df['fc'] = df['fc'].str.strip('\.')
df["fc"] = df["fc"].str.strip("\?")

# typos / unstandard usage
df["fc"] = df["fc"].str.replace(".", "")
df["fc"] = df["fc"].str.replace(" [unsure]", "")
df["fc"] = df["fc"].str.replace(" (user_preference)", "")
df["fc"] = df["fc"].str.replace(", error recovery", "")
df["fc"] = df["fc"].str.replace("code writting", "code writing")
df["fc"] = df["fc"].str.replace("planing", "planning")
df["fc"] = df["fc"].str.replace(
    "uses conditionchecker tool to see if mother is scheduled to visit ", ""
)
df["fc"] = df["fc"].str.replace("unable to figure out the right channel too ", "")
df["fc"] = df["fc"].str.replace("wrong user preference tool usage ", "")
df["fc"] = df["fc"].str.replace("formatting, hallucination", "hallucination")
df["fc"] = df["fc"].str.replace("tool population, formatting", "tool population")
df["fc"] = df["fc"].str.replace("formatting, tool selection", "tool selection")
df["fc"] = df["fc"].str.replace("tool selection, formmating", "tool selection")
df["fc"] = df["fc"].str.replace("tool selection, formatting", "tool selection")
df["fc"] = df["fc"].str.replace(
    "api request formatting, formatting", "api request formatting"
)
df["fc"] = df["fc"].str.replace("too selection, formatting", "tool selection")
df["fc"] = df["fc"].str.replace("faulty tools", "faulty tool")
df["fc"] = df["fc"].str.replace("command_understanding", "command understanding")

# this is for one specific test case failure_analysis_gpt3.5, trial_3, what_did_i_miss
df["fc"] = df["fc"].str.replace(
    "command understanding/faulty tool", "command understanding"
)

# for lemur trial_2 put_the_game_on_amal
df["fc"] = df["fc"].str.replace("tool selection faulty tool", "tool selection")

# failure_analysis_claude  trial_3  switch_off_everything
df["fc"] = df["fc"].str.replace("hallucination, tool selection", "hallucination")

# failure_analysis_lemur  trial_2  is_main_tv_on  llm error
df["fc"] = df["fc"].str.replace("llm error", "other")


tiers = {
    "formatting": 1,
    "command understanding": 1,
    "hallucination": 4,
    "planning": 2,
    "plan execution": 3,
    "tool selection": 3,
    "tool population": 3,
    "api request formatting": 3,
    "code writing": 3,
    "faulty tool": 4,
    "llm limitation": 4,
    "other": "NA",
}

df["count"] = 1

df_sum = df.groupby(["condition", "fc"])["count"].sum().reset_index()
df_sum2 = (
    df_sum.groupby("condition")["count"]
    .sum()
    .reset_index()
    .rename(columns={"count": "total"})
)
df_sum = df_sum.merge(df_sum2, on="condition")
df_sum["tier"] = df_sum["fc"].apply(lambda x: tiers[x])
llm_order = ["GPT4", "GPT4-turbo", "GPT3.5-turbo", "Lemur", "Claude2.1"]
llm_lookup = {
    "failure_analysis_claude": "Claude2.1",
    "failure_analysis_gpt3.5": "GPT3.5-turbo",
    "failure_analysis_gpt4": "GPT4",
    "failure_analysis_gpt4_turbo": "GPT4-turbo",
    "failure_analysis_lemur": "Lemur",
}
df_sum["llm"] = df_sum["condition"].apply(lambda x: llm_lookup[x])


fc_order = [
    "formatting",
    "command understanding",
    "planning",
    "plan execution",
    "tool selection",
    "tool population",
    "api request formatting",
    "code writing",
    "faulty tool",
    "llm limitation",
    "hallucination",
    "other",
]

# TODO: don't forget to update these if you move around tier categories, it is quite brittle
tier_counts = {
    "formatting": 2,
    "planning": 1,
    "plan execution": 5,
    "faulty tool": 3,
    "other": 1,
}

table_rows = []
for fc in fc_order:
    table_row = ""
    if fc in tier_counts:
        table_row += "\\hline \n \\multirow{%s}{*}{%s} & " % (
            tier_counts[fc],
            tiers[fc],
        )
    else:
        table_row += "& "

    table_row += fc + " "
    for llm in llm_order:
        df_row = df_sum[(df_sum["llm"] == llm) & (df_sum["fc"] == fc)]
        if len(df_row) > 1:
            raise Exception()
        elif len(df_row) == 0:
            table_row += "& 0 "
            continue
        table_row += "& {:.0f} ".format(
            100 * df_row.iloc[0]["count"] / df_row.iloc[0]["total"]
        )
    table_row += " \\\\"
    table_rows.append(table_row)

table_content = "\n".join(table_rows)

totals = df_sum.groupby("llm")["count"].sum()

# table_header = """
# \\begin{table*}[ht]
# \\centering
# \\begin{tabular}{|c|c|c|c|c|c|c|}
# \\hline
# \\textbf{Failure Tier} & \\textbf{Failure Type} & \\multirow{2}{*}{\\textbf{GPT4}} & \\multirow{2}{*}{\\textbf{GPT4-turbo}} & \\multirow{2}{*}{\\textbf{GPT3.5-turbo}} & \\multirow{2}{*}{\\textbf{Lemur}} & \\multirow{2}{*}{\\textbf{Claude2.1}} \\\\
# & &  \\textbf{total=%s} & \\textbf{total=%s} & \\textbf{total=%s} & \\textbf{total=%s} & \\textbf{total=%s} \\\\
# """ % (totals['GPT4'], totals['GPT4-turbo'], totals['GPT3.5-turbo'], totals['Lemur'], totals['Claude2.1'])


table_header = (
    """
\\begin{table*}[ht]
\\centering
\\caption{Results of manual failure analysis. This analysis categorizes failures into one of 12 failure types. The failure tiers column helps to contextualize the failure type category. In most cases, a failure on tier n implies that failures in lower tiers were avoided.}
\\begin{tabular}{|c|l|c|c|c|c|c|}
\\hline

\\textbf{Failure Tier} & \\textbf{Failure Type} & \\textbf{GPT4} & \\textbf{GPT4-turbo} & \\textbf{GPT3.5-turbo} & \\textbf{Lemur} & \\textbf{Claude2.1} \\\\
\\hline
& &  \\multicolumn{5}{c|}{Total Failures = \\# failures per run $\\times 3$}\\\\
\\cline{3-7}
& &  %s & %s & %s & %s & %s \\\\
\\cline{3-7}
"""
    % (
        totals["GPT4"],
        totals["GPT4-turbo"],
        totals["GPT3.5-turbo"],
        totals["Lemur"],
        totals["Claude2.1"],
    )
    + """
& &  \\multicolumn{5}{c|}{Failure rate (\%)}\\\\
"""
)


table_footer = """
\\hline
\\end{tabular}
\\label{table:manual_analysis}
\\end{table*}
"""

full_table = table_header + table_content + table_footer

with open("logs/failure_analysis_table.tex", "w") as f:
    f.write(full_table)

# template to fill out with descriptions of failure types

table_rows = []
for fc in fc_order:
    table_row = ""
    if fc in tier_counts:
        table_row += "\\hline \n \\multirow{%s}{*}{%s} & " % (
            tier_counts[fc],
            tiers[fc],
        )
    else:
        table_row += "& "

    table_row += fc + " & TODO"
    table_row += " \\\\"
    table_rows.append(table_row)

table_content = "\n".join(table_rows)

table_header = """
\\begin{table*}[ht]
\centering
\\begin{tabular}{|c|c|c|}
\\hline
\\textbf{Failure Tier} & \\textbf{Failure Type} & \\textbf{Description} \\\\ \\hline
"""


table_footer = """
\\hline
\\end{tabular}
\\caption{Description of manual failure analysis categories. The failure tiers column helps to contextualize the failure type category. In most cases, a failure on tier n implies that failures in lower tiers were avoided.}
\\label{table:failure_category_descriptions}
\\end{table*}
"""
full_table = table_header + table_content + table_footer

with open("logs/failure_categories_table.tex", "w") as f:
    f.write(full_table)
