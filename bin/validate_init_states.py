import os
import json
from sasha.answers.grade_answer import validate_schema


def validate_init_states():
    """
    Validates initial state json files. Assumes filenames are of format `h#s#.json`
    """
    init_st = os.listdir(f"{os.environ['SMARTHOME_ROOT']}/sasha/initial_states/")

    # houses = defaultdict(list)
    for filename in init_st:
        #     houses[filename[:2]].append()
        if filename[-5:] == ".json":
            with open(
                f"{os.environ['SMARTHOME_ROOT']}/sasha/layouts/{filename[:2]}.json"
            ) as f:
                schema = json.load(f)
            with open(
                f"{os.environ['SMARTHOME_ROOT']}/sasha/initial_states/{filename}"
            ) as f:
                init_state = json.load(f)
            print(f"validating {filename}")
            validate_schema(schema, init_state, filename)


if __name__ == "__main__":
    # sashapath = sys.argv[-1]
    # os.system("cd {sashapath}") # sashapath == "../../"

    # sys.path.append("")
    # os.path.abspath()
    validate_init_states()
