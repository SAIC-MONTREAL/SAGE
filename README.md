# SAGE

Implementation of the paper [Smarthome Agent with Grounded Execution](https://arxiv.org/abs/2311.00772).

# Usage

## Installation

SAGE is tested using Python 3.10.

### Set up environment variables

You will need to setup some environment variables before running the system.
These variables can be found under `smarthome-llms/bin/config.sh`.

Set the variables accordingly and run the script:

```
source ./bin/config.sh
```

**Note:** these will not persist in a new terminal. Add the export commands from the script to your `~/.bashrc` file if you don't want to run `config.sh` every time you use SAGE.

### Install libraries

Install the repo and the requirements

```
pip install -e .
pip install -r requirements.txt
```

## LLMs
Our framework supports closed- and open-source LLMs like:
- [chatGPT](https://chat.openai.com/)
- [Claude](https://www.anthropic.com/news/claude-2-1)
- [Lemur](https://github.com/OpenLemur/Lemur)

To host open-source LLMs, we used [Text generation API](https://github.com/huggingface/text-generation-inference) from hugging face.

## Using SAGE

1 - Start the mongo DB docker.

```
cd $SMARTHOME_ROOT/docker
docker compose up
```

2 - Run the demo script

```
python $SMARTHOME_ROOT/bin/demo.py
```

## Using our smart home performance test benchmark

We carefully designed and implemented an LLM evaluation benchmark for smarthomes.
The benchmark consists of 50 testcases belonging to different tasks within a smarthome.

You can look at the testcases in `$SMARTHOME_ROOT/sage/testing/testcases.py`

To reproduce the results in our paper, please follow these steps:

1 - Launch the trigger server for persistent command checking
```
python $SMARTHOME_ROOT/sage/testing/run_server.py
```

2 - Run the test suite on a single LLM
```
python $SMARTHOME_ROOT/sage/testing/test_runner.py
```

Optional: Launch the test benchmark (10 LLMs x 3 runs)
```
sh $SMARTHOME_ROOT/bin/run_tests.sh
```

## Enabling Gmail and Google Calendar tools (optional)

To use these tools with SAGE (after setup and authentication, described below), you must activate them with the `--enable-google` flag:

If using SAGE normally, with `demo.py`:
```
python $SMARTHOME_ROOT/bin/demo.py --enable-google
```

If running the test suite with `test_runner.py`:
```
python $SMARTHOME_ROOT/sage/testing/test_runner.py --enable-google
```

If running benchmark with `run_tests.sh`:
```
sh $SMARTHOME_ROOT/bin/run_tests.sh --enable-google
```


### Initial Setup
To use Gmail and Google Calendar with SAGE, you must create an app in the Google Cloud console and give it access to Gmail and Google Calendar. To do so, you can follow [this guide](https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application).
Once you have created the `credentials.json` file, download it to `$SMARTHOME_ROOT/sage/misc_tools/apis/`, and rename it to `gcloud_credentials.json`.
> **Note:** This will give SAGE access to the emails and calendar events associated with the Google account you used to set up the Google Cloud app.

<details>
<summary><b>Set up Google Calendar events for testing</b></summary>
If you want to run the set of tests that use Google Calendar, you will need to set up 2 events in the calendar associated with the Google account you used to set up the Google Cloud app. If you do not do this, the test runner will not crash, but the tests will not pass.

1. "Watch Casablanca with Mom" on the Saturday of the current week.
2. "Cook Dinner" on the Saturday of the current week.

</details>


Now, authenticate the application, as described in the following Authentication section:

### Authentication

To use Gmail and Google Calendar, you must authenticate the application. If this is not done, the system will throw an error similar to `google.auth.exceptions.RefreshError: ('invalid_grant: Token has been expired or revoked.', {'error': 'invalid_grant', 'error_description': 'Token has been expired or revoked.'})`.

Sometimes the authentication must be refreshed even when it has already been done previously. Usually after a couple of days.

#### To authenticate, run:

```
python $SMARTHOME_ROOT/misc_tools/gcloud_auth.py
```

This will give you a link to paste into your browser for authentication. Log in with your email credentials.

<details>
<summary><b>Troubleshooting</b></summary>

* If you get an error like `requests.exceptions.SSLError: HTTPSConnectionPool(host='oauth2.googleapis.com', port=443)`, make sure your `requests` package is the same as that given in `requirements.txt` (2.28.2 at the time of writing this).
* If you get `AttributeError: 'InstalledAppFlow' object has no attribute 'run_console'`, set your Google packages to the following:
```
google-api-core               2.11.1
google-api-python-client      2.98.0
google-auth                   2.23.0
google-auth-httplib2          0.1.1
google-auth-oauthlib          0.4.1
google-search-results         2.4.2
googleapis-common-protos      1.60.0
```

</details>


Finally, paste the authorization code it gives you into the terminal. This should generate a `token.pickle` file in `$SMARTHOME_ROOT/sage/misc_tools/apis/`. You should now be able to run everything as normal.

# Code

## Configuration

Our configuration system relies on dataclass configs that can be easily modified from the command line.

### Base Components
All basic, reusable config components can be found in `base.py`. There are two main config classes: `BaseConfig` and `BaseToolConfig`. The `BaseConfig` is the most basic config. The `BaseToolConfig` is specific to configure tools.

### Creating new configs
If you need to create a new config for your tool, there are two ways to do this

#### Create a new config class
You will need to create a corresponding config to your tool class where you expose the parameters that you want to be configurable.
As an example, let's say you want to create a new `Tool` class called `MyTool`. Before the tool definition, you define the config `MyToolConfig` which points to the `MyTool` class using the `_target` attribute.
```python
@dataclass
class MyToolConfig(BaseToolConfig):
    """MyTool Config"""
    _target: Type = field(default_factory=lambda: MyTool)
    field1: int = 5
    ....

class MyTool(SAGEBaseTool):
    """MyTool"""
    def setup(config: MyToolConfig)->None:
        field1 = config.field1
```

### Modifying from CLI
You can use CLI to change different parameters as showcase below:
```bash
python $SMARTHOME_ROOT/bin/demo.py --output_dir test --tool-configs.0.top_k 5
```
If you want to load an existing config:
```bash
python $SMARTHOME_ROOT/bin/demo.py --load_config PATH_TO_CONFIG
```


## Memory

This section details on:
* The memory construction and storage
* The usage of the memory bank

### Introduction
Before starting, it is good to know the different terms we will be using in this document and also in the code
| Syntax      | Description |
| ----------- | ----------- |
| profile/preferences      | Used interchangeably to denote the list of user preferences. The profile is presented by a dictionary where the keys are the theme and the value is the user preference. Example: {'movie_genre': ['Thriller', 'Drama'], ....}      |
| Memory   | is the data structure that stores interactions between the user and the assistant. These instructions can be zero-shot interaction (referred to as **instruction**) or a conservation. For now, only zero-shot interactions are supported.|
|Instruction | Is a zero-shot command that the user give to the assistant. This is similar to the instruction you would give to your Alexa|
|Index|To do memory retrieval, all the instructions in the memory needs to be indexed. The result is called index which will be use to conduct similarity search between a query and the memory|

### Memory construction and storage
#### Memory construction
To create a memory:

```bash
python $SMARTHOME_ROOT/bin/generate_multiuser_memory.py --save_dir SAVE_DIRECTORY --num_instructions_to_generate 150 --num_users 2
```

This script will use `GPT-4` to generate `instructions` for each user. These instructions are saved under separate folders, one for each user. Each instruction is a dictionary as follows:

```json
{
  "instruction": "Are there any new sci-fi shows available to watch?",
  "request_idx": 24,
  "date": "2023-08-10"
}
```
`request_idx` is the request number. `date` is the date when the instruction is given. `instruction` is the user command.

**NOTE1**: The date is given randomly for now

**NOTE2**: In hindsight, `$SMARTHOME_ROOT/bin/generate_multiuser_memory.py` uses specific _prompts_ available [here](https://github.com/SAIC-MONTREAL/SAGE/blob/main/sage/retrieval/data_generator/bootstrap_instructions.py).

#### Memory storage
After generating the instructions, we construct the `memory_bank` which is how the memory will be stored and used. The `memory_bank` consists of:
* `History`: This contains all the user interactions timestamped with the date
* `Profile`: This is the inferred user profile from the history

A sample of the memory bank is available [here](https://github.com/SAIC-MONTREAL/SAGE/blob/main/data/memory_data/large_memory_bank.json)

### Memory utilisation
The memory bank is used for (1) memory retrieval, (2) User Preference understanding

#### Memory Retrieval
To get the most relevant instructions from the memory to a user query:

```python
memory = MemoryBank()

memory.read_from_json(path_to_memory)

#This assumes that the indexes for the users are created

memory.search("user_name", query)
```
#### User Preference understanding
To infer user profiles/preferences from instructions, the `UserProfiler` class is used. It implements a hierarchical approach by first generating daily summaries and then aggregating them into one global user profile.
This approach is inspired from the [SiliconFriend](https://arxiv.org/pdf/2305.10250.pdf) paper.
