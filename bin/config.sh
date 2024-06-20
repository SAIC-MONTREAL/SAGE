#!/bin/bash

# OPENAI API KEY
export OPENAI_API_KEY=""

# ANTHROPIC API KEY
export ANTHROPIC_API_KEY=""

# HUGGING FACE API KEY
export HUGGINGFACEHUB_API_TOKEN=""

# WEATHER API KEY
export OPENWEATHERMAP_API_KEY=""

# SMARTTHINGS API TOKEN
# Go to https://account.smartthings.com/tokens
export SMARTTHINGS_API_TOKEN=""

# Leave as empty string:
export CURL_CA_BUNDLE=""

BIN_FOLDER="$(cd "$(dirname -- "$0")" >/dev/null; pwd -P)/$(basename -- "$1")"

export SMARTHOME_ROOT="$(dirname "$BIN_FOLDER")"
export TRIGGER_SERVER_URL="0.0.0.0:5797"
export MONGODB_SERVER_URL="0.0.0.0:27017"
