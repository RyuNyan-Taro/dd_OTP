#!/usr/bin/env bash

# Root directory for this example submission
EXAMPLE_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Current working directory
WORKING_DIR="$(pwd)"

(
    cd $EXAMPLE_ROOT_DIR \
    && uvx rpzip -r $WORKING_DIR/$1/submission.zip \
        mms-1b-all_final_model_20260219_180653 \
        main.py
)
