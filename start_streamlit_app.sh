#!/usr/bin/env bash

# Change to script's directory (project root)
cd "$(dirname "$0")"

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <string1> <string2(optional)>"
    exit 1
fi

# Run streamlit with provided arguments
uv run python -m streamlit run "subspace_partition/preimage/app.py" "$@"
