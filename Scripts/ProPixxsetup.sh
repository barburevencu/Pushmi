#! /usr/bin/bash
#  Set parameters for the ProPixx videoprojector @ MEG Neuropsin

# Append /usr/local/bin to the PATH environment variable to ensure vputil is found
# export PATH=$PATH:/usr/local/bin

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the log directory path
LOG_DIR="${SCRIPT_DIR}/temporary"

# Create the log directory if it does not exist
mkdir -p "$LOG_DIR"

# Enable debugging
set -x

# Define the log file path using the log directory variable
export LOG_FILE="$LOG_DIR/log_PROPixx.log"

# Log the start of the script by adding a separator line to the log file
echo "#######################" >> "$LOG_FILE"

# Append the current date and time to the log file
date >> "$LOG_FILE"

# Set ProPixx video mode
# 0 -> RGB 120Hz (Default)
# 2 -> RGB Quad 480Hz
# The command sets the video mode to RGB 120Hz (default) and logs the operation
echo 0 | vputil -seq -quit >>"$LOG_FILE"

# Set ProPixx luminosity. Must be placed at the end of the script.
# 0 -> 100% brightness
# 1 -> 50% brightness
# 2 -> 25% brightness
# 3 -> 12.5% brightness
# 4 -> 6.25% brightness
# The command sets the luminosity to 12.5% and logs the operation
echo 3 | vputil -ppi -quit >>"$LOG_FILE"

# Disable debugging
set +x
