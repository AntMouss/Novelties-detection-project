#!/bin/bash
# shellcheck disable=SC2164
cd /home/mouss/PycharmProjects/novelties-detection-git
export OUTPUT_PATH=$1
export PORT=5000
python server.py