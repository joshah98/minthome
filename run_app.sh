#!/bin/bash
echo "Running Flask App"
source ./venv/Scripts/activate
export FLASK_APP=server.py
flask run
