#!/bin/bash -e

python -m pytest tests/models/test_detection.py tests/models/yolo -v
