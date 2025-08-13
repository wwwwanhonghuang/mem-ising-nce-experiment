#!/bin/bash

# Recursively find all yaml files under ./configurations
find ./configurations -type f -name "*.yaml" | while read -r yamlfile; do
    echo "Running experiment with config: $yamlfile"
    python experiment_ising_training.py --configuration-file-path "$yamlfile"
done

