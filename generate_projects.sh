
configuration_folders=(./configurations/spike_chain_n24 ./configurations/reservoir_n24)

for configuration_folder in "${configuration_folders[@]}"
do
    find "$configuration_folder" -type f -name "*.yaml" | while read -r yamlfile; do
        echo "Running with config: $yamlfile"
        python generate_samples.py --configuration-file-path "$yamlfile"
    done
done

