generate_samples_all:
	@echo "Running all experiments..."
	@for yamlfile in $(shell find ./configurations -type f -name "*.yaml"); do \
		echo "Running experiment with config: $$yamlfile"; \
		python generate_samples.py --configuration-file-path "$$yamlfile"; \
	done

train_all:
	@echo "Running all experiments..."
	@for yamlfile in $(shell find ./configurations -type f -name "*.yaml"); do \
		echo "Running experiment with config: $$yamlfile"; \
		python experiment_ising_training.py --configuration-file-path "$$yamlfile"; \
	done

partitioning_all:
	@echo "Running all experiments..."
	@for yamlfile in $(shell find ./configurations -type f -name "*.yaml"); do \
		echo "Running partitioning experiment with config: $$yamlfile"; \
		python partitioning_main.py --configuration-file-path "$$yamlfile"; \
	done

evaluation_all:
	@echo "Running all experiments..."
	@for yamlfile in $(shell find ./configurations -type f -name "*.yaml"); do \
		echo "Running partitioning experiment with config: $$yamlfile"; \
		python evaluation.py --configuration-file-path "$$yamlfile"; \
	done