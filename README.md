## Reproduction

## 1 Produce encoded observation data

### 1.1 Produce encoded observation data from spyNNaker's records

``` bash
$ python -m 1_produce_observation_data.produce_observation_data_from_spynnaker_records
```

### 1.2 Produce encoded observation data from simulator LIF network records

``` bash
$ python -m 1_produce_observation_data.produce_observation_data_from_simulator_records
```



## 2 Train Ising Model

### 2.1 Build Project

``` bash 
$ cd \mem-ising-nce-experiment\2_ising_model_training\mem-ising-model-training-cpp
$ cmake -DCMAKE_BUILD_TYPE=Release .
$ make -j
```

### 2.2 Generate Full Configurations

It may need firstly modify the `config.yaml` to ensure path of files and numbers of neuron populations are set to expected values. 

After `config.yaml` is prepared, follow command can generate full ising spin configurations, with size of (1 << n_neuron_populations) * 4 bytes.

Each configuration is encoded into a 32-bit integer. As such, currently, the trainer and the generator support at most 32 neuron populations.

``` bash
$ cd <path-to-2_ising_model_training>
$ ./bin/configuration_generator
```

















