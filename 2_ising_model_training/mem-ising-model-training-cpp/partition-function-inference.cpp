#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <cassert>
#include <chrono>
#include <iomanip>

#include "macros.def"
#include "models/ising_model.hpp"

#include "utils/ising_model_utils.hpp"
#include "inferencer/inferencer.hpp"
#include "mem_training/mem_trainer.hpp"
#include "utils/ising_io.hpp"
#include "models/initializers/normal_initializer.hpp"

int main(){
    
    
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    int temp_n = config["partition_function_inference"]["n"].as<int>();
    if (temp_n < 0) {
        std::cerr << "Error: n cannot be negative." << std::endl;
        return -1;
    }
    std::string ising_model_serialized_file = config["partition_function_inference"]["ising_model_serialized_file"].as<std::string>();
    std::string all_ising_spin_configuration_file = config["partition_function_inference"]["full_spin_configuration_file_path"].as<std::string>();

    std::cout << "1. load ising model from serialized file " << ising_model_serialized_file << std::endl;
    std::shared_ptr<IsingModel> ising_model = 
        ISINGIO::load_ising_model_from_file(ising_model_serialized_file);
    for(int i = 0; i < ising_model->n_sites; i++){
        std::cout << "H[" << i << "] = " << ising_model->H[i] << std::endl;
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
            std::cout << "J[" << i << ", " << j << "] = " << 
                ising_model->J[i][j] << std::endl;
        }
    }


    std::vector<int> all_configurations = 
        ISINGIO::read_spin_configurations(all_ising_spin_configuration_file);


    std::cout << "2. Calculate ising model partition functions. " << std::endl;

    std::shared_ptr<IsingInferencer> ising_model_inferencer = std::make_shared<IsingInferencer>();
    ising_model_inferencer->update_partition_function(ising_model, all_configurations, true);
    std::cout << "  - partition functions Z1 = " << std::setprecision(39) << ising_model_inferencer->get_Z(1) << std::endl;
    std::cout << "  - partition functions Z2 = " << std::setprecision(39) << ising_model_inferencer->get_Z(2) << std::endl;

    
}

