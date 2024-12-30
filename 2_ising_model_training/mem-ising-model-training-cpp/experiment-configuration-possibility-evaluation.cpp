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
    int temp_n = config["mem-trainer"]["n"].as<int>();
    if (temp_n < 0) {
        std::cerr << "Error: n cannot be negative." << std::endl;
        return -1;
    }
    std::string ising_model_serialized_file = config["partition_function_inference"]["ising_model_serialized_file"].as<std::string>();
    std::string all_ising_spin_configuration_file = config["partition_function_inference"]["all_ising_spin_configuration_file"].as<std::string>();

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

    std::cout << "2. Calculate ising model partition functions. " << std::endl;

    std::shared_ptr<IsingInferencer> ising_model_inferencer = std::make_shared<IsingInferencer>();
    
    const long double Z1 = 7457414.45870293505822701263241469860077;
    const long double Z2 = 429.486658403292718311217157634018803947;
    
    std::cout << "  - partition functions Z1 = " << std::setprecision(39) << Z1 << std::endl;
    std::cout << "  - partition functions Z2 = " << std::setprecision(39) << Z2 << std::endl;

}