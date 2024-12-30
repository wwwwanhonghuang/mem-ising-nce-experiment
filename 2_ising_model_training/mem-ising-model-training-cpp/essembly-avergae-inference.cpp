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


std::string current_time() {
    // Get the current time from the system clock
    auto now = std::chrono::system_clock::now();
    
    // Convert the current time to a time_t object
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    
    // Convert to a tm struct for formatting
    std::tm now_tm = *std::localtime(&now_time_t);
    
    // Format the time into a string
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    
    return oss.str();
}

int main(){
    
    
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    int temp_n = config["essembly_average_inference"]["n"].as<int>();
    if (temp_n < 0) {
        std::cerr << "Error: n cannot be negative." << std::endl;
        return -1;
    }
    std::string ising_model_serialized_file = config["essembly_average_inference"]["ising_model_serialized_file"].as<std::string>();

    std::string full_spin_configuration_file_path = config["essembly_average_inference"]["full_spin_configuration_file_path"].as<std::string>();
    std::string output_file_path = config["essembly_average_inference"]["output"].as<std::string>();


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


    std::vector<int> full_configurations = 
        ISINGIO::read_spin_configurations(full_spin_configuration_file_path);


    std::cout << "2. Calculate ising model partition functions. " << std::endl;

    std::shared_ptr<IsingInferencer> ising_model_inferencer = std::make_shared<IsingInferencer>();
    ising_model_inferencer->update_partition_function(ising_model, full_configurations, true);
    std::cout << "  - partition functions Z1 = " << std::setprecision(39) << ising_model_inferencer->get_Z(1) << std::endl;
    std::cout << "  - partition functions Z2 = " << std::setprecision(39) << ising_model_inferencer->get_Z(2) << std::endl;


    std::cout << "3. Calculate essembly averages. " << std::endl;
    
    std::cout << "        - calculate model_essembly_avgerage_si" << std::endl;
    std::vector<long double> model_essembly_average_si = 
        calculate_model_proposed_essembly_average_si(full_configurations, 
            ising_model, ising_model_inferencer);


    std::cout << "        - calculate model_essembly_avgerage_si_sj" << std::endl;
    std::vector<std::vector<long double>> model_essembly_average_si_sj = 
        calculate_model_proposed_essembly_average_si_sj(full_configurations, 
            ising_model, ising_model_inferencer);
            
       
     std::cout << "4. Write essembly averages to file " << output_file_path << std::endl;
	// Open the file for writing
	std::ofstream output_file(output_file_path);
	if (!output_file.is_open()) {
	    std::cerr << "Error: Could not open file for writing: " << output_file_path << std::endl;
	    return -1;
	}

	// Write model_assembly_average_si to file
	for (size_t i = 0; i < model_essembly_average_si.size(); ++i) {
	    output_file << std::setprecision(16) << model_essembly_average_si[i] << "\n";  // Optional precision formatting
	}

	// Write model_assembly_average_si_sj to file
	for (size_t i = 0; i < model_essembly_average_si_sj.size(); ++i) {
	    for (size_t j = 0; j < model_essembly_average_si_sj[i].size(); ++j) {
		output_file << std::setprecision(16) << model_essembly_average_si_sj[i][j] << " ";  // Optional precision formatting
	    }
	    output_file << "\n";
	}
	// Close the file
	output_file.close();


}

