#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <bitset>

int main(){
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: cannot open configuration file." << std::endl;
        return -1;
    }
    size_t n = config["configuration_generator"]["n"].as<int>();
    std::string output_file_path = config["configuration_generator"]["output_file"].as<std::string>();
    std::ofstream output_stream(output_file_path, std::ios::binary);
    if (!output_stream){
        std::cerr << "Error: cannot create stream for output file " << output_file_path << std::endl;
        return -1;
    }

    int total_configurations = (1 << n) - 1;
    std::cout << "generate " << total_configurations << " configurations, n = " << n << std::endl;
    for(int c = 0; c <= total_configurations; c++){
        output_stream.write(reinterpret_cast<const char*>(&c), sizeof(c));
    }
    output_stream.close();
    return 0;
}
