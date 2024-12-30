#ifndef ISING_IO_HPP
#define ISING_IO_HPP
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>

#include "models/ising_model.hpp"
struct ISINGIO
{    
    static void serialize_ising_model_to_file(std::shared_ptr<IsingModel> ising_model, const std::string& model_filepath);

    static std::shared_ptr<IsingModel> load_ising_model_from_file(const std::string& model_filepath);
    static std::vector<int> read_spin_configurations(const std::string& file_path);
};

#endif
