#include "utils/ising_io.hpp"

// Method to serialize the Ising model to a file
void ISINGIO::serialize_ising_model_to_file(std::shared_ptr<IsingModel> ising_model, const std::string& model_filepath) {
    std::ofstream out_file(model_filepath);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open file for writing.\n";
        return;
    }

    // Save the number of sites and temperature
    out_file << ising_model->n_sites << "\n";
    out_file << ising_model->temperature << "\n";

    // Save the interaction matrix J
    for (const auto& row : ising_model->J) {
        for (long double value : row) {
            out_file << value << " ";
        }
        out_file << "\n";
    }

    // Save the external field H
    for (long double h : ising_model->H) {
        out_file << h << " ";
    }
    out_file << "\n";

    out_file.close();
}

std::shared_ptr<IsingModel> ISINGIO::load_ising_model_from_file(const std::string& model_filepath) {
    std::ifstream in_file(model_filepath);
    if (!in_file.is_open()) {
        std::cerr << "Error: Could not open file for reading.\n";
        return nullptr;
    }

    int n_sites;
    double temperature;

    // Load the number of sites and temperature
    in_file >> n_sites;
    in_file >> temperature;

    // Resize J and H based on the number of sites
    std::vector<std::vector<long double>> J(n_sites, std::vector<long double>(n_sites, 0.0));
    std::vector<long double> H(n_sites, 0.0);

    // Load the interaction matrix J
    for (int i = 0; i < n_sites; ++i) {
        for (int j = 0; j < n_sites; ++j) {
            in_file >> J[i][j];
        }
    }

    // Load the external field H
    for (int i = 0; i < n_sites; ++i) {
        in_file >> H[i];
    }

    in_file.close();

    // Create and return the Ising model instance
    std::shared_ptr<IsingModel> ising_model = std::make_shared<IsingModel>(n_sites, temperature);
    ising_model->J = std::move(J);
    ising_model->H = std::move(H);
    
    return ising_model;
}

std::vector<int> ISINGIO::read_spin_configurations(const std::string& file_path) {
    std::ifstream input_training_data(file_path, std::ios::binary);
    std::vector<int> configurations_for_training;

    if (!input_training_data) {
        std::cerr << "Error: cannot open Ising model configuration data file: " << file_path << std::endl;
        return configurations_for_training;
    }

    while (input_training_data) {
        std::vector<char> buffer(4);
        input_training_data.read(buffer.data(), 4);
        std::streamsize bytes_read = input_training_data.gcount();

        if (bytes_read > 0) {
            int value = *reinterpret_cast<int*>(buffer.data());
            configurations_for_training.push_back(value);
        }
    }

    input_training_data.close();
    return configurations_for_training;
}
