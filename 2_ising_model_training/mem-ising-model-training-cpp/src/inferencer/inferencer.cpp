#include "inferencer/inferencer.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <omp.h>

#include "utils/ising_model_utils.hpp"

void IsingInferencer::update_partition_function(std::shared_ptr<IsingModel> ising_model, 
                const std::vector<int>& configurations, bool update_order_1_partition_function) {
    Z2 = 0.0;
    Z1 = 0.0;
    #pragma omp parallel for reduction(+:Z2, Z1)
    for (int configuration : configurations) {
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        Z2 += std::exp(-energy(ising_model, v) / ising_model->temperature);
        if (update_order_1_partition_function) {
            long double nagative_energy_term = -energy(ising_model, v, 1);
            // assert(!std::isnan(nagative_energy_term));
            Z1 += std::exp(nagative_energy_term / ising_model->temperature);
            // assert(!std::isnan(Z1));
        }
    }
    // assert(!std::isnan(Z1));
    // assert(!std::isnan(Z2));
    // assert(Z1 > 0);
    // assert(Z2 > 0);
}

long double IsingInferencer::energy(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order) {
    long double energy = 0.0;
    // std::cout << "Configuration: ";
    // for (size_t i = 0; i < configuration.size(); i++) {
    //     std::cout << static_cast<int>(configuration[i]) << " ";
    // }
    // std::cout << std::endl;
    
    if (order != 1 && order != 2) {
        std::cout << "Error: Order should equal to 1 or 2." << std::endl;
        assert(false);
    }

    if (order == 2) {
        for (int i = 0; i < ising_model->n_sites; i++) {
            for (int j = 0; j < ising_model->n_sites; j++) {
                if (i != j) {
                    // assert(!std::isnan(ising_model->J[i][j]));
                    // assert(!std::isnan(configuration[i]));
                    // assert(!std::isnan(configuration[j]));

                    // assert(!std::isnan(energy));
                    // std::cout << " i = " << i << " j = " << j << " J = " << ising_model->J[i][j] << std::endl;

                    energy += -ising_model->J[i][j] * configuration[i] * configuration[j];
                }
            }
        }
    }
    
    for (int i = 0; i < ising_model->n_sites; i++) {
        // assert(!std::isnan(ising_model->H[i]));
        // assert(!std::isnan(configuration[i]));
        // assert(!std::isnan(energy));

        // std::cout << "H[" << i << "] = " << ising_model->H[i] 
        //       << ", configuration[" << i << "] = " << static_cast<int>(configuration[i]) << std::endl;
        // std::cout << energy << " += " << -ising_model->H[i] * static_cast<int>(configuration[i]) << std::endl;
        energy += -ising_model->H[i] * static_cast<int>(configuration[i]);
        // assert(!std::isnan(energy));
    }
    // assert(!std::isnan(energy));

    return energy;
}

long double IsingInferencer::calculate_configuration_possibility(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order) {
    if (order == 2) {
        // assert(!std::isnan(Z2));
        // assert(Z2 > 0);
        return std::exp(-energy(ising_model, configuration) / ising_model->temperature) / Z2;
    }

    if (order == 1) {
        // assert(!std::isnan(Z1));
        // assert(Z1 > 0);
        return std::exp(-energy(ising_model, configuration, 1) / ising_model->temperature) / Z1;
    }

    std::cout << "Model inference order should equal to 1 or 2." << std::endl;
    assert(false);
}

long double IsingInferencer::get_Z(int order){
    if (order != 1 && order != 2) {
        std::cout << "Error: Order should equal to 1 or 2." << std::endl;
        assert(false);
    }
    if(order == 1) return Z1;
    if(order == 2) return Z2;
    return -1;
}

void IsingInferencer::set_Z(long double value, int order){
    if(order == 1){
        this->Z1 = value;
    }else if (order == 2){
        this->Z2 = value;
    }
    std::cout << "Error: Order should equal to 1 or 2." << std::endl;
    assert(false);    
}