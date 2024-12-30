#ifndef ISING_MODEL_HPP
#define ISING_MODEL_HPP

#include <vector>
#include <random>

class IsingModel {
public:
    std::vector<std::vector<long double>> J;
    std::vector<long double> H;
    int n_sites;
    double temperature;

    // Constructor
    IsingModel(int n_sites, double temperature);

};

#endif  // ISING_MODEL_HPP
