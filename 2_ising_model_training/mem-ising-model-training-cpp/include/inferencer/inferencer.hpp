#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <memory>
#include <vector>
#include <cmath>
#include "models/ising_model.hpp"

class IsingInferencer {
public:
    // Function prototypes
    void update_partition_function(std::shared_ptr<IsingModel> ising_model, const std::vector<int>& configurations, bool update_order_1_partition_function = false);
    long double energy(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2);
    long double calculate_configuration_possibility(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2);
    long double get_Z(int order);
    void set_Z(long double value, int order);
private:
    long double Z1 = 0.0;
    long double Z2 = 0.0;
};

#endif
