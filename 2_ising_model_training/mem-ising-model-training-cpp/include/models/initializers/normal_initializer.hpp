
#ifndef NORMAL_INITIALIZER_HPP
#define NORMAL_INITIALIZER_HPP
#include "models/initializers/base_initializer.hpp"

class NormalIsingModelInitializer : public BaseIsingModelInitializer {
public:
    NormalIsingModelInitializer(double j_min, double j_max, double h_min, double h_max)
        : _j_min(j_min), _j_max(j_max), _h_min(h_min), _h_max(h_max) {}

    void initialize(std::shared_ptr<IsingModel> ising_model) override;

private:
    double _j_min, _j_max, _h_min, _h_max;  
};
#endif
