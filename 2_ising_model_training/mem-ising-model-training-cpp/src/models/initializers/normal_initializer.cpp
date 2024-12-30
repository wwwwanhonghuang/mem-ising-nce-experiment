
#include "models/initializers/normal_initializer.hpp"

void NormalIsingModelInitializer::initialize(std::shared_ptr<IsingModel> ising_model) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> j_dist(_j_min, _j_max);
    std::normal_distribution<double> h_dist(_h_min, _h_max);

    for (int i = 0; i < ising_model->n_sites; ++i) {
        for (int j = 0; j < ising_model->n_sites; ++j) {
            double random_value = j_dist(gen);
            ising_model->J[i][j] = random_value;
        }
        ising_model->J[i][i] = 0.0;  // Zero diagonal
    }

    for (int i = 0; i < ising_model->n_sites; ++i) {
        ising_model->H[i] = h_dist(gen);
    }
}