#ifndef ISING_MODEL_UTILS_HPP
#define ISING_MODEL_UTILS_HPP
#include <vector>
#include <memory>
#include "models/ising_model.hpp"

std::vector<char> to_binary_representation(int n_bits, int configuration);
#endif