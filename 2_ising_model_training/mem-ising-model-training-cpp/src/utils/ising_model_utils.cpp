#include <vector>
#include "utils/ising_model_utils.hpp"

/* Method from a 32-bit integer encoded ising spin configuration into a bit vector */
std::vector<char> to_binary_representation(int n_bits, int configuration){
    std::vector<char> binary_representation(n_bits, 0);
    int i = 0;
    while(configuration && i < n_bits){
        binary_representation[i++] = (char)(configuration & 1);
        configuration >>= 1;
    }
    return binary_representation;
};

