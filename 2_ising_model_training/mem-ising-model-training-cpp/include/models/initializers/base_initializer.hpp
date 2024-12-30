#ifndef BASE_INITIALIZER_HPP
#define BASE_INITIALIZER_HPP
#include <memory>
#include "models/ising_model.hpp"

class BaseIsingModelInitializer {
public:
    virtual void initialize(std::shared_ptr<IsingModel> ising_model) = 0;
    virtual ~BaseIsingModelInitializer() {}
};
#endif
