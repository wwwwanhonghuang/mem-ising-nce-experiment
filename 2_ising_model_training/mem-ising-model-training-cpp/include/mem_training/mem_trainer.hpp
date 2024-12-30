#ifndef MEM_TRAINER_HPP
#define MEM_TRAINER_HPP
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include "models/ising_model.hpp"
#include "utils/ising_model_utils.hpp"
#include "inferencer/inferencer.hpp"

std::vector<long double> calculate_observation_essembly_average_si(const std::vector<int>& observation_configurations, std::shared_ptr<IsingModel> ising_model);

std::vector<long double> calculate_dynamical_observation_essembly_average_si(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model);


std::vector<std::vector<long double>> calculate_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model);


std::vector<long double> calculate_model_proposed_essembly_average_si(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer);
std::vector<std::vector<long double>> calculate_model_proposed_essembly_average_si_sj(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer);

std::vector<std::vector<long double>> calculate_dynamical_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model);

struct IsingMEMTrainer{
    private:
    std::shared_ptr<IsingModel> ising_model;
    std::shared_ptr<IsingInferencer> ising_model_inferencer;
    std::vector<long double> buffer_beta_H;
    std::vector<std::vector<long double>> buffer_beta_J;
    bool require_evaluation;
    const std::vector<int>& train_configurations;
    const std::vector<int>& observation_configurations;
    double alpha;
    std::unordered_map<int, long double> observation_configuration_possibility_map;
    int n_configurations;
    
    long double clip_threshold = 0.1;
    long double clip_gradient(long double grad) {
        if(grad < -clip_threshold) return -clip_threshold;
        if(grad > clip_threshold) return clip_threshold;
        return grad;
    }
    
    bool is_dynamical = false;
    
    public:
    void set_dynamical_version(bool value){
        this->is_dynamical = value;
    }
    
    IsingMEMTrainer(std::shared_ptr<IsingModel> ising_model, 
                    std::shared_ptr<IsingInferencer> inferencer, 
                    const std::vector<int>& train_configurations, 
                    const std::vector<int>& observation_configurations,
                    double alpha, bool require_evaluation, long double clip_threshold);

    void prepare_training();

    void update_model_parameters();

    void update_model_partition_functions();

    long double evaluation();

    void gradient_descending_step();

    void scale_parameters(double max_norm_J, double max_norm_H);
};

#endif
