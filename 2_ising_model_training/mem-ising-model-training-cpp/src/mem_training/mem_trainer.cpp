#include "mem_training/mem_trainer.hpp"
#include "utils/ising_model_utils.hpp"
#include <cmath>
#include <omp.h>
#include <cassert>
#define PLUS_EPISLON_SMOOTHING(value) (value == 0 ? 1e-39 : value)


#include <numeric>

void IsingMEMTrainer::scale_parameters(double max_norm_J, double max_norm_H) {
    long double J_min = INFINITY;
    long double J_max = -INFINITY;
    for (int i = 0; i < ising_model->n_sites; i++) {
        for (int j = 0; j < ising_model->n_sites; j++) {
            if(ising_model->J[i][j] < J_min) J_min = ising_model->J[i][j];
            if(ising_model->J[i][j] > J_min) J_max = ising_model->J[i][j];

        }
    }
    for (int i = 0; i < ising_model->n_sites; i++) {
        for (int j = 0; j < ising_model->n_sites; j++) {
            ising_model->J[i][j] /= (ising_model->J[i][j] - J_min) / (J_max- J_min);
        }
    }

    long double H_min = INFINITY;
    long double H_max = -INFINITY;
    for (int i = 0; i < ising_model->n_sites; i++) {
        if(ising_model->H[i] < H_min) H_min = ising_model->H[i];
        if(ising_model->H[i] > H_min) H_max = ising_model->H[i];
    }

    for (int i = 0; i < ising_model->n_sites; i++) {
        ising_model->H[i] /= (ising_model->H[i] - H_min) / (H_max - H_min);
    }
    
}

std::vector<long double> calculate_observation_essembly_average_si(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){
    std::vector<long double> essembly_average(ising_model->n_sites, 0);
    
    for(int configuration : observation_configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        for(int i = 0; i < v.size(); i++){
            essembly_average[i] += v[i];
        }
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        essembly_average[i] /= observation_configurations.size();
    }

    return essembly_average;
};

std::vector<std::vector<long double>> calculate_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){
    if (ising_model->n_sites <= 0) {
        std::cerr << "Error: Invalid value of n_sites: " << ising_model->n_sites << std::endl;
    }

    std::vector<std::vector<long double>> 
        essembly_average(ising_model->n_sites, std::vector<long double>(ising_model->n_sites, 0.0));

    for(int configuration : observation_configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        for(int i = 0; i < ising_model->n_sites; i++){
            for(int j = 0; j < ising_model->n_sites; j++){
                if(v[i] == 1 && v[j] == 1) essembly_average[i][j] += 1;
            }
        }
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
            essembly_average[i][j] /= observation_configurations.size();
        }
    }

    return essembly_average;
};


std::vector<long double> calculate_dynamical_observation_essembly_average_si(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){

    std::vector<long double> essembly_average(ising_model->n_sites, 0);
    
    int n_time_samples = observation_configurations.size();
    for(int t = 0; t < n_time_samples - 1; t++){
        int configuration = observation_configurations[t];
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        for(int i = 0; i < v.size(); i++){
            essembly_average[i] += v[i];
        }
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        essembly_average[i] /= (observation_configurations.size() - 1);
    }

    return essembly_average;
}


std::vector<std::vector<long double>> calculate_dynamical_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){
    int total_time_step = observation_configurations.size();

    if (ising_model->n_sites <= 0) {
        std::cerr << "Error: Invalid value of n_sites: " << ising_model->n_sites << std::endl;
    }

    std::vector<std::vector<long double>> 
        essembly_average(ising_model->n_sites, std::vector<long double>(ising_model->n_sites, 0.0));

    for(int t = 0; t < total_time_step - 1; t++){
        std::vector<char> configuration_t = to_binary_representation(ising_model->n_sites, observation_configurations[t]);
        std::vector<char> configuration_t_plus_1 = to_binary_representation(ising_model->n_sites, observation_configurations[t + 1]);
        for(int i = 0; i < ising_model->n_sites; i++){
            for(int j = 0; j < ising_model->n_sites; j++){
                if(configuration_t[i] == 1 && configuration_t_plus_1[j] == 1) 
                    essembly_average[i][j] += 1;
            }
        }

    }

    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
            essembly_average[i][j] /= (observation_configurations.size() - 1);
        }
    }
    return essembly_average;
};

std::vector<long double> calculate_model_proposed_essembly_average_si(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer){
    int n_sites = ising_model->n_sites;
    std::vector<long double> essembly_average(n_sites, 0.0);

    int num_configs = configurations.size();

    #pragma omp parallel
    {
        std::vector<long double> local_essembly_average(n_sites, 0.0);

        #pragma omp for
        for (int idx = 0; idx < num_configs; idx++) {
            int configuration = configurations[idx];
            std::vector<char> v = to_binary_representation(n_sites, configuration);
            long double possibility = ising_inferencer->calculate_configuration_possibility(ising_model, v);
            assert(!std::isnan(possibility));
            for (int i = 0; i < n_sites; i++) {
                if (v[i] != 0) {
                    local_essembly_average[i] += possibility;
                }
            }
        }

        // Safely accumulate the local results into the global vector
        #pragma omp critical
        {
            for (int i = 0; i < n_sites; i++) {
                essembly_average[i] += local_essembly_average[i];
            }
        }
    }

    return essembly_average;
};

std::vector<std::vector<long double>> calculate_model_proposed_essembly_average_si_sj(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer){
    int n_sites = ising_model->n_sites;
    std::vector<std::vector<long double>> essembly_average(n_sites, std::vector<long double>(n_sites, 0.0));

    int num_configs = configurations.size();

    #pragma omp parallel
    {
        std::vector<std::vector<long double>> local_essembly_average(n_sites, std::vector<long double>(n_sites, 0.0));

        #pragma omp for
        for (int idx = 0; idx < num_configs; idx++) {
            int configuration = configurations[idx];
            std::vector<char> v = to_binary_representation(n_sites, configuration);
            long double possibility = ising_inferencer->calculate_configuration_possibility(ising_model, v);
            assert(!std::isnan(possibility));

            for (int i = 0; i < n_sites; i++) {
                for (int j = 0; j < n_sites; j++) {
                    if (v[i] == 1 && v[j] == 1) {
                        local_essembly_average[i][j] += possibility;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < n_sites; i++) {
                for (int j = 0; j < n_sites; j++) {
                    essembly_average[i][j] += local_essembly_average[i][j];
                }
            }
        }
    }

    return essembly_average;
};

void IsingMEMTrainer::update_model_parameters(){
    for (int i = 0; i < ising_model->n_sites; i++) {
        long double delta = buffer_beta_H[i];
        long double grad = clip_gradient(delta);
        ising_model->H[i] += grad;
        assert(!std::isnan(grad));

    }

    for (int i = 0; i < ising_model->n_sites; i++) {
        for (int j = 0; j < ising_model->n_sites; j++) {
            long double delta = clip_gradient(buffer_beta_J[i][j]);
            long double grad = clip_gradient(delta);
            if(std::isnan(grad)){
                std::cout << grad << " clip from [" << buffer_beta_J[i][j] << "]" <<  std::endl;
            }
            assert(!std::isnan(grad));

            ising_model->J[i][j] += grad;
        }
    }
};

void IsingMEMTrainer::update_model_partition_functions(){
    ising_model_inferencer->update_partition_function(ising_model, train_configurations, require_evaluation);
};



inline long double laplace_smoothing(long double p, int n_configurations){
    const long double e = 1e-39;
    return (p + e) / (e * n_configurations + 1);
}

inline long double laplace_smoothing(long double x, int N, int d, double alpha){
    return (x + alpha) / (N + alpha * d);
}
long double IsingMEMTrainer::evaluation(){
    long double S1 = 0.0; // order-1 entropy
    long double S2 = 0.0; // order-2 entropy
    long double SN = 0.0; // empirical entropy
    long double D1 = 0.0;
    long double D2 = 0.0;

    #pragma omp parallel for reduction(+:S1, S2, SN)
    for(int configuration : train_configurations){
        long double p2 = 
            ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 2);
        long double p1 = 
            ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 1);
        p2 = p2 + 1e-39; // laplace_smoothing(p2, n_configurations);
        p1 = p1 + 1e-39; // laplace_smoothing(p1, n_configurations);
        S2 += -p2 * std::log(p2);
        S1 += -p1 * std::log(p1);

        long double p_observation = 0.0;
        if(observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end()){
            p_observation = observation_configuration_possibility_map[configuration];
        }
        p_observation = p_observation + 1e-39; // laplace_smoothing(p_observation, n_configurations);
        SN += -p_observation * std::log(p_observation);
    }

    long double r_s = (S1 - S2) / (S1 - SN); 
    // r_s measures how well the model captures the higher-order 
    // dependencies (order-2) compared to lower-order dependencies (order-1) 
    // relative to the empirical entropy:
    // If r_s is close to 1: The model’s higher-order terms significantly improve its fit to the data.
    /*
        If r_s is close to 1: The model’s higher-order terms significantly improve its fit to the data.
        If r_s is close to 0: Higher-order terms provide no improvement.
        If r_s is negative: The higher-order model is performing worse than the lower-order model.
    */
    
    #pragma omp parallel for reduction(+:D1, D2)
    for(int configuration : train_configurations){
        long double obs_possibility = 0.0;
        if(observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end()){
            obs_possibility = observation_configuration_possibility_map[configuration];
        }
        obs_possibility = obs_possibility + 1e-39; // laplace_smoothing(obs_possibility, n_configurations);

        long double model_possibility_order_1 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 1);
        long double model_possibility_order_2 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 2);
        model_possibility_order_1 = model_possibility_order_1 + 1e-39; // laplace_smoothing(model_possibility_order_1, n_configurations);
        model_possibility_order_2 = model_possibility_order_2 + 1e-39; // laplace_smoothing(model_possibility_order_2, n_configurations);

        D1 += obs_possibility * std::log(obs_possibility / model_possibility_order_1);
        D2 += obs_possibility * std::log(obs_possibility / model_possibility_order_2);
    }

    long double r_d = (D1 - D2) / D1;
    std::cout << "r_d = " << r_d << std::endl;
    std::cout << "Reliablity: r_s / r_d = " << r_s / r_d << std::endl;
    return r_s / r_d;
};


void print_si(const std::vector<long double>& si){
    for(int i = 0; i < si.size(); i++){
        std::cout << si[i] << " ";
    }
    std::cout << std::endl;
}

void print_sisj(const std::vector<std::vector<long double>>& sisj){
    for(int i = 0; i < sisj.size(); i++){
        const std::vector<long double>& row = sisj[i];
        for(int j = 0; j < row.size(); j++){
            std::cout << sisj[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void IsingMEMTrainer::gradient_descending_step(){
    std::cout << "      - begin gradient_descending_step()" << std::endl;

    std::cout << "        - calculate obs_essembly_avgerage_si" << std::endl;
    std::vector<long double> obs_essembly_avgerage_si = 
        this->is_dynamical ? 
        calculate_dynamical_observation_essembly_average_si(observation_configurations, ising_model):
        calculate_observation_essembly_average_si(observation_configurations, ising_model);
    // print_si(obs_essembly_avgerage_si);

    std::cout << "        - calculate obs_essembly_avgerage_si_sj" << std::endl;
    std::vector<std::vector<long double>> obs_essembly_avgerage_si_sj = 
        this->is_dynamical ?
        calculate_dynamical_observation_essembly_average_si_sj(observation_configurations, ising_model) :
        calculate_observation_essembly_average_si_sj(observation_configurations, ising_model);
    // print_sisj(obs_essembly_avgerage_si_sj);

    std::cout << "        - calculate model_essembly_avgerage_si" << std::endl;
    std::vector<long double> model_essembly_avgerage_si = 
        calculate_model_proposed_essembly_average_si(train_configurations, 
            ising_model, ising_model_inferencer);
    // print_si(model_essembly_avgerage_si);

    std::cout << "        - calculate model_essembly_avgerage_si_sj" << std::endl;
    std::vector<std::vector<long double>> model_essembly_avgerage_si_sj = 
        calculate_model_proposed_essembly_average_si_sj(train_configurations, 
            ising_model, ising_model_inferencer);
    // print_sisj(model_essembly_avgerage_si_sj);

    const long double lambda = 1e-19L;
    int N = observation_configurations.size();

    // calculate delta H, and update H
    for(int i = 0; i < ising_model->n_sites; i++){
        long double smoothed_observation_value = PLUS_EPISLON_SMOOTHING(obs_essembly_avgerage_si[i]); // obs_essembly_avgerage_si[i] == 0 ? 1e-39 : obs_essembly_avgerage_si[i]; // laplace_smoothing(obs_essembly_avgerage_si[i], N, 1 << ising_model->n_sites, 1e-39);
        long double model_value = model_essembly_avgerage_si[i]; // == 0 ? 1e-39 : model_essembly_avgerage_si[i];  // laplace_smoothing(model_essembly_avgerage_si[i], 1 << ising_model->n_sites, 1 << ising_model->n_sites, 1e-39);
        long double delta = alpha * std::log(smoothed_observation_value / model_value); 
            buffer_beta_H[i] = delta;
    }

    // calculate delta J, and update J
    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
		long double smoothed_observation_value = PLUS_EPISLON_SMOOTHING(obs_essembly_avgerage_si_sj[i][j]); //  == 0 ? 1e-39 : 0; // laplace_smoothing(obs_essembly_avgerage_si_sj[i][j], N, 1 << ising_model->n_sites, 1e-39);
        long double model_value = model_essembly_avgerage_si_sj[i][j]; // == 0 ? 1e-39 : 0; // laplace_smoothing(model_essembly_avgerage_si_sj[i][j], 1 << ising_model->n_sites, 1 << ising_model->n_sites, 1e-39);
        long double delta = alpha * 
                std::log(smoothed_observation_value / model_value); 
            buffer_beta_J[i][j] = delta;
        }
    }
}

IsingMEMTrainer::IsingMEMTrainer(std::shared_ptr<IsingModel> ising_model, 
                    std::shared_ptr<IsingInferencer> inferencer, 
                    const std::vector<int>& train_configurations, 
                    const std::vector<int>& observation_configurations,
                    double alpha, bool require_evaluation, long double clip_threshold)
        : ising_model(ising_model), 
          ising_model_inferencer(inferencer), 
          train_configurations(train_configurations), 
          observation_configurations(observation_configurations),
          alpha(alpha), 
          require_evaluation(require_evaluation), clip_threshold(clip_threshold) {
        
    buffer_beta_H.resize(ising_model->n_sites, 0.0);
    buffer_beta_J.resize(ising_model->n_sites, 
        std::vector<long double>(ising_model->n_sites, 0.0));
    
    // Build observation frequency map.
    int n_time_samples = observation_configurations.size();

    for(int t = 0; t < n_time_samples - 1; t++){
        int configuration = observation_configurations[t];
        observation_configuration_possibility_map[configuration]++;
    }

    for(auto& configuration : observation_configuration_possibility_map){
        observation_configuration_possibility_map[configuration.first] /= observation_configurations.size();
    }

    n_configurations = (1 << ising_model->n_sites);
}

void IsingMEMTrainer::prepare_training(){
    std::cout << "prepare trainning.." << std::endl;
    std::cout << "  - prepare partition functions.." << std::endl;
    ising_model_inferencer->update_partition_function(ising_model, train_configurations, require_evaluation);
    std::cout << "  - partition functions Z1 = " << ising_model_inferencer->get_Z(1) << std::endl;
    std::cout << "  - partition functions Z2 = " << ising_model_inferencer->get_Z(2) << std::endl;
}