#include "neuron_models/neuron.hpp"
#include "neuron_models/lif_neuron.hpp"
#include "neuron_models/possion_neuron.hpp"
#include "neuron_models/initializer.hpp"

#include "network/network.hpp"
#include "network/network_builder.hpp"

#include "synapse_models/synapse.hpp"

#include "simulator/snn_simulator.hpp"

#include "network/initializer/weight_initializers.hpp"

#include "recorder/recorder.hpp"
#include "recorder/connection_recorder.hpp"
#include "recorder/neuron_recorder.hpp"

#include "connections/all_to_all_conntection.hpp"
#include "recorder/recorder_function_examples.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <memory>
#include <yaml-cpp/yaml.h>


// TODO: add dynamical building of initializer and recorder.


struct neuron_info {
    double x;
    double y;
    double z;
    bool is_excitory = true;

    neuron_info(double x, double y, double z, bool is_excitory):x(x), y(y), z(z), is_excitory(is_excitory){}
};

double random_double(double min, double max, std::mt19937 &gen) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

std::vector<neuron_info> generate_neuron_infos(int N, double L, std::mt19937 &gen) {
    std::vector<neuron_info> neuron_infos;
    for (int i = 0; i < N; ++i) {
        double x = random_double(0, L, gen);
        double y = random_double(0, L, gen);
        double z = random_double(0, L, gen);
        neuron_infos.emplace_back(x, y, z, true);
    }
    return neuron_infos;
}

int sample_power_law(int k_min, int k_max, std::mt19937 &gen) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(gen);
    double k = std::pow((std::pow(k_min, -1.0) + u * (std::pow(k_max, -1.0) - std::pow(k_min, -1.0))), -1.0);
    return static_cast<int>(k);
}

std::vector<int> generate_outgoing_edges(int N, int k_min, int k_max, std::mt19937 &gen) {
    std::vector<int> outgoing_edges(N);
    for (int i = 0; i < N; ++i) {
        outgoing_edges[i] = sample_power_law(k_min, k_max, gen);
    }
    return outgoing_edges;
}

std::vector<double> get_normalized_connected_distribution(int neuron_id_source, int N, const std::vector<neuron_info>& neuron_infos,
    double lambda_r){
    // Compute distances and probabilities
    std::vector<double> distances(N);
    std::vector<double> probabilities(N);
    for (int neuron_id_target = 0; neuron_id_target < N; ++neuron_id_target) {
        if (neuron_id_source == neuron_id_target) {
            distances[neuron_id_target] = std::numeric_limits<double>::infinity();
            probabilities[neuron_id_target] = 0.0;
        } else {
            double dx = neuron_infos[neuron_id_target].x - neuron_infos[neuron_id_source].x;
            double dy = neuron_infos[neuron_id_target].y - neuron_infos[neuron_id_source].y;
            double dz = neuron_infos[neuron_id_target].z - neuron_infos[neuron_id_source].z;
            distances[neuron_id_target] = std::sqrt(dx * dx + dy * dy + dz * dz);
            probabilities[neuron_id_target] = std::exp(-lambda_r * distances[neuron_id_target]);
        }
    }

    // Normalize probabilities
    double sum_prob = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    for (double &p : probabilities) {
        p /= sum_prob;
    }
    return probabilities;
}

// Step 3: Assign targets based on p(r)
std::unordered_map<int, std::vector<std::pair<int, int>>>
    assign_connections(const std::vector<neuron_info> &neuron_infos,
        const std::vector<int> &outgoing_edges,
        double lambda_r,
        std::mt19937 &gen) {
    std::unordered_map<int, std::vector<std::pair<int, int>>> connection_reocrds;
    int N = neuron_infos.size();

    for (int i = 0; i < N; ++i) {
        int k = outgoing_edges[i];
        connection_reocrds[i] = std::vector<std::pair<int, int>>();

        std::vector<double> probabilities = get_normalized_connected_distribution(i, N, neuron_infos, lambda_r);

        // Sample targets based on probabilities
        std::discrete_distribution<int> target_dist(probabilities.begin(), probabilities.end());
        for (int edge = 0; edge < k; ++edge) {
            int target = target_dist(gen);
            connection_reocrds[i].emplace_back(std::make_pair(i, target));
        }
    }

    return connection_reocrds;
}

std::vector<int> select_inhibitory_neurons(int N, int n_inhibitory_neurons, std::mt19937 &gen){
    std::vector<int> items(N, 0);
    std::iota(items.begin(), items.end(), 0); // Fill with 0, 1, ..., N-1
    std::vector<int> result;
    if(n_inhibitory_neurons > N) return result;
    std::sample(items.begin(), items.end(), std::back_inserter(result), n_inhibitory_neurons, gen);
    return result;
}

int main(){
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    
    const size_t N = (size_t)(config["lif-simulation"]["N"].as<int>());
    const double rho = config["lif-simulation"]["rho"].as<double>();
    const int time_steps = config["lif-simulation"]["time_steps"].as<int>();
    const double dt = config["lif-simulation"]["dt"].as<double>();
    const double L = std::cbrt(N / rho);

    // Connect neurons
    const int max_k_out = N;
    double Z_k_out = 0.0;    
    const int k_min = 2;
    const int k_max = N;
    const double lambda_r = 1 / 5.0;

    std::shared_ptr<snnlib::NetworkBuilder> builder = std::make_shared<snnlib::NetworkBuilder>();

    std::random_device rd;
    std::mt19937 gen(rd());



    std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder =
        std::make_shared<snnlib::NeuronRecorder>();
    std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder =
        std::make_shared<snnlib::ConnectionRecorder>();

    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = 
                            std::make_shared<snnlib::RecorderFacade>();
    

    std::cout << "1. Generate neuron in L^3 space, L = " << L << std::endl;
    std::vector<neuron_info> neuron_infos = generate_neuron_infos(N, L, gen);
    for(size_t i = 0; i < neuron_infos.size(); i++){
        neuron_info& info = neuron_infos[i];
        std::cout << " - Neuron " << i + 1 << "'s Coordination: " << 
            info.x << ", " << info.y << ", " << info.z << std::endl;
    }

    // Generate outgoing edge counts
    std::cout << "2. Generate amount of outgoing edges" << std::endl;

    std::vector<int> outgoing_edges = generate_outgoing_edges(N, k_min, k_max, gen);
    for(size_t i = 0; i < outgoing_edges.size(); i++){
        int n_edges = outgoing_edges[i];
        std::cout << " - Amount of outgoing edges from neuron " << i + 1 << ": " << 
            n_edges << std::endl;
    }

    std::cout << "3. Generate connection topology. " << std::endl;
    // Assign connections
    std::unordered_map<int, std::vector<std::pair<int, int>>> connections = 
        assign_connections(neuron_infos, outgoing_edges, lambda_r, gen);

    for(size_t i = 0; i < N; i++){
        int neuron_id = i;
        if(connections.find(i) == connections.end()){
            // A neuron must has at least one entry edge.
            std::vector<double> probabilities = get_normalized_connected_distribution(i, N, neuron_infos, lambda_r);
            std::discrete_distribution<int> target_dist(probabilities.begin(), probabilities.end());
            int target = target_dist(gen);
            connections[i].emplace_back(std::make_pair(i, target));
            continue;
        }
        auto connection_record = connections[neuron_id];
        std::cout << " - Neuron " << neuron_id << ": " << std::endl;
        assert(outgoing_edges[neuron_id] == (int) connection_record.size());
        for(int edge_id = 0; edge_id < (int) connection_record.size(); edge_id++){
            std::cout   << "   - " << " " << connection_record[edge_id].first 
                        << " --> " << connection_record[edge_id].second
                        << std::endl;
        }
    }

    const double ratio_inh_neurons = 0.3;
    const int n_inh_neurons = (int)(ratio_inh_neurons * N);

    std::cout << "4. Select inhibitory neuron IDs, with inhibitory neuron ratio = " << 
        ratio_inh_neurons << std::endl;

    std::vector<int> inh_neuron_ids = select_inhibitory_neurons(N, n_inh_neurons, gen);
    for(size_t i = 0; i < inh_neuron_ids.size(); i++){
        int id = inh_neuron_ids[i];
        std::cout << " - inhibitory neuron ID = " << id << std::endl;
        neuron_infos[id].is_excitory = false;
    }

    std::cout << "5. Build " << N + 1 << " neurons (include 1 possion input neuron)" << std::endl;
      
    builder->build_neuron<snnlib::PossionNeuron>("possion_input_30hz", 10,
        snnlib::PossionNeuron::default_parameters()->set("freq", 30)
    );

    for(size_t i = 0; i < N; i++){
        std::string neuron_name = "neuron_" + std::to_string(i);
            
        builder->build_neuron<snnlib::LIFNeuron>(neuron_name, 200,
            snnlib::LIFNeuron::default_parameters()
                ->set("V_rest", -65.0)
                ->set("V_th", -40.0)
                ->set("V_reset", -60.0)
                ->set("tau_m",  1e-2)
                ->set("R",  1.0)
                ->set("t_ref",  5e-3)
                ->set("V_peak",  20))
            ->add_initializer("reset_potential_initializer");
         // recorder_facade->add_neuron_record_item(neuron_name, 
         // snnlib::recorder_function_examples::generate_membrane_potential_recorder(neuron_recorder));
        
        recorder_facade->add_neuron_record_item(neuron_name, 
            
                [neuron_recorder](const std::string& neuron_name, std::shared_ptr<snnlib::AbstractSNNNeuron> neuron, int t, double dt) -> void{
                    if((t - 1) * dt == neuron->x[1]){
                        neuron_recorder->record_membrane_potential_to_file(
                            std::string("data/logs/")  + neuron_name
                            + std::string(".spikes"), neuron, t
                        );
                    }
                }
            );
        
    }

    std::shared_ptr<snnlib::NormalWeightInitializer> initializer =
        std::make_shared<snnlib::NormalWeightInitializer>();

    std::normal_distribution<> d(0.0, 0.4);

    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);


    std::cout << "6. Build connections" << std::endl;
    for(size_t i = 0; i < N; i++){
        std::string neuron_name = "neuron_" + std::to_string(i);
        double random = std::clamp(std::abs(d(gen)), 0.0, 1.0);

        
        builder->build_synapse<snnlib::CurrentBasedKernalSynapse>("syn_possion_input_to_" + neuron_name, 
        "possion_input_30hz", neuron_name, "double_exponential", 
        snnlib::CurrentBasedKernalSynapse::default_parameters()
            ->set("tau_rise", 1e-2)
            ->set("tau_decay", 1e-2))
        ->build_connection<snnlib::AllToAllConnection>("conn_possion_input_to_" + neuron_name, builder)
        ->add_initializer(std::make_shared<snnlib::IdenticalWeightInitializer>(random));
        
    }
    for(auto connection : connections){
        std::unordered_map<int, int> target_count;

        for(std::pair<int, int> connection_pair : connection.second){
            int neuron_id_source = connection_pair.first;
            int neuron_id_target = connection_pair.second;
            std::string synapse_name = std::string("synapse_") 
               + std::to_string(neuron_id_source)
               + std::string("_")
               + std::to_string(neuron_id_target)
               + std::string("_")
               + std::to_string(target_count[neuron_id_target]);

            std::string connection_name = std::string("connection_") 
               + std::to_string(neuron_id_source)
               + std::string("_")
               + std::to_string(neuron_id_target)
               + std::string("_")
               + std::to_string(target_count[neuron_id_target]);


            std::string pre_synapse_neuron_name = "neuron_" + std::to_string(neuron_id_source);
            std::string post_synapse_neuron_name = "neuron_" + std::to_string(neuron_id_target);

            double random = std::clamp(std::abs(d(gen)), 0.0, 1.0);
            if (uniform_dist(gen) < 0.25) {
                random = 0.0;
            }
            builder->build_synapse<snnlib::CurrentBasedKernalSynapse>(synapse_name, 
                pre_synapse_neuron_name, post_synapse_neuron_name, "double_exponential", 
            snnlib::CurrentBasedKernalSynapse::default_parameters()
                ->set("tau_rise", 1e-2)
                ->set("tau_decay", 1e-2))
            ->build_connection<snnlib::AllToAllConnection>(connection_name, builder)
            ->add_initializer(std::make_shared<snnlib::IdenticalWeightInitializer>(random * 
                (neuron_infos[neuron_id_source].is_excitory ? 1: -1)));
            
            recorder_facade->
                add_connection_record_item(connection_name, snnlib::recorder_function_examples::generate_weight_recorder());

            target_count[neuron_id_target]++;
        }        
    }
    std::cout << "7. Build network" << std::endl;

    std::shared_ptr<snnlib::SNNNetwork> network = builder->build_network();

    snnlib::SNNSimulator simulator;
    simulator.simulate(network, time_steps, dt, recorder_facade);

    return 0;
}
