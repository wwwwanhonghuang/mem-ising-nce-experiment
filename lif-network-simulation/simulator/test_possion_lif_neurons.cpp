
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

void test(){

    std::shared_ptr<snnlib::NetworkBuilder> builder = std::make_shared<snnlib::NetworkBuilder>();

    
    builder->build_neuron<snnlib::PossionNeuron>("possion_input_30hz", 10,
        snnlib::PossionNeuron::default_parameters()->set("freq", 30)
    );

    builder->build_neuron<snnlib::LIFNeuron>("lif_neuron", 10,
            snnlib::LIFNeuron::default_parameters()
                ->set("V_rest", -65.0)
                ->set("V_th", -40.0)
                ->set("V_reset", -60.0)
                ->set("tau_m",  1e-2)
                ->set("R",  1.0)
                ->set("t_ref",  5e-3)
                ->set("V_peak",  20))
    ->add_initializer("reset_potential_initializer");

    std::vector<double> weights(100, 0.0);
    std::vector<double> single_post_neuron_connection_weight{ 
                0.35281047,  0.08003144,  0.1957476 ,  0.44817864,  0.3735116 ,
               -0.19545558,  0.19001768, -0.03027144, -0.02064377,  0.0821197
    };
    for(int pre_id = 0; pre_id < 10; pre_id++){
        for(int post_id = 0; post_id < 10; post_id++){
            weights[pre_id * 10 + post_id] = single_post_neuron_connection_weight[pre_id];
        }
    }
    
    builder->build_synapse<snnlib::CurrentBasedKernalSynapse>("double_exponential_synapse", 
        "possion_input_30hz", "lif_neuron", "double_exponential", 
        snnlib::CurrentBasedKernalSynapse::default_parameters()
            ->set("tau_rise", 1e-2)
            ->set("tau_decay", 1e-2))
        ->build_connection<snnlib::AllToAllConnection>("full_connection_1", builder)
        ->add_initializer(std::make_shared<snnlib::SpecificWeightInitializer>(
            weights));
    
    // Build Recorder
    std::shared_ptr<snnlib::NeuronRecorder> neuron_recorder =
        std::make_shared<snnlib::NeuronRecorder>();
    std::shared_ptr<snnlib::ConnectionRecorder> connection_recorder =
        std::make_shared<snnlib::ConnectionRecorder>();

    std::shared_ptr<snnlib::RecorderFacade> recorder_facade = 
                            std::make_shared<snnlib::RecorderFacade>();
    recorder_facade->add_connection_record_item("full_connection_1", 
        snnlib::recorder_function_examples::generate_weight_recorder());

    recorder_facade->add_connection_record_item("full_connection_1", 
        snnlib::recorder_function_examples::generate_response_recorder(connection_recorder));

    recorder_facade->add_neuron_record_item("possion_input_30hz", 
        snnlib::recorder_function_examples::generate_membrane_potential_recorder(neuron_recorder));
    recorder_facade->add_neuron_record_item("lif_neuron",
        snnlib::recorder_function_examples::generate_membrane_potential_recorder(neuron_recorder));

    std::shared_ptr<snnlib::SNNNetwork> network = builder->build_network();
    snnlib::SNNSimulator simulator;
    simulator.simulate(network, 10000, 1e-4, recorder_facade);

}


int main(){
    test();
    return 0;
}