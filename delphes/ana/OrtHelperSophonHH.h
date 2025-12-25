#ifndef OrtHelperSophonHH_h
#define OrtHelperSophonHH_h

#include <algorithm>
#include <iostream>

#include "ONNXRuntime.h"

class OrtHelperSophonHH {
public:
    OrtHelperSophonHH(std::string model_path, bool debug = false) {
        ort_ = std::make_unique<myOrt::ONNXRuntime>(model_path);
        debug_ = debug;
        init_data();
    }
    ~OrtHelperSophonHH() {}

    void infer_model(std::map<std::string, std::vector<float>*>& particleVars, std::map<std::string, float>& globalVars) {

        // Extract input and perform preprocessing
        make_input(particleVars, globalVars);

        // Inference via onnxruntime
        output_ = ort_->run(input_names_, data_, input_shapes_)[0];
        if (debug_) {
            std::cout << "model output (size = " << output_.size() << "):\n";
            for (auto v: output_) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<float>& get_output() {
        return output_;
    }

private:
    std::unique_ptr<myOrt::ONNXRuntime> ort_ = nullptr;
    std::vector<std::string> input_names_ = {"pf_features", "pf_vectors", "pf_mask"};
    std::vector<std::vector<int64_t>> input_shapes_ = {{1, 19, 256}, {1, 4, 256}, {1, 1, 256}}; // (batch_size=1, channel, length)
    std::vector<std::vector<std::tuple<std::string, float, float, float, float>>> input_var_info_ = {
        // (name, subtract_val, multiply_val, clip_min, clip_max)
        {
            {"part_pt_log", 1.7, 0.7, -5, 5},
            {"part_e_log", 2.0, 0.7, -5, 5},
            {"part_logptrel", -4.7, 0.7, -5, 5},
            {"part_logerel", -4.7, 0.7, -5, 5},
            {"part_deltaR", 0.2, 4.0, -5, 5},
            {"part_charge", 0, 1, -1e8, 1e8},
            {"part_isChargedHadron", 0, 1, -1e8, 1e8},
            {"part_isNeutralHadron", 0, 1, -1e8, 1e8},
            {"part_isPhoton", 0, 1, -1e8, 1e8},
            {"part_isElectron", 0, 1, -1e8, 1e8},
            {"part_isMuon", 0, 1, -1e8, 1e8},
            {"part_d0", 0, 1, -1e8, 1e8},
            {"part_d0err", 0, 1, 0, 1},
            {"part_dz", 0, 1, -1e8, 1e8},
            {"part_dzerr", 0, 1, 0, 1},
            {"part_deta", 0, 1, -1e8, 1e8},
            {"part_dphi", 0, 1, -1e8, 1e8},
            {"part_eta", 0, 1, -1e8, 1e8},
            {"part_phi", 0, 1, -1e8, 1e8}
        },
        {
            {"part_px", 0, 1, -1e8, 1e8},
            {"part_py", 0, 1, -1e8, 1e8},
            {"part_pz", 0, 1, -1e8, 1e8},
            {"part_energy", 0, 1, -1e8, 1e8}
        },
        {
            {"part_mask", 0, 1, -1e8, 1e8}
        }
    };
    std::map<std::string, std::vector<float>> input_feats_;
    std::vector<std::vector<float>> data_;
    std::vector<float> output_;
    bool debug_ = false;

    void init_data() {
        // initialize the data_ vector
        for (size_t i = 0; i < input_names_.size(); i++) {
            data_.emplace_back(input_shapes_[i][1] * input_shapes_[i][2], 0);
        }
        // initialize input_feats_
        for (auto v: std::vector<std::string>({"part_eta", "part_phi", "part_deta", "part_dphi", "part_charge", "part_d0err", "part_dzerr", "part_px", "part_py", "part_pz", "part_energy", "part_pt", "part_pt_log", "part_e_log", "part_logptrel", "part_logerel", "part_deltaR", "part_d0", "part_dz", "part_isElectron", "part_isMuon", "part_isPhoton", "part_isChargedHadron", "part_isNeutralHadron", "part_mask"})) {
            input_feats_[v] = std::vector<float>();
        }
    }

    void make_input(std::map<std::string, std::vector<float>*>& particleVars, std::map<std::string, float>& globalVars) {
        // make inputs for ParT with non-scaled features

        for (auto &v: input_feats_) {
            v.second.clear();
        }

        // fill input_feats_
        // existing features
        input_feats_["part_eta"].assign(particleVars["part_eta"]->begin(), particleVars["part_eta"]->end());
        input_feats_["part_phi"].assign(particleVars["part_phi"]->begin(), particleVars["part_phi"]->end());
        input_feats_["part_deta"].assign(particleVars["part_deta"]->begin(), particleVars["part_deta"]->end());
        input_feats_["part_dphi"].assign(particleVars["part_dphi"]->begin(), particleVars["part_dphi"]->end());
        input_feats_["part_charge"].assign(particleVars["part_charge"]->begin(), particleVars["part_charge"]->end());
        input_feats_["part_d0err"].assign(particleVars["part_d0err"]->begin(), particleVars["part_d0err"]->end());
        input_feats_["part_dzerr"].assign(particleVars["part_dzerr"]->begin(), particleVars["part_dzerr"]->end());
        input_feats_["part_px"].assign(particleVars["part_px"]->begin(), particleVars["part_px"]->end());
        input_feats_["part_py"].assign(particleVars["part_py"]->begin(), particleVars["part_py"]->end());
        input_feats_["part_pz"].assign(particleVars["part_pz"]->begin(), particleVars["part_pz"]->end());
        input_feats_["part_energy"].assign(particleVars["part_energy"]->begin(), particleVars["part_energy"]->end());

        for (size_t i = 0; i < particleVars["part_px"]->size(); i++) {
            // calculating new features
            input_feats_["part_mask"].push_back(1);

            input_feats_["part_pt"].push_back(std::hypot(particleVars["part_px"]->at(i), particleVars["part_py"]->at(i)));
            input_feats_["part_pt_log"].push_back(std::log(input_feats_["part_pt"].at(i)));
            input_feats_["part_e_log"].push_back(std::log(particleVars["part_energy"]->at(i)));
            input_feats_["part_logptrel"].push_back(std::log(input_feats_["part_pt"].at(i) / globalVars["pfcand_sum_pt"]));
            input_feats_["part_logerel"].push_back(std::log(particleVars["part_energy"]->at(i) / globalVars["pfcand_sum_energy"]));
            input_feats_["part_deltaR"].push_back(std::hypot(particleVars["part_deta"]->at(i), particleVars["part_dphi"]->at(i)));
            input_feats_["part_d0"].push_back(std::tanh(particleVars["part_d0val"]->at(i)));
            input_feats_["part_dz"].push_back(std::tanh(particleVars["part_dzval"]->at(i)));
            input_feats_["part_isElectron"].push_back(particleVars["part_pid"]->at(i) == 11 || particleVars["part_pid"]->at(i) == -11);
            input_feats_["part_isMuon"].push_back(particleVars["part_pid"]->at(i) == 13 || particleVars["part_pid"]->at(i) == -13);
            input_feats_["part_isPhoton"].push_back(particleVars["part_pid"]->at(i) == 22);
            input_feats_["part_isChargedHadron"].push_back(particleVars["part_charge"]->at(i) != 0 && !input_feats_["part_isElectron"].at(i) && !input_feats_["part_isMuon"].at(i));
            input_feats_["part_isNeutralHadron"].push_back(particleVars["part_charge"]->at(i) == 0 && !input_feats_["part_isPhoton"].at(i));
        }

        // reset data_ to all zeros
        for (size_t i = 0; i < input_names_.size(); i++) {
            data_[i].assign(data_[i].size(), 0);
        }
        // construct the input data_
        for (size_t i = 0; i < input_names_.size(); i++) { // loop over input names
            for (int j = 0; j < input_shapes_[i][1]; j++) { // loop over channels

                auto name = std::get<0>(input_var_info_[i][j]);
                auto subtract_val = std::get<1>(input_var_info_[i][j]);
                auto multiply_val = std::get<2>(input_var_info_[i][j]);
                auto clip_min = std::get<3>(input_var_info_[i][j]);
                auto clip_max = std::get<4>(input_var_info_[i][j]);
                float val;

                int len = std::min((int)input_shapes_[i][2], (int)input_feats_[name].size());
                for (auto l = 0; l < len; l++) { // loop over particle length
                    data_[i][j * input_shapes_[i][2] + l] = std::clamp((input_feats_[name][l] - subtract_val) * multiply_val, clip_min, clip_max);
                }
            }
            if (debug_) {
                std::cout << "input: " << input_names_[i] << ":\n";
                for (int j = 0; j < input_shapes_[i][1]; j++) {
                    std::cout << "> var: " << std::get<0>(input_var_info_[i][j]) << ":\n";
                    for (int k = 0; k < input_shapes_[i][2]; k++) {
                        std::cout << data_[i][j * input_shapes_[i][2] + k] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

    }
};

#endif