#include <iostream>
#include <unordered_set>
#include <utility>
#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>

#include "EventData.h"
#include "OrtHelperSophonHH.h"

// #ifdef __CLING__
// R__LOAD_LIBRARY(libDelphes)
// #include "classes/DelphesClasses.h"
// #include "external/ExRootAnalysis/ExRootTreeReader.h"
// #else
// class ExRootTreeReader;
// #endif


void openFile(const std::string& filePath, TFile*& file, TTree*& tree, EventData& data) {
    file = TFile::Open(filePath.c_str(), "READ");
    if (!file || file->IsZombie()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    tree = nullptr;
    file->GetObject("tree", tree);
    if (!tree) {
        file->Close();
        throw std::runtime_error("Failed to get TTree from file: " + filePath);
    }
    data.setBranchAddresses(tree);
}


void closeFile(TFile*& file) {
    file->Close();
    file = nullptr;
}


void createOutputFile(const std::string& filePath, TFile*& file, TTree*& tree, EventData& data) {
    file = TFile::Open(filePath.c_str(), "RECREATE");
    if (!file || file->IsZombie()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    // prepare the file with LZ4 compression for fast loading
    file->SetCompressionAlgorithm(ROOT::kLZ4);
    file->SetCompressionLevel(4);

    tree = new TTree("tree", "tree");
    data.setOutputBranch(tree);
}


void writeOutputFile(TFile*& file) {
    file->Write();
    file->Close();
    file = nullptr;
}

int calculateClsIndex(float gen_higgs1_mass, float gen_higgs2_mass, int process_index) {
    if ((gen_higgs1_mass < 40) || (gen_higgs1_mass > 200) || (gen_higgs2_mass < 40) || (gen_higgs2_mass > 200)) {
        return (process_index == 0) ? 136 : 137;
    }
    
    // Sort masses to handle symmetric cases
    float mass1 = std::min(gen_higgs1_mass, gen_higgs2_mass);
    float mass2 = std::max(gen_higgs1_mass, gen_higgs2_mass);
    
    // Define mass bins
    std::vector<std::pair<float, float>> bins = {
        {40, 50}, {50, 60}, {60, 70}, {70, 80}, {80, 90}, {90, 100},
        {100, 110}, {110, 120}, {120, 130}, {130, 140}, {140, 150},
        {150, 160}, {160, 170}, {170, 180}, {180, 190}, {190, 200}
    };
    
    int bin1 = -1, bin2 = -1;
    for (int i = 0; i < bins.size(); i++) {
        if (mass1 >= bins[i].first && mass1 <= bins[i].second) bin1 = i;
        if (mass2 >= bins[i].first && mass2 <= bins[i].second) bin2 = i;
    }
    
    if (bin1 == -1 || bin2 == -1) return (process_index == 0) ? 136 : 137;
    
    // Calculate index based on bin positions
    int index = 0;
    for (int i = 0; i <= bin1; i++) {
        for (int j = i; j < bins.size(); j++) {
            if (i == bin1 && j == bin2) return index;
            index++;
        }
    }
    
    return (process_index == 0) ? 136 : 137;
}

//------------------------------------------------------------------------------
// This macro aims to process ntuples to ntuples with the SophonHH model inferenced

void InferSophonHH(TString inputFile, TString outputFile, TString modelPath, bool debug = false) {
    // gSystem->Load("libDelphes");

    // define branches
    std::vector<std::pair<std::string, std::string>> branchListIn = {
        {"pass_selection", "int"},
        {"pass_4j3b_selection", "int"},
        {"pass_4j2b_selection", "int"},
        {"pass_boosted_trigger", "int"},
        {"HT", "float"},
        {"pfcand_sum_mass", "float"},
        {"pfcand_sum_HT", "float"},
        {"pfcand_sum_pt", "float"},
        {"pfcand_sum_eta", "float"},
        {"pfcand_sum_phi", "float"},
        {"pfcand_sum_energy", "float"},
        
        // Higgs variables
        {"gen_higgs1_pt", "float"},
        {"gen_higgs1_eta", "float"},
        {"gen_higgs1_phi", "float"},
        {"gen_higgs1_mass", "float"},
        {"gen_higgs2_pt", "float"},
        {"gen_higgs2_eta", "float"},
        {"gen_higgs2_phi", "float"},
        {"gen_higgs2_mass", "float"},
        {"gen_dihiggs_mass", "float"},
        {"gen_dihiggs_HT", "float"},

        // Particle variables
        {"part_px", "vector<float>"},
        {"part_py", "vector<float>"},
        {"part_pz", "vector<float>"},
        {"part_energy", "vector<float>"},
        {"part_mass", "vector<float>"},
        {"part_pt", "vector<float>"},
        {"part_eta", "vector<float>"},
        {"part_phi", "vector<float>"},
        {"part_charge", "vector<int>"},
        {"part_pid", "vector<int>"},
        {"part_d0val", "vector<float>"},
        {"part_d0err", "vector<float>"},
        {"part_dzval", "vector<float>"},
        {"part_dzerr", "vector<float>"},
        {"part_dr", "vector<float>"},
        {"part_deta", "vector<float>"},
        {"part_dphi", "vector<float>"},
                
        // GenParticle variables
        {"gen_bhadron_fromhh", "vector<int>"},
        {"gen_bhadron_px", "vector<float>"},
        {"gen_bhadron_py", "vector<float>"},
        {"gen_bhadron_pz", "vector<float>"},
        {"gen_bhadron_energy", "vector<float>"},
        {"gen_bhadron_mass", "vector<float>"},
        {"gen_bhadron_pt", "vector<float>"},
        {"gen_bhadron_eta", "vector<float>"},
        {"gen_bhadron_phi", "vector<float>"},
        {"gen_bhadron_charge", "vector<int>"},
        {"gen_bhadron_pid", "vector<int>"},

        // Gen-level info
        {"genpart_pt", "vector<float>"},
        {"genpart_eta", "vector<float>"},
        {"genpart_phi", "vector<float>"},
        {"genpart_energy", "vector<float>"},
        {"genpart_pid", "vector<int>"},
        {"process_index", "int"},
        {"gen_weight", "vector<float>"},
    };
    
    std::vector<std::pair<std::string, std::string>> branchListOut = {
        {"cls_index", "int"},
        {"score_0", "float"}, {"score_1", "float"}, {"score_2", "float"}, {"score_3", "float"},
        {"score_4", "float"}, {"score_5", "float"}, {"score_6", "float"}, {"score_7", "float"},
        {"score_8", "float"}, {"score_9", "float"}, {"score_10", "float"}, {"score_11", "float"},
        {"score_12", "float"}, {"score_13", "float"}, {"score_14", "float"}, {"score_15", "float"},
        {"score_16", "float"}, {"score_17", "float"}, {"score_18", "float"}, {"score_19", "float"},
        {"score_20", "float"}, {"score_21", "float"}, {"score_22", "float"}, {"score_23", "float"},
        {"score_24", "float"}, {"score_25", "float"}, {"score_26", "float"}, {"score_27", "float"},
        {"score_28", "float"}, {"score_29", "float"}, {"score_30", "float"}, {"score_31", "float"},
        {"score_32", "float"}, {"score_33", "float"}, {"score_34", "float"}, {"score_35", "float"},
        {"score_36", "float"}, {"score_37", "float"}, {"score_38", "float"}, {"score_39", "float"},
        {"score_40", "float"}, {"score_41", "float"}, {"score_42", "float"}, {"score_43", "float"},
        {"score_44", "float"}, {"score_45", "float"}, {"score_46", "float"}, {"score_47", "float"},
        {"score_48", "float"}, {"score_49", "float"}, {"score_50", "float"}, {"score_51", "float"},
        {"score_52", "float"}, {"score_53", "float"}, {"score_54", "float"}, {"score_55", "float"},
        {"score_56", "float"}, {"score_57", "float"}, {"score_58", "float"}, {"score_59", "float"},
        {"score_60", "float"}, {"score_61", "float"}, {"score_62", "float"}, {"score_63", "float"},
        {"score_64", "float"}, {"score_65", "float"}, {"score_66", "float"}, {"score_67", "float"},
        {"score_68", "float"}, {"score_69", "float"}, {"score_70", "float"}, {"score_71", "float"},
        {"score_72", "float"}, {"score_73", "float"}, {"score_74", "float"}, {"score_75", "float"},
        {"score_76", "float"}, {"score_77", "float"}, {"score_78", "float"}, {"score_79", "float"},
        {"score_80", "float"}, {"score_81", "float"}, {"score_82", "float"}, {"score_83", "float"},
        {"score_84", "float"}, {"score_85", "float"}, {"score_86", "float"}, {"score_87", "float"},
        {"score_88", "float"}, {"score_89", "float"}, {"score_90", "float"}, {"score_91", "float"},
        {"score_92", "float"}, {"score_93", "float"}, {"score_94", "float"}, {"score_95", "float"},
        {"score_96", "float"}, {"score_97", "float"}, {"score_98", "float"}, {"score_99", "float"},
        {"score_100", "float"}, {"score_101", "float"}, {"score_102", "float"}, {"score_103", "float"},
        {"score_104", "float"}, {"score_105", "float"}, {"score_106", "float"}, {"score_107", "float"},
        {"score_108", "float"}, {"score_109", "float"}, {"score_110", "float"}, {"score_111", "float"},
        {"score_112", "float"}, {"score_113", "float"}, {"score_114", "float"}, {"score_115", "float"},
        {"score_116", "float"}, {"score_117", "float"}, {"score_118", "float"}, {"score_119", "float"},
        {"score_120", "float"}, {"score_121", "float"}, {"score_122", "float"}, {"score_123", "float"},
        {"score_124", "float"}, {"score_125", "float"}, {"score_126", "float"}, {"score_127", "float"},
        {"score_128", "float"}, {"score_129", "float"}, {"score_130", "float"}, {"score_131", "float"},
        {"score_132", "float"}, {"score_133", "float"}, {"score_134", "float"}, {"score_135", "float"},
        {"score_136", "float"}, {"score_137", "float"},
        {"pass_selection", "int"},
        {"pass_4j3b_selection", "int"},
        {"process_index", "int"}
    };

    // Create output files
    TFile *fout = nullptr;
    TTree *tree_out = nullptr;
    EventData data_out(branchListOut);
    createOutputFile(outputFile.Data(), fout, tree_out, data_out);

    // Read input
    TFile *file_in = nullptr;
    TTree *tree_in = nullptr;
    EventData data_in(branchListIn);
    openFile(inputFile.Data(), file_in, tree_in, data_in);
    Long64_t allEntries = tree_in->GetEntries();

    std::cerr << "** Input file:    " << inputFile << std::endl;
    std::cerr << "** Total events:  " << allEntries << std::endl;

    // Initialize onnx helper
    auto orthelper = OrtHelperSophonHH(modelPath.Data(), debug);

    // Loop over all events
    int num_processed = 0;
    for (Long64_t entry = 0; entry < allEntries; ++entry) {
        if (entry > 0) {
            continue;
        }
        
        if (entry % 1000 == 0) {
            std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
        }

        
        // Load selected branches with data from specified event
        tree_in->GetEntry(entry);
        ++num_processed;

        // reset data
        data_out.reset();
        
        // Calculate cls_index
        int cls_index = calculateClsIndex(data_in.floatVars.at("gen_higgs1_mass"), 
                                        data_in.floatVars.at("gen_higgs2_mass"), 
                                        data_in.intVars.at("process_index"));
        data_out.intVars.at("cls_index") = cls_index;
        
        // Infer the SophonHH model
        std::map<std::string, std::vector<float>*> particleVars;
        std::map<std::string, float> globalVars;
        for (auto v: std::vector<std::string>({"part_px", "part_py", "part_pz", "part_energy", "part_pt", "part_eta", "part_phi", "part_deta", "part_dphi", "part_d0val", "part_d0err", "part_dzval", "part_dzerr"})) {
            particleVars[v] = data_in.vfloatVars.at(v);
        }
        for (auto v: std::vector<std::string>({"part_charge", "part_pid"})) {
            particleVars[v] = new std::vector<float>();
            for (size_t idx = 0; idx < data_in.vintVars.at(v)->size(); idx++) {
                particleVars[v]->push_back(data_in.vintVars.at(v)->at(idx));
            }
        }
        globalVars["pfcand_sum_pt"] = data_in.floatVars.at("pfcand_sum_pt");
        globalVars["pfcand_sum_energy"] = data_in.floatVars.at("pfcand_sum_energy");

        orthelper.infer_model(particleVars, globalVars);
        const auto &output = orthelper.get_output();

        // Get inference output and fill score branches
        for (size_t i = 0; i < 138; i++) {
            std::string score_name = "score_" + std::to_string(i);
            data_out.floatVars.at(score_name) = output[i];
            // std::cerr << score_name << "  " << output[i] << std::endl;
        }

        // Fill other required branches
        data_out.intVars.at("pass_selection") = data_in.intVars.at("pass_selection");
        data_out.intVars.at("pass_4j3b_selection") = data_in.intVars.at("pass_4j3b_selection");
        data_out.intVars.at("process_index") = data_in.intVars.at("process_index");

        tree_out->Fill();

    } // end loop of events

    closeFile(file_in);
    writeOutputFile(fout);

    std::cerr << TString::Format("** Written %d events to output %s", num_processed, outputFile.Data()) << std::endl;

}

//------------------------------------------------------------------------------
