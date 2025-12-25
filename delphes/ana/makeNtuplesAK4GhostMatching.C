#include <iostream>
#include <unordered_set>
#include <utility>
#include <string>
#include "TClonesArray.h"
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"

#include "JetMatching.h"
#include "JetGhostMatching.h"
#include "EventData.h"
#include "ParticleInfo.h"

void makeNtuplesAK4GhostMatching(TString inputFile, TString outputFile, TString jetBranch = "JetPUPPI", bool debug = false) {
    TFile *fout = new TFile(outputFile, "RECREATE");
    TTree *tree = new TTree("tree", "tree");

    // Define branches
    std::vector<std::pair<std::string, std::string>> branchList = {
        // Particle features
        {"part_px", "vector<float>"},
        {"part_py", "vector<float>"},
        {"part_pz", "vector<float>"},
        {"part_energy", "vector<float>"},
        {"part_deta", "vector<float>"},
        {"part_dphi", "vector<float>"},
        {"part_d0val", "vector<float>"},
        {"part_d0err", "vector<float>"},
        {"part_dzval", "vector<float>"},
        {"part_dzerr", "vector<float>"},
        {"part_charge", "vector<int>"},
        {"part_isElectron", "vector<bool>"},
        {"part_isMuon", "vector<bool>"},
        {"part_isPhoton", "vector<bool>"},
        {"part_isChargedHadron", "vector<bool>"},
        {"part_isNeutralHadron", "vector<bool>"},
        
        // Jet features
        {"jet_pt", "float"},
        {"jet_eta", "float"},
        {"jet_phi", "float"},
        {"jet_energy", "float"},
        {"jet_mass", "float"},
        {"jet_nparticles", "int"},
        
        // Ghost Matching related branches (renamed from ghost_ to jet_)
        {"jet_nBHadron", "int"},
        {"jet_nCHadron", "int"},
        {"jet_nElectron", "int"},
        {"jet_nMuon", "int"},
        {"jet_nTau", "int"},
        {"jet_nBParton", "int"},
        {"jet_nBbarParton", "int"},
        {"jet_nCParton", "int"},
        {"jet_nCbarParton", "int"},
        {"jet_nSParton", "int"},
        {"jet_nSbarParton", "int"},
        {"jet_nUParton", "int"},
        {"jet_nUbarParton", "int"},
        {"jet_nDParton", "int"},
        {"jet_nDbarParton", "int"},
        {"jet_nGParton", "int"},
        
        {"jet_hadronFlavor", "int"}, 
        {"jet_partonFlavor", "int"}, 
        
        // Auxiliary gen particle features
        {"aux_genparton_pt", "vector<float>"},
        {"aux_genparton_eta", "vector<float>"},
        {"aux_genparton_phi", "vector<float>"},
        {"aux_genparton_mass", "vector<float>"},
        {"aux_genparton_pid", "vector<int>"},
        
        {"aux_genhadron_pt", "vector<float>"},
        {"aux_genhadron_eta", "vector<float>"},
        {"aux_genhadron_phi", "vector<float>"},
        {"aux_genhadron_mass", "vector<float>"},
        {"aux_genhadron_pid", "vector<int>"},
    };
    
    EventData data(branchList);
    data.setOutputBranch(tree);

    // Read input
    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);
    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    Long64_t allEntries = treeReader->GetEntries();

    std::cerr << "** Input file:    " << inputFile << std::endl;
    std::cerr << "** Jet branch:    " << jetBranch << std::endl;
    std::cerr << "** Total events:  " << allEntries << std::endl;

    // Analysis
    TClonesArray *branchVertex = treeReader->UseBranch("Vertex"); // for pileup
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchPFCand = treeReader->UseBranch("ParticleFlowCandidate");
    TClonesArray *branchJet = treeReader->UseBranch(jetBranch);

    double jetR = 0.4;
    std::cerr << "jetR = " << jetR << std::endl;

    // Initialize Ghost Matching method
    JetGhostMatching ghostMatch(treeReader, jetR, 1e-18);  

    // Process all events
    int num_processed = 0;
    for (Long64_t entry = 0; entry < allEntries; ++entry) {
        if (debug) {
            if (entry>=50) {break;}
        }
        
        if (entry % 1000 == 0) {
            std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
            std::cerr << "processed " << num_processed << " jets." << std::endl;
        }

        // Load data for the specified event
        treeReader->ReadEntry(entry);

        // Prepare GenParticle list for the event
        std::vector<GenParticle*> genParticles;
        for (Int_t j = 0; j < branchParticle->GetEntriesFast(); ++j) {
            genParticles.push_back((GenParticle *)branchParticle->At(j));
        }
        
        // Collect all jets in the event
        std::vector<Jet*> eventJets;
        for (Int_t i = 0; i < branchJet->GetEntriesFast(); ++i) {
            eventJets.push_back((Jet *)branchJet->At(i));
        }
        
        // Get detailed information using Ghost Matching method for all jets at once
        std::vector<JetGhostMatching::JetGhostContent> ghostContents = 
            ghostMatch.getDetailedGhostContent(eventJets, genParticles);
        
        // Process each jet with its ghost content
        for (size_t i = 0; i < eventJets.size(); ++i) {
            Jet *jet = eventJets[i];
            const auto& ghostContent = ghostContents[i];
            
            // Skip light jets with no parton flavor
            // if (ghostContent.hadronFlavorLabel == "light" && ghostContent.partonFlavorLabel == "Invalid")
            //     continue;
                
            data.reset();
            
            if (debug) {
                std::cerr << "=========================================" << std::endl;
                std::cerr << ">> Jet information: PT=" << jet->PT 
                          << ", Eta=" << jet->Eta 
                          << ", Phi=" << jet->Phi 
                          << ", Mass=" << jet->Mass 
                          << ", Energy=" << jet->P4().E() << std::endl;
                std::cerr << "-----------------------------------------" << std::endl;
                
                std::cerr << ">> Ghost hadron flavor: " << ghostContent.hadronFlavorLabel 
                          << " (" << ghostContent.hadronFlavor << ")" << std::endl;
                std::cerr << ">> Ghost parton flavor: " << ghostContent.partonFlavorLabel 
                          << " (" << ghostContent.partonFlavor << ")" << std::endl;
                
                // Print debug information about ghost matching quality
                if (ghostContent.hasValidGhostJet) {
                    std::cerr << ">> Ghost matching quality:" << std::endl;
                    std::cerr << "   deltaR: " << ghostContent.deltaR_to_original << std::endl;
                    std::cerr << "   PT ratio: " << ghostContent.pt_ratio << std::endl;
                    std::cerr << "   Eta diff: " << ghostContent.eta_diff << std::endl;
                    std::cerr << "   Phi diff: " << ghostContent.phi_diff << std::endl;
                    std::cerr << "   Mass ratio: " << ghostContent.mass_ratio << std::endl;
                } else {
                    std::cerr << ">> No valid ghost jet match!" << std::endl;
                }
                
                std::cerr << "-----------------------------------------" << std::endl;
                
                // Debug information for hadron flavor particles
                auto hadronFlavorParticles = ghostMatch.getHadronFlavorParticles(ghostContent);
                std::cerr << ">> Hadron flavor particles: " << hadronFlavorParticles.size() << std::endl;
                for (size_t j = 0; j < hadronFlavorParticles.size(); ++j) {
                    const auto& part = hadronFlavorParticles[j];
                    double deltaR = std::sqrt(std::pow(jet->Eta - part->Eta, 2) + 
                                            std::pow(TVector2::Phi_mpi_pi(jet->Phi - part->Phi), 2));
                    std::cerr << "   Hadron[" << j << "]: PID=" << part->PID 
                            << ", PT=" << part->PT 
                            << ", Eta=" << part->Eta 
                            << ", Phi=" << part->Phi 
                            << ", Mass=" << part->Mass 
                            << ", deltaR=" << deltaR << std::endl;
                }
                
                std::cerr << "-----------------------------------------" << std::endl;
                
                // Debug information for parton flavor particles
                auto partonFlavorParticles = ghostMatch.getPartonFlavorParticles(ghostContent);
                std::cerr << ">> Parton flavor particles: " << partonFlavorParticles.size() << std::endl;
                for (size_t j = 0; j < partonFlavorParticles.size(); ++j) {
                    const auto& part = partonFlavorParticles[j];
                    double deltaR = std::sqrt(std::pow(jet->Eta - part->Eta, 2) + 
                                            std::pow(TVector2::Phi_mpi_pi(jet->Phi - part->Phi), 2));
                    std::cerr << "   Parton[" << j << "]: PID=" << part->PID 
                            << ", PT=" << part->PT 
                            << ", Eta=" << part->Eta 
                            << ", Phi=" << part->Phi 
                            << ", Mass=" << part->Mass 
                            << ", deltaR=" << deltaR << std::endl;
                }
                
                std::cerr << "=========================================" << std::endl;
            }

            // Fill basic jet information
            data.floatVars["jet_pt"] = jet->PT;
            data.floatVars["jet_eta"] = jet->Eta;
            data.floatVars["jet_phi"] = jet->Phi;
            data.floatVars["jet_energy"] = jet->P4().E();
            data.floatVars["jet_mass"] = jet->Mass;
            
            // Fill ghost matching information (renamed from ghost_ to jet_)
            data.intVars["jet_nBHadron"] = ghostContent.nBHadron;
            data.intVars["jet_nCHadron"] = ghostContent.nCHadron;
            data.intVars["jet_nElectron"] = ghostContent.nElectron;
            data.intVars["jet_nMuon"] = ghostContent.nMuon;
            data.intVars["jet_nTau"] = ghostContent.nTau;
            data.intVars["jet_nBParton"] = ghostContent.nBParton;
            data.intVars["jet_nBbarParton"] = ghostContent.nBbarParton;
            data.intVars["jet_nCParton"] = ghostContent.nCParton;
            data.intVars["jet_nCbarParton"] = ghostContent.nCbarParton;
            data.intVars["jet_nSParton"] = ghostContent.nSParton;
            data.intVars["jet_nSbarParton"] = ghostContent.nSbarParton;
            data.intVars["jet_nUParton"] = ghostContent.nUParton;
            data.intVars["jet_nUbarParton"] = ghostContent.nUbarParton;
            data.intVars["jet_nDParton"] = ghostContent.nDParton;
            data.intVars["jet_nDbarParton"] = ghostContent.nDbarParton;
            data.intVars["jet_nGParton"] = ghostContent.nGParton;
            data.intVars["jet_hadronFlavor"] = ghostContent.hadronFlavor;
            data.intVars["jet_partonFlavor"] = ghostContent.partonFlavor;

            // Process jet constituents using ParticleInfo
            std::vector<ParticleInfo> particles;
            for (Int_t j = 0; j < jet->Constituents.GetEntriesFast(); ++j) {
                TObject *constituent = jet->Constituents.At(j);
                if (constituent == nullptr) continue;

                // Handle only GenParticle and ParticleFlowCandidate
                if (constituent->IsA() == GenParticle::Class()) {
                    particles.emplace_back((GenParticle *)constituent);
                } else if (constituent->IsA() == ParticleFlowCandidate::Class()) {
                    particles.emplace_back((ParticleFlowCandidate *)constituent);
                }
                
                // Check if the particle is valid
                if (!particles.empty()) {
                    const auto &p = particles.back();
                    if (std::abs(p.pz) > 10000 || std::abs(p.eta) > 5 || p.pt <= 0) {
                        particles.pop_back();
                    }
                }
            }

            // Sort particles by pt
            std::sort(particles.begin(), particles.end(), [](const auto &a, const auto &b) { return a.pt > b.pt; });

            // Load the primary vertex
            const Vertex *pv = (branchVertex != nullptr) ? ((Vertex *)branchVertex->At(0)) : nullptr;

            data.intVars["jet_nparticles"] = particles.size();
            for (const auto &p : particles) {
                data.vfloatVars.at("part_px")->push_back(p.px);
                data.vfloatVars.at("part_py")->push_back(p.py);
                data.vfloatVars.at("part_pz")->push_back(p.pz);
                data.vfloatVars.at("part_energy")->push_back(p.energy);
                data.vfloatVars.at("part_deta")->push_back((jet->Eta > 0 ? 1 : -1) * (p.eta - jet->Eta));
                data.vfloatVars.at("part_dphi")->push_back(TVector2::Phi_mpi_pi(p.phi - jet->Phi));
                data.vfloatVars.at("part_d0val")->push_back(p.d0);
                data.vfloatVars.at("part_d0err")->push_back(p.d0err);
                data.vfloatVars.at("part_dzval")->push_back((pv && p.dz != 0) ? (p.dz - pv->Z) : p.dz);
                data.vfloatVars.at("part_dzerr")->push_back(p.dzerr);
                data.vintVars.at("part_charge")->push_back(p.charge);
                data.vboolVars.at("part_isElectron")->push_back(p.pid == 11 || p.pid == -11);
                data.vboolVars.at("part_isMuon")->push_back(p.pid == 13 || p.pid == -13);
                data.vboolVars.at("part_isPhoton")->push_back(p.pid == 22);
                data.vboolVars.at("part_isChargedHadron")->push_back(p.charge != 0 && !(p.pid == 11 || p.pid == -11 || p.pid == 13 || p.pid == -13));
                data.vboolVars.at("part_isNeutralHadron")->push_back(p.charge == 0 && !(p.pid == 22));
            }

            // Define functions to fill auxiliary gen particle information
            auto fillAuxPartonVars = [](EventData& data, const auto& part) {
                data.vfloatVars.at("aux_genparton_pt")->push_back(part->PT);
                data.vfloatVars.at("aux_genparton_eta")->push_back(part->Eta);
                data.vfloatVars.at("aux_genparton_phi")->push_back(part->Phi);
                data.vfloatVars.at("aux_genparton_mass")->push_back(part->Mass);
                data.vintVars.at("aux_genparton_pid")->push_back(part->PID);
            };
            
            auto fillAuxHadronVars = [](EventData& data, const auto& part) {
                data.vfloatVars.at("aux_genhadron_pt")->push_back(part->PT);
                data.vfloatVars.at("aux_genhadron_eta")->push_back(part->Eta);
                data.vfloatVars.at("aux_genhadron_phi")->push_back(part->Phi);
                data.vfloatVars.at("aux_genhadron_mass")->push_back(part->Mass);
                data.vintVars.at("aux_genhadron_pid")->push_back(part->PID);
            };
            
            // Fill auxiliary gen particle information based on flavor
            // Get particles that determine parton flavor label
            auto partonFlavorParticles = ghostMatch.getPartonFlavorParticles(ghostContent);
            for (const auto& part : partonFlavorParticles) {
                fillAuxPartonVars(data, part);
            }
            
            // Get particles that determine hadron flavor label
            auto hadronFlavorParticles = ghostMatch.getHadronFlavorParticles(ghostContent);
            for (const auto& part : hadronFlavorParticles) {
                fillAuxHadronVars(data, part);
            }

            tree->Fill();
            num_processed++;
        } // end loop of jets
    } // end loop of events

    // Write and close
    fout->cd();
    tree->Write();
    fout->Close();

    std::cerr << "Processed " << num_processed << " jets." << std::endl;
    std::cerr << "Output written to " << outputFile << std::endl;
}
