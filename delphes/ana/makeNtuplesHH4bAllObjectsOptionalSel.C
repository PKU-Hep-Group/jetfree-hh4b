#include <iostream>
#include <unordered_set>
#include <utility>
#include "TClonesArray.h"
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "EventData.h"
#include "GenPartGenericProcessor.h"
// #include "FatJetMatching.h"
#include "OrtHelperSophon.h"
#include "OrtHelperSophonAK4.h"
#include "JetMatching.h"
#include "JetGhostMatching.h"

// Function to process jet-related information
void processJet(const Jet* jet, EventData& data, OrtHelperSophonAK4* sp4helper = nullptr, const Vertex* pv = nullptr, const JetGhostMatching::JetGhostContent* ghostContent = nullptr) {
    data.vfloatVars["jet_pt"]->push_back(jet->PT);
    data.vfloatVars["jet_eta"]->push_back(jet->Eta);
    data.vfloatVars["jet_phi"]->push_back(jet->Phi);
    data.vfloatVars["jet_energy"]->push_back(jet->P4().Energy());
    data.vfloatVars["jet_mass"]->push_back(jet->Mass);
    
    if (ghostContent) {
        data.vintVars["jet_hadronFlavor"]->push_back(ghostContent->hadronFlavor);
        data.vintVars["jet_partonFlavor"]->push_back(ghostContent->partonFlavor);
    } else {
        data.vintVars["jet_hadronFlavor"]->push_back(-1);
        data.vintVars["jet_partonFlavor"]->push_back(-1);
    }
    data.vintVars["jet_flavor"]->push_back(jet->Flavor);

    int nparticles = 0;
    // Loop over all jet's constituents
    for (Int_t idx_ct = 0; idx_ct < jet->Constituents.GetEntriesFast(); ++idx_ct) {
        const TObject *object = jet->Constituents.At(idx_ct);
        if (object && object->IsA() == ParticleFlowCandidate::Class()) {
            nparticles++;
        }
    }
    data.vfloatVars["jet_nparticles"]->push_back(nparticles);
    
    // Apply SophonAK4 tagging if available
    if (sp4helper != nullptr) {
        std::map<std::string, std::vector<float>> particleVars;
        std::map<std::string, float> jetVars;
        
        // Prepare input for SophonAK4
        for (Int_t idx_ct = 0; idx_ct < jet->Constituents.GetEntriesFast(); ++idx_ct) {
            const TObject *object = jet->Constituents.At(idx_ct);
            if (!object) continue;
            
            if (object->IsA() == ParticleFlowCandidate::Class()) {
                const ParticleFlowCandidate *pfcand = (ParticleFlowCandidate *)object;
                
                if (std::abs(pfcand->Eta) > 5 || pfcand->PT <= 0) continue;
                
                TLorentzVector p4 = pfcand->P4();
                
                particleVars["part_px"].push_back(p4.Px());
                particleVars["part_py"].push_back(p4.Py());
                particleVars["part_pz"].push_back(p4.Pz());
                particleVars["part_energy"].push_back(p4.E());
                particleVars["part_pt"].push_back(pfcand->PT);
                particleVars["part_deta"].push_back((jet->Eta > 0 ? 1 : -1) * (pfcand->Eta - jet->Eta));
                particleVars["part_dphi"].push_back(deltaPhi(pfcand->Phi, jet->Phi));
                particleVars["part_charge"].push_back(pfcand->Charge);
                particleVars["part_pid"].push_back(pfcand->PID);
                particleVars["part_d0val"].push_back(pfcand->D0);
                particleVars["part_d0err"].push_back(pfcand->ErrorD0);
                particleVars["part_dzval"].push_back((pv && pfcand->DZ != 0) ? (pfcand->DZ - pv->Z) : pfcand->DZ);
                particleVars["part_dzerr"].push_back(pfcand->ErrorDZ);
            }
        }
        
        jetVars["jet_pt"] = jet->PT;
        jetVars["jet_eta"] = jet->Eta;
        jetVars["jet_phi"] = jet->Phi;
        jetVars["jet_energy"] = jet->P4().Energy();
        
        // Infer SophonAK4 model
        sp4helper->infer_model(particleVars, jetVars);
        const auto &sp4output = sp4helper->get_output();
        
        // Fill SophonAK4 scores
        data.vfloatVars["jet_sophonAK4_probB"]->push_back(
            std::accumulate(sp4output.begin() + 0, sp4output.begin() + 5, 0.0)
        );
        data.vfloatVars["jet_sophonAK4_probC"]->push_back(
            std::accumulate(sp4output.begin() + 5, sp4output.begin() + 10, 0.0)
        );
        data.vfloatVars["jet_sophonAK4_probL"]->push_back(
            std::accumulate(sp4output.begin() + 10, sp4output.begin() + 32, 0.0)
        );

    } else {
        // If no SophonAK4 model, fill with default values
        data.vfloatVars["jet_sophonAK4_probB"]->push_back(-1.0);
        data.vfloatVars["jet_sophonAK4_probC"]->push_back(-1.0);
        data.vfloatVars["jet_sophonAK4_probL"]->push_back(-1.0);
    }
}

// Function to process fat jet-related information
void processFatJet(const Jet* fatjet, EventData& data, OrtHelperSophon* sp8helper = nullptr, const Vertex* pv = nullptr) {
    data.vfloatVars["fj_pt"]->push_back(fatjet->PT);
    data.vfloatVars["fj_eta"]->push_back(fatjet->Eta);
    data.vfloatVars["fj_phi"]->push_back(fatjet->Phi);
    data.vfloatVars["fj_energy"]->push_back(fatjet->P4().Energy());
    data.vfloatVars["fj_mass"]->push_back(fatjet->Mass);
    data.vfloatVars["fj_sdmass"]->push_back(fatjet->SoftDroppedP4[0].M());
    data.vfloatVars["fj_trmass"]->push_back(fatjet->TrimmedP4[0].M());

    int nparticles = 0;
    // Loop over all fatjet's constituents
    for (Int_t idx_ct = 0; idx_ct < fatjet->Constituents.GetEntriesFast(); ++idx_ct) {
        const TObject *object = fatjet->Constituents.At(idx_ct);
        if (object && object->IsA() == ParticleFlowCandidate::Class()) {
            nparticles++;
        }
    }
    data.vfloatVars["fj_nparticles"]->push_back(nparticles);
    
    // Apply Sophon tagging if available
    if (sp8helper != nullptr) {
        std::map<std::string, std::vector<float>> particleVars;
        std::map<std::string, float> jetVars;
        
        // Prepare input for Sophon
        // Loop over all jet's constituents
        std::vector<ParticleInfo> particles;
        for (Int_t j = 0; j < fatjet->Constituents.GetEntriesFast(); ++j) {
            const TObject *object = fatjet->Constituents.At(j);
            
            // Check if the constituent is accessible
            if (!object)
                continue;
            
            if (object->IsA() == GenParticle::Class()) {
                particles.emplace_back((GenParticle *)object);
            } else if (object->IsA() == ParticleFlowCandidate::Class()) {
                particles.emplace_back((ParticleFlowCandidate *)object);
            }
            const auto &p = particles.back();
            if (std::abs(p.pz) > 10000 || std::abs(p.eta) > 5 || p.pt <= 0) {
                particles.pop_back();
            }
        }
        
        // Sort particles by pt
        std::sort(particles.begin(), particles.end(), [](const auto &a, const auto &b) { return a.pt > b.pt; });
        
        // Fill particleVars and jetVars
        for (const auto &p : particles) {
            particleVars["part_px"].push_back(p.px);
            particleVars["part_py"].push_back(p.py);
            particleVars["part_pz"].push_back(p.pz);
            particleVars["part_energy"].push_back(p.energy);
            particleVars["part_pt"].push_back(p.pt);
            particleVars["part_deta"].push_back((fatjet->Eta > 0 ? 1 : -1) * (p.eta - fatjet->Eta));
            particleVars["part_dphi"].push_back(deltaPhi(p.phi, fatjet->Phi));
            particleVars["part_charge"].push_back(p.charge);
            particleVars["part_pid"].push_back(p.pid);
            particleVars["part_d0val"].push_back(p.d0);
            particleVars["part_d0err"].push_back(p.d0err);
            particleVars["part_dzval"].push_back((pv && p.dz != 0) ? (p.dz - pv->Z) : p.dz);
            particleVars["part_dzerr"].push_back(p.dzerr);
        }
        
        jetVars["jet_pt"] = fatjet->PT;
        jetVars["jet_eta"] = fatjet->Eta;
        jetVars["jet_phi"] = fatjet->Phi;
        jetVars["jet_energy"] = fatjet->P4().Energy();
        
        // Infer Sophon model
        sp8helper->infer_model(particleVars, jetVars);
        const auto &spoutput = sp8helper->get_output();
        
        // Fill Sophon scores
        data.vfloatVars["fj_sophon_probXbb"]->push_back(spoutput[0]);
        data.vfloatVars["fj_sophon_probXcc"]->push_back(spoutput[1]);
        data.vfloatVars["fj_sophon_probXqq"]->push_back(spoutput[3]);
        data.vfloatVars["fj_sophon_probXbc"]->push_back(spoutput[4]);
        data.vfloatVars["fj_sophon_probXcs"]->push_back(spoutput[5]);
        data.vfloatVars["fj_sophon_probXbq"]->push_back(spoutput[6]);
        data.vfloatVars["fj_sophon_probXcq"]->push_back(spoutput[7]);
        data.vfloatVars["fj_sophon_probXbqq"]->push_back(spoutput[70] + spoutput[127]); // bqq + bcs
        data.vfloatVars["fj_sophon_probQCD"]->push_back(
            std::accumulate(spoutput.begin() + 161, spoutput.begin() + 188, 0.0)
        );
    } else {
        // If no Sophon model, fill with default values
        data.vfloatVars["fj_sophon_probXbb"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXcc"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXqq"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXbc"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXcs"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXbq"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXcq"]->push_back(-1.0);
        data.vfloatVars["fj_sophon_probXbqq"]->push_back(-1.0);
        data.vfloatVars["fj)sophon_probQCD"]->push_back(-1.0);
    }
}

// Function to process electron-related information
void processElectron(const Electron* electron, EventData& data) {
    data.vfloatVars["lep_pt"]->push_back(electron->PT);
    data.vfloatVars["lep_eta"]->push_back(electron->Eta);
    data.vfloatVars["lep_phi"]->push_back(electron->Phi);
    data.vfloatVars["lep_energy"]->push_back(electron->P4().E());
    data.vintVars["lep_charge"]->push_back(electron->Charge);
    data.vintVars["lep_pid"]->push_back(-11 * electron->Charge);
    data.vfloatVars["lep_iso"]->push_back(electron->IsolationVar);
}

// Function to process muon-related information
void processMuon(const Muon* muon, EventData& data) {
    data.vfloatVars["lep_pt"]->push_back(muon->PT);
    data.vfloatVars["lep_eta"]->push_back(muon->Eta);
    data.vfloatVars["lep_phi"]->push_back(muon->Phi);
    data.vfloatVars["lep_energy"]->push_back(muon->P4().E());
    data.vintVars["lep_charge"]->push_back(muon->Charge);
    data.vintVars["lep_pid"]->push_back(-13 * muon->Charge);
    data.vfloatVars["lep_iso"]->push_back(muon->IsolationVar);
}

// Function to process particle information
void processParticle(const ParticleFlowCandidate* pfcand, EventData& data, const Vertex* pv, 
                    const TLorentzVector& pfcand_sum) {
    if (std::abs(pfcand->Eta) > 5 || pfcand->PT <= 0) {
        return;
    }

    TLorentzVector p4 = pfcand->P4();

    // Calculate delta values with pfcand_sum
    float deta = p4.Eta() - pfcand_sum.Eta();
    float dphi = deltaPhi(p4.Phi(), pfcand_sum.Phi());
    float dr = std::sqrt(deta*deta + dphi*dphi);

    data.vfloatVars["part_px"]->push_back(p4.Px());  
    data.vfloatVars["part_py"]->push_back(p4.Py());  
    data.vfloatVars["part_pz"]->push_back(p4.Pz());  
    data.vfloatVars["part_energy"]->push_back(p4.E());
    data.vfloatVars["part_mass"]->push_back(pfcand->Mass);
    data.vfloatVars["part_pt"]->push_back(pfcand->PT);
    data.vfloatVars["part_eta"]->push_back(pfcand->Eta);
    data.vfloatVars["part_phi"]->push_back(pfcand->Phi);
    data.vintVars["part_charge"]->push_back(pfcand->Charge);
    data.vintVars["part_pid"]->push_back(pfcand->PID);
    data.vfloatVars["part_d0val"]->push_back(pfcand->D0);
    data.vfloatVars["part_d0err"]->push_back(pfcand->ErrorD0);
    data.vfloatVars["part_dzval"]->push_back((pv && pfcand->DZ != 0) ? (pfcand->DZ - pv->Z) : pfcand->DZ);
    data.vfloatVars["part_dzerr"]->push_back(pfcand->ErrorDZ);
    
    // Add new delta variables
    data.vfloatVars["part_dr"]->push_back(dr);
    data.vfloatVars["part_deta"]->push_back(deta);
    data.vfloatVars["part_dphi"]->push_back(dphi);
}

// Function to build mother particle map
void buildMotherMap(TClonesArray* branchParticle, std::map<int, std::vector<int>>& motherMap) {
    for (int i = 0; i < branchParticle->GetEntriesFast(); ++i) {
        const GenParticle* particle = (GenParticle*)branchParticle->At(i);
        if (particle->M1 >= 0) {
            motherMap[i].push_back(particle->M1);
        }
        if (particle->M2 >= 0 && particle->M2 != particle->M1) {
            motherMap[i].push_back(particle->M2);
        }
    }
}

// Non-recursive function to trace decay chains, truncating at first Higgs
std::vector<std::vector<int>> traceDecayChains(int particleIdx, const std::map<int, std::vector<int>>& motherMap, TClonesArray* branchParticle) {
    std::vector<std::vector<int>> chains;
    std::vector<int> currentChain;
    std::vector<std::pair<int, int>> stack; // pair of (particleIdx, motherIdx)
    
    // Cycle detection
    std::unordered_set<int> visitedInCurrentChain;
    const int MAX_ITERATIONS = 1000;
    int iteration = 0;
    
    // Start with the particle itself
    currentChain.push_back(particleIdx);
    visitedInCurrentChain.insert(particleIdx);
    
    // If particle has mothers, add them to the stack
    auto it = motherMap.find(particleIdx);
    if (it != motherMap.end()) {
        for (int i = it->second.size() - 1; i >= 0; --i) {
            stack.push_back({particleIdx, it->second[i]});
        }
    } else {
        // No mothers, this is a complete chain
        chains.push_back(currentChain);
        return chains;
    }
    
    while (!stack.empty() && iteration < MAX_ITERATIONS) {
        iteration++;
        
        auto [childIdx, motherIdx] = stack.back();
        stack.pop_back();
        
        // Check for invalid mother index
        if (motherIdx < 0 || motherIdx >= branchParticle->GetEntriesFast()) {
            continue;
        }
        
        // Check for cycle
        if (visitedInCurrentChain.count(motherIdx)) {
            std::cout << "WARNING: Cycle detected at particle " << motherIdx << " in chain starting from " << particleIdx << std::endl;
            continue;
        }
        
        // Find position of child in current chain
        auto childPos = std::find(currentChain.begin(), currentChain.end(), childIdx);
        
        // If child is in the chain, remove everything after it
        if (childPos != currentChain.end()) {
            // Clean up visitedInCurrentChain for removed elements
            for (auto it = childPos + 1; it != currentChain.end(); ++it) {
                visitedInCurrentChain.erase(*it);
            }
            currentChain.erase(childPos + 1, currentChain.end());
        } else {
            // This should not happen in a well-formed tree
            continue;
        }
        
        // Add mother to chain
        currentChain.push_back(motherIdx);
        visitedInCurrentChain.insert(motherIdx);
        
        // Check if mother is a Higgs
        const GenParticle* mother = (GenParticle*)branchParticle->At(motherIdx);
        bool isHiggs = (std::abs(mother->PID) == 25 || std::abs(mother->PID) == 35);
        
        if (isHiggs) {
            // If we found a Higgs, save this chain and stop tracing further
            chains.push_back(currentChain);
            // Remove the Higgs to backtrack
            currentChain.pop_back();
            visitedInCurrentChain.erase(motherIdx);
            continue;
        }
        
        // Check if mother has further mothers
        auto motherIt = motherMap.find(motherIdx);
        if (motherIt != motherMap.end() && !motherIt->second.empty()) {
            // Add mother's mothers to stack (only if not already visited)
            for (int i = motherIt->second.size() - 1; i >= 0; --i) {
                int grandMotherIdx = motherIt->second[i];
                if (!visitedInCurrentChain.count(grandMotherIdx) && 
                    grandMotherIdx >= 0 && grandMotherIdx < branchParticle->GetEntriesFast()) {
                    stack.push_back({motherIdx, grandMotherIdx});
                }
            }
        } else {
            // No more mothers, this is a complete chain
            chains.push_back(currentChain);
            
            // Remove the last mother to backtrack
            currentChain.pop_back();
            visitedInCurrentChain.erase(motherIdx);
        }
    }
    
    if (iteration >= MAX_ITERATIONS) {
        std::cout << "WARNING: Maximum iterations reached for particle " << particleIdx 
                 << ". Possible infinite loop prevented!" << std::endl;
    }
    
    return chains;
}


// // Non-recursive function to trace decay chains, truncating at first Higgs
// std::vector<std::vector<int>> traceDecayChains(int particleIdx, const std::map<int, std::vector<int>>& motherMap, TClonesArray* branchParticle) {
//     std::vector<std::vector<int>> chains;
//     std::vector<int> currentChain;
//     std::vector<std::pair<int, int>> stack; // pair of (particleIdx, motherIdx)
    
//     // Start with the particle itself
//     currentChain.push_back(particleIdx);
    
//     // If particle has mothers, add them to the stack
//     auto it = motherMap.find(particleIdx);
//     if (it != motherMap.end()) {
//         for (int i = it->second.size() - 1; i >= 0; --i) {
//             stack.push_back({particleIdx, it->second[i]});
//         }
//     } else {
//         // No mothers, this is a complete chain
//         chains.push_back(currentChain);
//         return chains;
//     }
    
//     while (!stack.empty()) {
//         auto [childIdx, motherIdx] = stack.back();
//         stack.pop_back();
        
//         // Find position of child in current chain
//         auto childPos = std::find(currentChain.begin(), currentChain.end(), childIdx);
        
//         // If child is in the chain, remove everything after it
//         if (childPos != currentChain.end()) {
//             currentChain.erase(childPos + 1, currentChain.end());
//         } else {
//             // This should not happen in a well-formed tree
//             continue;
//         }
        
//         // Add mother to chain
//         currentChain.push_back(motherIdx);
        
//         // Check if mother is a Higgs
//         const GenParticle* mother = (GenParticle*)branchParticle->At(motherIdx);
//         bool isHiggs = (std::abs(mother->PID) == 25 || std::abs(mother->PID) == 35);
        
//         if (isHiggs) {
//             // If we found a Higgs, save this chain and stop tracing further
//             chains.push_back(currentChain);
//             // Remove the Higgs to backtrack
//             currentChain.pop_back();
//             continue;
//         }
        
//         // Check if mother has further mothers
//         auto motherIt = motherMap.find(motherIdx);
//         if (motherIt != motherMap.end() && !motherIt->second.empty()) {
//             // Add mother's mothers to stack
//             for (int i = motherIt->second.size() - 1; i >= 0; --i) {
//                 stack.push_back({motherIdx, motherIt->second[i]});
//             }
//         } else {
//             // No more mothers, this is a complete chain
//             chains.push_back(currentChain);
            
//             // Remove the last mother to backtrack
//             currentChain.pop_back();
//         }
//     }
    
//     return chains;
// }

// Check if chain contains direct Higgs to b decay
bool hasDirectHiggsToBDecay(const std::vector<int>& chain, TClonesArray* branchParticle) {
    for (size_t i = 0; i < chain.size() - 1; ++i) {
        const GenParticle* particle = (GenParticle*)branchParticle->At(chain[i]);
        const GenParticle* mother = (GenParticle*)branchParticle->At(chain[i+1]);
        
        // Check if particle is b quark and mother is Higgs
        if (std::abs(particle->PID) == 5 && 
            (std::abs(mother->PID) == 25 || std::abs(mother->PID) == 35)) {
            return true;
        }
    }
    return false;
}

// Helper function to print decay chain
std::string printChain(const std::vector<int>& chain, TClonesArray* branchParticle, bool markHiggsToBChain) {
    std::stringstream ss;
    
    for (size_t i = 0; i < chain.size(); ++i) {
        const GenParticle* particle = (GenParticle*)branchParticle->At(chain[i]);
        ss << particle->PID << "[" << chain[i] << "]";
        
        if (i < chain.size() - 1) {
            ss << " <- ";
        }
    }
    
    // Check if the last particle in the chain is a Higgs
    bool endsWithHiggs = false;
    if (!chain.empty()) {
        const GenParticle* lastParticle = (GenParticle*)branchParticle->At(chain.back());
        endsWithHiggs = (std::abs(lastParticle->PID) == 25 || std::abs(lastParticle->PID) == 35);
    }
    
    if (markHiggsToBChain && hasDirectHiggsToBDecay(chain, branchParticle)) {
        ss << " (Higgs->b chain)";
    } else if (endsWithHiggs) {
        ss << " (ends with Higgs)";
    }
    
    return ss.str();
}

// Function to process GenParticle information
void processGenParticle(const GenParticle* genparticle, int particleIdx, EventData& data, TClonesArray* branchParticle, 
                       int& higgs_count, TLorentzVector& higgs1_p4, TLorentzVector& higgs2_p4,
                       std::vector<std::tuple<int, int, bool, std::vector<std::vector<int>>>>& bhadrons_info,
                       const std::map<int, std::vector<int>>& motherMap) {
    if(((std::abs(genparticle->PID) == 25)||(std::abs(genparticle->PID) == 35)) && genparticle->Status == 22) {
        TLorentzVector p4 = genparticle->P4();
        
        if(higgs_count == 0) {
            // First Higgs
            data.floatVars["gen_higgs1_pt"] = genparticle->PT;
            data.floatVars["gen_higgs1_eta"] = genparticle->Eta;
            data.floatVars["gen_higgs1_phi"] = genparticle->Phi;
            data.floatVars["gen_higgs1_mass"] = genparticle->Mass;
            higgs1_p4 = p4;
            higgs_count++;
        } else if(higgs_count == 1) {
            // Second Higgs
            data.floatVars["gen_higgs2_pt"] = genparticle->PT;
            data.floatVars["gen_higgs2_eta"] = genparticle->Eta;
            data.floatVars["gen_higgs2_phi"] = genparticle->Phi;
            data.floatVars["gen_higgs2_mass"] = genparticle->Mass;
            higgs2_p4 = p4;
            higgs_count++;
            
            // Now that we have both Higgs, calculate dihiggs variables
            TLorentzVector dihiggs_p4 = higgs1_p4 + higgs2_p4;
            data.floatVars["gen_dihiggs_mass"] = dihiggs_p4.M();
            data.floatVars["gen_dihiggs_HT"] = data.floatVars["gen_higgs1_pt"] + data.floatVars["gen_higgs2_pt"];
        }
        // Ignore additional Higgs bosons if there are more than 2
    }
    else if (genparticle->PT > 0) {
        int absPID = std::abs(genparticle->PID);
        int code1 = (absPID / 100) % 10;
        int code2 = (absPID / 1000) % 10;
        bool isBHadron = (code1 == 5 || code2 == 5);
    
        if (isBHadron) {
            bool hasBHadronDaughter = false;
            for (int i = genparticle->D1; i <= genparticle->D2; ++i) {
                if (i >= 0 && i < branchParticle->GetEntriesFast()) {
                    const GenParticle* daughter = (GenParticle*)branchParticle->At(i);
                    int absDauPID = std::abs(daughter->PID);
                    int dau_code1 = (absDauPID / 100) % 10;
                    int dau_code2 = (absDauPID / 1000) % 10;
                    if (dau_code1 == 5 || dau_code2 == 5) {
                        hasBHadronDaughter = true;
                        break;
                    }
                }
            }
            
            if (!hasBHadronDaughter) {
                // This is a final state B-hadron
                std::vector<std::vector<int>> decay_chains = traceDecayChains(particleIdx, motherMap, branchParticle);
                
                // Check if from Higgs decay
                bool isFromHiggsDecay = false;
                for (const auto& chain : decay_chains) {
                    if (hasDirectHiggsToBDecay(chain, branchParticle)) {
                        isFromHiggsDecay = true;
                        break;
                    }
                }
                
                // Store B-hadron info
                bhadrons_info.push_back(std::make_tuple(genparticle->PID, particleIdx, isFromHiggsDecay, decay_chains));
                
                TLorentzVector p4 = genparticle->P4();
                data.vintVars["gen_bhadron_fromhh"]->push_back(isFromHiggsDecay ? 1 : 0);
                data.vfloatVars["gen_bhadron_px"]->push_back(p4.Px());
                data.vfloatVars["gen_bhadron_py"]->push_back(p4.Py());
                data.vfloatVars["gen_bhadron_pz"]->push_back(p4.Pz());
                data.vfloatVars["gen_bhadron_energy"]->push_back(p4.E());
                data.vfloatVars["gen_bhadron_mass"]->push_back(genparticle->Mass);
                data.vfloatVars["gen_bhadron_pt"]->push_back(genparticle->PT);
                data.vfloatVars["gen_bhadron_eta"]->push_back(genparticle->Eta);
                data.vfloatVars["gen_bhadron_phi"]->push_back(genparticle->Phi);
                data.vintVars["gen_bhadron_charge"]->push_back(genparticle->Charge);
                data.vintVars["gen_bhadron_pid"]->push_back(genparticle->PID);
            }
        }
    }
}



void makeNtuplesHH4bAllObjectsOptionalSel(TString inputFile, TString outputFile, TString modelPathAK4, TString modelPathFatJet, TString jetBranch = "JetPUPPI", TString fatJetBranch = "JetPUPPIAK8", TString selectionLevel = "4j") {
    // Verify selection level
    if (selectionLevel != "full" && selectionLevel != "4j" && selectionLevel != "4j3b" && selectionLevel != "4j3bor2b") {
        std::cerr << "Invalid selection level: " << selectionLevel << std::endl;
        return;
    }

    TFile *fout = new TFile(outputFile, "RECREATE");
    TTree *tree = new TTree("tree", "tree");

    // Define all branches
    std::vector<std::pair<std::string, std::string>> branchList = {
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
        
        // Jet variables (as vectors)
        {"jet_pt", "vector<float>"},
        {"jet_eta", "vector<float>"},
        {"jet_phi", "vector<float>"},
        {"jet_energy", "vector<float>"},
        {"jet_mass", "vector<float>"},
        {"jet_hadronFlavor", "vector<int>"},
        {"jet_partonFlavor", "vector<int>"},
        {"jet_flavor", "vector<int>"},
        {"jet_nparticles", "vector<float>"},
        {"jet_sophonAK4_probB", "vector<float>"},
        {"jet_sophonAK4_probC", "vector<float>"},
        {"jet_sophonAK4_probL", "vector<float>"},
        
        // FatJet variables (as vectors)
        {"fj_pt", "vector<float>"},
        {"fj_eta", "vector<float>"},
        {"fj_phi", "vector<float>"},
        {"fj_energy", "vector<float>"},
        {"fj_mass", "vector<float>"},
        {"fj_sdmass", "vector<float>"},
        {"fj_trmass", "vector<float>"},
        {"fj_nparticles", "vector<float>"},
        {"fj_sophon_probXbb", "vector<float>"},
        {"fj_sophon_probXcc", "vector<float>"},
        {"fj_sophon_probXqq", "vector<float>"},
        {"fj_sophon_probXbc", "vector<float>"},
        {"fj_sophon_probXcs", "vector<float>"},
        {"fj_sophon_probXbq", "vector<float>"},
        {"fj_sophon_probXcq", "vector<float>"},
        {"fj_sophon_probXbqq", "vector<float>"},
        {"fj_sophon_probQCD", "vector<float>"},
        
        // Lepton variables (as vectors)
        {"lep_pt", "vector<float>"},
        {"lep_eta", "vector<float>"},
        {"lep_phi", "vector<float>"},
        {"lep_energy", "vector<float>"},
        {"lep_charge", "vector<int>"},
        {"lep_pid", "vector<int>"},
        {"lep_iso", "vector<float>"},

        // MET variables
        {"met_pt", "float"},
        {"met_phi", "float"},

        // Particle variables
        {"part_label", "vector<int>"},
        {"part_fjlabel", "vector<int>"}, // New: FatJet label for particles
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

    // Initialize EventData
    EventData data(branchList);
    data.setOutputBranch(tree);

    // Read input file
    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);
    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    Long64_t allEntries = treeReader->GetEntries();

    std::cerr << "** Input file: " << inputFile << std::endl;
    std::cerr << "** Output file: " << outputFile << std::endl;
    std::cerr << "** AK4 model path: " << modelPathAK4 << std::endl;
    std::cerr << "** FatJet model path: " << modelPathFatJet << std::endl;
    std::cerr << "** Jet branch: " << jetBranch << std::endl;
    std::cerr << "** FatJet branch: " << fatJetBranch << std::endl;
    std::cerr << "** Total events: " << allEntries << std::endl;

    // set branches
    TClonesArray *branchVertex = treeReader->UseBranch("Vertex");
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchPFCand = treeReader->UseBranch("ParticleFlowCandidate");
    TClonesArray *branchJet = treeReader->UseBranch(jetBranch);
    TClonesArray *branchFatJet = treeReader->UseBranch(fatJetBranch);
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");
    TClonesArray *branchMET = treeReader->UseBranch("PuppiMissingET");
    TClonesArray *branchWeight = treeReader->UseBranch("Weight");

    double jetR = 0.4;
    double fatJetR = fatJetBranch.Contains("AK15") ? 1.5 : 0.8;
    std::cerr << "jetR = " << jetR << std::endl;
    std::cerr << "fatJetR = " << fatJetR << std::endl;
    
    // Initialize JetGhostMatching
    JetGhostMatching ghostMatch(treeReader, jetR, 1e-18);
    
    // Initialize SophonAK4 helper
    OrtHelperSophonAK4 *sp4helper = nullptr;
    bool use_sophon_ak4 = !modelPathAK4.IsNull() && modelPathAK4 != "";
    
    if (use_sophon_ak4) {
        std::cerr << "Initializing SophonAK4 model..." << std::endl;
        sp4helper = new OrtHelperSophonAK4(modelPathAK4.Data(), false); // debug=false
        std::cerr << "SophonAK4 model initialized." << std::endl;
    } else {
        std::cerr << "No SophonAK4 model provided, will not use jet tagging." << std::endl;
    }
    
    // Initialize Sophon helper for FatJet
    OrtHelperSophon *sp8helper = nullptr;
    bool use_sophon_fatjet = !modelPathFatJet.IsNull() && modelPathFatJet != "";
    
    if (use_sophon_fatjet) {
        std::cerr << "Initializing Sophon FatJet model..." << std::endl;
        sp8helper = new OrtHelperSophon(modelPathFatJet.Data(), false); // debug=false
        std::cerr << "Sophon FatJet model initialized." << std::endl;
    } else {
        std::cerr << "No Sophon FatJet model provided, will not use fatjet tagging." << std::endl;
    }

    // Initialize GenPartGenericProcessor
    GenPartGenericProcessor* genhelper = new GenPartGenericProcessor(false); // debug=false

    // event loop
    int num_processed = 0;
    int num_pass_selection = 0;
    int num_pass_boosted_trigger = 0;
    
    for (Long64_t entry = 0; entry < allEntries; ++entry) {
        // if (entry != 433) {continue; }

        // std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
        
        if (entry % 100 == 0) {
            std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
        }

        // if (entry % 20 == 0) {
        //     std::cout << "Event " << entry << std::endl;
        //     system("cat /proc/self/status | grep VmRSS");
        //     system("free -m");
        // }

        treeReader->ReadEntry(entry);
        ++num_processed;

        // reset data
        data.reset();

        // if (entry >= 420 && entry < 440) {
        //     std::cerr << "processing " << entry << " of " << allEntries << " events." << std::endl;
        //     data.reset();
        //     std::cout << "Event " << entry << ": PFCands=" << branchPFCand->GetEntriesFast() 
        //               << ", GenParticles=" << branchParticle->GetEntriesFast() << std::endl;
        // }
        
        // Initialize boosted trigger variable
        data.intVars["pass_boosted_trigger"] = 0;
        
        // Calculate fatjet HT and leading fatjet mass
        double fjHT = 0;
        double leadfjMass = 0;
        
        for(Int_t idx_fatjet = 0; idx_fatjet < branchFatJet->GetEntriesFast(); ++idx_fatjet) {
            const Jet *fatjet = (Jet*) branchFatJet->At(idx_fatjet);
            
            // Apply pt and eta selection criteria
            if (fatjet->PT > 200 && std::abs(fatjet->Eta) < 2.5) {
                fjHT += fatjet->PT;
                
                // Record the first (highest pt) fatjet's trimmed mass
                if (leadfjMass == 0) {
                    leadfjMass = fatjet->TrimmedP4[0].M();
                }
            }
        }
        
        // Check if event passes boosted trigger
        if (fjHT > 800 && leadfjMass > 50) {
            data.intVars["pass_boosted_trigger"] = 1;
            ++num_pass_boosted_trigger;
        }

        // event selection
        bool pass_selection = false;
        float HT = 0;
        
        // find jets that pass selection criteria for HT calculation and event selection
        std::vector<std::pair<float, int>> selected_jets;
        for(Int_t idx_jet = 0; idx_jet < branchJet->GetEntriesFast(); ++idx_jet) {
            const Jet *jet = (Jet*) branchJet->At(idx_jet);
            if (jet->PT > 30 && std::abs(jet->Eta) < 2.5) {
                selected_jets.emplace_back(jet->PT, idx_jet);
                HT += jet->PT;
            }
        }
        
        // sort by pt
        std::sort(selected_jets.begin(), selected_jets.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        // Check if event passes selection criteria
        int num_selected_jets = selected_jets.size();
        const Jet *jet1 = num_selected_jets > 0 ? (Jet*)branchJet->At(selected_jets[0].second) : nullptr;
        const Jet *jet2 = num_selected_jets > 1 ? (Jet*)branchJet->At(selected_jets[1].second) : nullptr;
        const Jet *jet3 = num_selected_jets > 2 ? (Jet*)branchJet->At(selected_jets[2].second) : nullptr;
        const Jet *jet4 = num_selected_jets > 3 ? (Jet*)branchJet->At(selected_jets[3].second) : nullptr;

        if ((jet1 && jet1->PT > 75 && std::abs(jet1->Eta) < 2.5) && 
            (jet2 && jet2->PT > 60 && std::abs(jet2->Eta) < 2.5) && 
            (jet3 && jet3->PT > 45 && std::abs(jet3->Eta) < 2.5) && 
            (jet4 && jet4->PT > 40 && std::abs(jet4->Eta) < 2.5) && 
            (HT > 330)) {
            pass_selection = true;
        }
        data.intVars["pass_selection"] = pass_selection;
        data.floatVars["HT"] = HT;

        if (!pass_selection) {
            if (selectionLevel == "4j" || selectionLevel == "4j3b" || selectionLevel == "4j3bor2b") {
                std::cerr << "Event failed selection 4j criteria. This shouldn't happen if the delphes events have passed this filter" << std::endl;
                continue;
            }
        }

        // Create mapping for particles to jets and fatjets
        std::map<const TObject*, int> objectToIndexMap;
        std::map<const TObject*, int> objectToFatJetMap;
        
        // Get primary vertex for SophonAK4
        const Vertex *pv = (branchVertex != nullptr) ? ((Vertex *)branchVertex->At(0)) : nullptr;

        // Collect jets
        std::vector<Jet*> eventJets;
        for (Int_t i = 0; i < branchJet->GetEntriesFast(); ++i) {
            eventJets.push_back((Jet *)branchJet->At(i));
        }

        // Get GenParticle
        std::vector<GenParticle*> genParticles;
        for (Int_t j = 0; j < branchParticle->GetEntriesFast(); ++j) {
            genParticles.push_back((GenParticle *)branchParticle->At(j));
        }

        // Ghost Matching for AK4 jets
        std::vector<JetGhostMatching::JetGhostContent> ghostContents = 
            ghostMatch.getDetailedGhostContent(eventJets, genParticles);
        
        // Process ALL AK4 jets, regardless of selection
        for(Int_t idx_jet = 0; idx_jet < branchJet->GetEntriesFast(); ++idx_jet) {
            const Jet *jet = (Jet*) branchJet->At(idx_jet);
            const auto& ghostContent = ghostContents[idx_jet];
            processJet(jet, data, sp4helper, pv, &ghostContent);
            
            // Record jet components for particle labeling
            for (Int_t j = 0; j < jet->Constituents.GetEntriesFast(); ++j) {
                const TObject *object = jet->Constituents.At(j);
                if (object && object->IsA() == ParticleFlowCandidate::Class()) {
                    objectToIndexMap[object] = idx_jet;
                }
            }
        }
        
        // require 4j3b trigger criteria
        int num_loosebtagged_jet = 0;
        int num_tightbtagged_jet = 0;
        for (Int_t idx_seljet = 0; idx_seljet < std::min(4, (int)selected_jets.size()); ++idx_seljet) {
            Int_t idx_jet = selected_jets[idx_seljet].second;
            if (data.vfloatVars["jet_sophonAK4_probB"]->at(idx_jet) > 0.0243) {
                ++num_loosebtagged_jet;
            }
            if (data.vfloatVars["jet_sophonAK4_probB"]->at(idx_jet) > 0.643) {
                ++num_tightbtagged_jet;
            }
        }
        data.intVars["pass_4j3b_selection"] = (num_loosebtagged_jet >= 3) ? 1 : 0;
        data.intVars["pass_4j2b_selection"] = (num_tightbtagged_jet >= 2) ? 1 : 0;

        if (selectionLevel == "4j3b" && data.intVars["pass_4j3b_selection"] == 0) {
            continue;
        }
        if (selectionLevel == "4j3bor2b" && data.intVars["pass_4j3b_selection"] == 0 && data.intVars["pass_4j2b_selection"] == 0) {
            continue;
        }
        
        // Process ALL FatJets
        for(Int_t idx_fatjet = 0; idx_fatjet < branchFatJet->GetEntriesFast(); ++idx_fatjet) {
            const Jet *fatjet = (Jet*) branchFatJet->At(idx_fatjet);
            processFatJet(fatjet, data, sp8helper, pv);
            
            // Record fatjet components for particle labeling
            for (Int_t j = 0; j < fatjet->Constituents.GetEntriesFast(); ++j) {
                const TObject *object = fatjet->Constituents.At(j);
                if (object && object->IsA() == ParticleFlowCandidate::Class()) {
                    objectToFatJetMap[object] = idx_fatjet;
                }
            }
        }

        // Process ALL Leptons
        for (Int_t i = 0; i < branchElectron->GetEntries(); ++i) {
            const Electron *electron = (Electron *)branchElectron->At(i);
            processElectron(electron, data);
        }
        for (Int_t i = 0; i < branchMuon->GetEntries(); ++i) {
            const Muon *muon = (Muon *)branchMuon->At(i);
            processMuon(muon, data);
        }

        // Process MET
        const MissingET *met = (MissingET *)branchMET->At(0);
        data.floatVars["met_pt"] = met->MET;
        data.floatVars["met_phi"] = met->Phi;

        // PF candidates
        TLorentzVector pfcand_sum(0, 0, 0, 0);
        double pfcand_sum_HT = 0.0;
        
        // First loop to calculate the sum
        for(int i = 0; i < branchPFCand->GetEntriesFast(); ++i) {
            const ParticleFlowCandidate *pfcand = (ParticleFlowCandidate*)branchPFCand->At(i);
            
            if (std::abs(pfcand->Eta) > 5 || pfcand->PT <= 0) {
                continue;
            }
        
            TLorentzVector pfcand_p4;
            pfcand_p4.SetPtEtaPhiM(pfcand->PT, pfcand->Eta, pfcand->Phi, pfcand->Mass);
            pfcand_sum += pfcand_p4;
            pfcand_sum_HT += pfcand->PT;
        }
        
        // Store pfcand_sum variables
        data.floatVars["pfcand_sum_mass"] = pfcand_sum.M();
        data.floatVars["pfcand_sum_HT"] = pfcand_sum_HT;
        data.floatVars["pfcand_sum_pt"] = pfcand_sum.Pt();
        data.floatVars["pfcand_sum_eta"] = pfcand_sum.Eta();
        data.floatVars["pfcand_sum_phi"] = pfcand_sum.Phi();
        data.floatVars["pfcand_sum_energy"] = pfcand_sum.E();

        
        // Second loop to process individual particles with the sum information
        for(int i = 0; i < branchPFCand->GetEntriesFast(); ++i) {
            const ParticleFlowCandidate *pfcand = (ParticleFlowCandidate*)branchPFCand->At(i);

            if (std::abs(pfcand->Eta) > 5 || pfcand->PT <= 0) {
                continue;
            }
            
            // Get AK4 jet label
            int part_label = objectToIndexMap.count(pfcand) ? objectToIndexMap[pfcand] : -1;
            
            // Get FatJet label
            int part_fjlabel = objectToFatJetMap.count(pfcand) ? objectToFatJetMap[pfcand] : -1;
            
            // Add labels to vectors
            data.vintVars["part_label"]->push_back(part_label);
            data.vintVars["part_fjlabel"]->push_back(part_fjlabel);
            
            // Process particle
            processParticle(pfcand, data, pv, pfcand_sum);
        }

        // GenParticles
        // Build mother particle map
        std::map<int, std::vector<int>> motherMap;
        buildMotherMap(branchParticle, motherMap);

        // Store B-hadron info
        std::vector<std::tuple<int, int, bool, std::vector<std::vector<int>>>> bhadrons_info;

        
        // GenParticles
        int higgs_count = 0;
        TLorentzVector higgs1_p4, higgs2_p4;

        for(int i = 0; i < branchParticle->GetEntriesFast(); ++i) {
            const GenParticle *genparticle = (GenParticle*)branchParticle->At(i);
            processGenParticle(genparticle, i, data, branchParticle, higgs_count, higgs1_p4, higgs2_p4, bhadrons_info, motherMap);
        }

        // selected truth-particles info
        genhelper->process("generic", branchParticle);

        data.vfloatVars["genpart_pt"]->insert(data.vfloatVars["genpart_pt"]->end(), genhelper->getData().pt.begin(), genhelper->getData().pt.end());
        data.vfloatVars["genpart_eta"]->insert(data.vfloatVars["genpart_eta"]->end(), genhelper->getData().eta.begin(), genhelper->getData().eta.end());
        data.vfloatVars["genpart_phi"]->insert(data.vfloatVars["genpart_phi"]->end(), genhelper->getData().phi.begin(), genhelper->getData().phi.end());
        data.vfloatVars["genpart_energy"]->insert(data.vfloatVars["genpart_energy"]->end(), genhelper->getData().energy.begin(), genhelper->getData().energy.end());
        data.vintVars["genpart_pid"]->insert(data.vintVars["genpart_pid"]->end(), genhelper->getData().pid.begin(), genhelper->getData().pid.end());
        data.intVars["process_index"] = genhelper->getData().user_index;
        
        // Read generator weights
        std::vector<float> gen_weight_vec;
        int nWeights = branchWeight->GetEntriesFast();
        for (int iw = 0; iw < nWeights; ++iw) {
            const Weight* weight = (Weight*)branchWeight->At(iw);
            gen_weight_vec.push_back(weight->Weight);
        }
        data.vfloatVars["gen_weight"]->insert(data.vfloatVars["gen_weight"]->end(), gen_weight_vec.begin(), gen_weight_vec.end());

        tree->Fill();
        ++num_pass_selection;

    } // end event loop

    tree->Write();
    std::cerr << TString::Format("** Written %d events to output %s; %d events passing customized selection, %d events passing boosted trigger", 
                                num_processed, outputFile.Data(), num_pass_selection, num_pass_boosted_trigger) << std::endl;

    // Clean up
    if (sp4helper) {
        delete sp4helper;
    }
    if (sp8helper) {
        delete sp8helper;
    }
    if (genhelper) {
        delete genhelper;
    }
    
    delete treeReader;
    delete chain;
    delete fout;
}
