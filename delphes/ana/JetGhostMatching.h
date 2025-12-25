#ifndef JETGHOSTMATCHING_H
#define JETGHOSTMATCHING_H

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include "Math/LorentzVector.h"
#include <Math/Vector4D.h>
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "ParticleInfo.h"

// Add FastJet headers
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

class JetGhostMatching {
public:
  // Enum for ghost particle type identification
  enum GhostType {
    GHOST_NORMAL = 0,
    GHOST_BHADRON = 1,
    GHOST_CHADRON = 2,
    GHOST_LEPTON = 3,
    GHOST_PARTON = 4
  };
  
  // Result storage structure
  struct JetGhostContent {
    // B/C hadron counting
    int nBHadron, nCHadron;
    
    // Lepton counting
    int nElectron, nMuon, nTau;
    
    // Parton counting
    int nBParton, nBbarParton;
    int nCParton, nCbarParton;
    int nSParton, nSbarParton;
    int nUParton, nUbarParton;
    int nDParton, nDbarParton;
    int nGParton;
    
    // Flavor labels
    std::string hadronFlavorLabel;
    std::string partonFlavorLabel;
    
    // Flavor numeric codes
    int hadronFlavor; // 5 for b, 4 for c, 0 for light
    int partonFlavor; // Corresponds to index in labels_ vector
    
    // Matched particles
    std::vector<const GenParticle*> matchedBHadrons;
    std::vector<const GenParticle*> matchedCHadrons;
    std::vector<const GenParticle*> matchedLeptons;
    std::vector<const GenParticle*> matchedPartons;
    
    // Debug info for ghost matching validation
    fastjet::PseudoJet ghostMatchedJet;
    bool hasValidGhostJet;
    double deltaR_to_original;
    double pt_ratio;
    double eta_diff;
    double phi_diff;
    double mass_ratio;
    
    // Distance info for matched particles
    struct ParticleDistanceInfo {
      const GenParticle* particle;
      double deltaR;
      std::string type;
      
      ParticleDistanceInfo(const GenParticle* p, double dr, const std::string& t) 
        : particle(p), deltaR(dr), type(t) {}
      
      bool operator<(const ParticleDistanceInfo& other) const {
        return deltaR < other.deltaR;
      }
    };
    
    std::vector<ParticleDistanceInfo> particleDistances;
    
    // Constructor, initialize all counters to 0
    JetGhostContent() : 
      nBHadron(0), nCHadron(0),
      nElectron(0), nMuon(0), nTau(0),
      nBParton(0), nBbarParton(0),
      nCParton(0), nCbarParton(0),
      nSParton(0), nSbarParton(0),
      nUParton(0), nUbarParton(0),
      nDParton(0), nDbarParton(0),
      nGParton(0),
      hadronFlavorLabel("light"),
      partonFlavorLabel("Invalid"),
      hadronFlavor(0),
      partonFlavor(-1),
      hasValidGhostJet(false),
      deltaR_to_original(999.0),
      pt_ratio(0.0),
      eta_diff(999.0),
      phi_diff(999.0),
      mass_ratio(0.0) {}
  };
  
  // Constructor
  JetGhostMatching(ExRootTreeReader* reader, double jetR = 0.4, double ghostRescaling = 1e-18) 
    : reader_(reader), jetR_(jetR), ghostRescaling_(ghostRescaling) {
    // Initialize branches
    branchParticle_ = reader_->UseBranch("Particle");
  }
  
  // Modified to include all jets at once
  std::vector<JetGhostContent> getDetailedGhostContent(const std::vector<Jet*>& jets, const std::vector<GenParticle*>& genParticles) {
    // Identify B/C hadrons
    std::vector<const GenParticle*> bHadrons, cHadrons;
    identifyHeavyHadrons(genParticles, bHadrons, cHadrons);
    
    // Identify leptons (e/μ/τ with status==23)
    std::vector<const GenParticle*> leptons;
    identifyLeptons(genParticles, leptons);
    
    // Identify partons
    std::vector<const GenParticle*> partons;
    identifyPartons(genParticles, partons);
    
    // Perform ghost matching for all jets at once
    return performGhostMatching(jets, bHadrons, cHadrons, leptons, partons);
  }
  
  // Processing for single jet
  // JetGhostContent getDetailedGhostContent(Jet* jet, const std::vector<GenParticle*>& genParticles) {
  //   std::vector<Jet*> jets = {jet};
  //   std::vector<JetGhostContent> results = getDetailedGhostContent(jets, genParticles);
    
  //   if (!results.empty()) {
  //     return results[0];
  //   }
    
  //   // Return empty result
  //   return JetGhostContent();
  // }
  
  // Find hadron flavor index
  int getHadronFlavorCode(const std::string& label) const {
    if (label == "b") return 5;
    if (label == "c") return 4;
    return 0; // light
  }
  
  // Find parton flavor index
  int getPartonFlavorCode(const std::string& label) const {
    auto it = std::find(labels_.begin(), labels_.end(), label);
    if (it != labels_.end()) {
      return std::distance(labels_.begin(), it);
    }
    return -1; // Return -1 if label not found
  }

  // Get particles that determine hadron flavor label
  std::vector<const GenParticle*> getHadronFlavorParticles(const JetGhostContent& content) {
    std::vector<const GenParticle*> result;
    
    // For b-jet, return the highest PT b-hadron
    if (content.nBHadron > 0 && !content.matchedBHadrons.empty()) {
      auto sortedBHadrons = sortByPT(content.matchedBHadrons);
      result.push_back(sortedBHadrons[0]);
    }
    // For c-jet, return the highest PT c-hadron
    else if (content.nCHadron > 0 && !content.matchedCHadrons.empty()) {
      auto sortedCHadrons = sortByPT(content.matchedCHadrons);
      result.push_back(sortedCHadrons[0]);
    }
    // For light jet, return empty vector
    
    return result;
  }

  // Get particles that determine parton flavor label
  std::vector<const GenParticle*> getPartonFlavorParticles(const JetGhostContent& content) {
    std::vector<const GenParticle*> result;
    
    // Combine all matched leptons and partons
    std::vector<const GenParticle*> allParticles;
    allParticles.insert(allParticles.end(), content.matchedLeptons.begin(), content.matchedLeptons.end());
    allParticles.insert(allParticles.end(), content.matchedPartons.begin(), content.matchedPartons.end());
    
    // If no particles matched, return empty vector
    if (allParticles.empty()) {
      return result;
    }
    
    // Sort all particles by PT
    std::vector<const GenParticle*> sortedParticles = sortByPT(allParticles);
    
    // Check for particle pair matching
    if (sortedParticles.size() >= 2) {
      const GenParticle* p1 = sortedParticles[0];
      const GenParticle* p2 = sortedParticles[1];
      
      // Check for electron pair
      if (isElectron(p1) && isElectron(p2) && p1->PID * p2->PID < 0) {
        result.push_back(p1);
        result.push_back(p2);
        return result;
      }
      
      // Check for muon pair
      if (isMuon(p1) && isMuon(p2) && p1->PID * p2->PID < 0) {
        result.push_back(p1);
        result.push_back(p2);
        return result;
      }
      
      // Check for tau pair
      if (isTau(p1) && isTau(p2) && p1->PID * p2->PID < 0) {
        // For tau pairs, add their decay products
        auto tau1Daughters = getTauDaughters(p1);
        auto tau2Daughters = getTauDaughters(p2);
        result.insert(result.end(), tau1Daughters.begin(), tau1Daughters.end());
        result.insert(result.end(), tau2Daughters.begin(), tau2Daughters.end());
        return result;
      }
      
      // Check for quark pair
      if (abs(p1->PID) >= 1 && abs(p1->PID) <= 5 && 
          abs(p2->PID) >= 1 && abs(p2->PID) <= 5 && 
          p1->PID == -p2->PID) {
        result.push_back(p1);
        result.push_back(p2);
        return result;
      }
      
      // Check for gluon pair
      if (p1->PID == 21 && p2->PID == 21) {
        result.push_back(p1);
        result.push_back(p2);
        return result;
      }
    }
    
    // If no pair matching, use leading particle
    const GenParticle* leadingParticle = sortedParticles[0];
    
    // Check for single lepton
    if (isElectron(leadingParticle) || isMuon(leadingParticle)) {
      result.push_back(leadingParticle);
    }
    // Check for single tau
    else if (isTau(leadingParticle)) {
      auto tauDaughters = getTauDaughters(leadingParticle);
      result.insert(result.end(), tauDaughters.begin(), tauDaughters.end());
    }
    // Check for single quark or gluon
    else if ((abs(leadingParticle->PID) >= 1 && abs(leadingParticle->PID) <= 5) || leadingParticle->PID == 21) {
      result.push_back(leadingParticle);
    }
    
    return result;
  }

private:
  ExRootTreeReader* reader_;
  TClonesArray* branchParticle_;
  double jetR_;
  double ghostRescaling_;
  
  // Define labels list for parton flavor
  const std::vector<std::string> labels_ = {
    "b", "bbar", "c", "cbar", "s", "sbar", "d", "dbar", "u", "ubar", "g", 
    "em", "ep", "mm", "mp", "tauhm", "tauhp",
    "bbbar", "ccbar", "ssbar", "ddbar", "uubar", "gg", 
    "epem", "mpmm", "tauhptauhm"
  };
  
  // Get final State of a particle (for selecting tauh)
  const GenParticle* getFinalState(const GenParticle* particle) {
    if (!particle)
      return nullptr;
      
    const GenParticle* final = particle;
    
    while (final->D1 >= 0) {
      const GenParticle* chain = nullptr;
      for (int idau = final->D1; idau <= final->D2; ++idau) {
        if (idau >= 0 && idau < branchParticle_->GetEntriesFast()) {
          GenParticle* daughter = (GenParticle*)branchParticle_->At(idau);
          if (daughter->PID == particle->PID) {
            chain = daughter;
            break;
          }
        }
      }
      if (!chain)
        break;
      final = chain;
    }
    
    return final;
  }
  
  // Check if particle is a B hadron
  bool hasBottom(const GenParticle* particle) {
    int absPID = abs(particle->PID);
    int code1 = (absPID / 100) % 10;
    int code2 = (absPID / 1000) % 10;
    return (code1 == 5 || code2 == 5);
  }
  
  // Check if particle is a C hadron
  bool hasCharm(const GenParticle* particle) {
    int absPID = abs(particle->PID);
    int code1 = (absPID / 100) % 10;
    int code2 = (absPID / 1000) % 10;
    return (code1 == 4 || code2 == 4);
  }
  
  // Check if particle is an electron
  bool isElectron(const GenParticle* particle) {
    return (abs(particle->PID) == 11);
  }
  
  // Check if particle is a muon
  bool isMuon(const GenParticle* particle) {
    return (abs(particle->PID) == 13);
  }
  
  // Check if particle is a tau
  bool isTau(const GenParticle* particle) {
    return (abs(particle->PID) == 15);
  }

  // Check if tau is a hadronic tau
  bool isHadronicTau(const GenParticle* particle) {
    const GenParticle* tau = getFinalState(particle);
    
    for (int i = tau->D1; i <= tau->D2; ++i) {
      if (i >= 0 && i < branchParticle_->GetEntriesFast()) {
        GenParticle* daughter = (GenParticle*)branchParticle_->At(i);
        if (abs(daughter->PID) == 11 || abs(daughter->PID) == 13) {
          return false;
        }
      }
    }
  
    return true;
  }

  // Get tau decay products
  std::vector<const GenParticle*> getTauDaughters(const GenParticle* particle) {
    const GenParticle* tau = getFinalState(particle);
    std::vector<const GenParticle*> daughters;
  
    for (int i = tau->D1; i <= tau->D2; ++i) {
      if (i >= 0 && i < branchParticle_->GetEntriesFast()) {
        GenParticle* daughter = (GenParticle*)branchParticle_->At(i);
        daughters.push_back(daughter);
      }
    }
  
    return daughters;
  }
  
  // Identify B/C hadrons - based on CMS implementation
  void identifyHeavyHadrons(const std::vector<GenParticle*>& genParticles,
                          std::vector<const GenParticle*>& bHadrons,
                          std::vector<const GenParticle*>& cHadrons) {
    for (const GenParticle* particle : genParticles) {
      // Add PT>0 condition
      if (particle->PT <= 0) continue;
      
      // Check if it's a B hadron
      if (hasBottom(particle)) {
        // Check if it has a B hadron daughter
        bool hasBHadronDaughter = false;
        for (int i = particle->D1; i <= particle->D2; ++i) {
          if (i >= 0 && i < branchParticle_->GetEntriesFast()) {
            GenParticle* daughter = (GenParticle*)branchParticle_->At(i);
            if (hasBottom(daughter)) {
              hasBHadronDaughter = true;
              break;
            }
          }
        }
        
        // Only select B hadrons without B hadron daughters
        if (!hasBHadronDaughter) {
          bHadrons.push_back(particle);
        }
      }
      
      // Check if it's a C hadron
      if (hasCharm(particle)) {
        // Check if it has a C hadron daughter
        bool hasCHadronDaughter = false;
        for (int i = particle->D1; i <= particle->D2; ++i) {
          if (i >= 0 && i < branchParticle_->GetEntriesFast()) {
            GenParticle* daughter = (GenParticle*)branchParticle_->At(i);
            if (hasCharm(daughter)) {
              hasCHadronDaughter = true;
              break;
            }
          }
        }
        
        // Only select C hadrons without C hadron daughters
        if (!hasCHadronDaughter) {
          cHadrons.push_back(particle);
        }
      }
    }
  }
  
  // Identify leptons - modified to use status==23 and require hadronic taus
  void identifyLeptons(const std::vector<GenParticle*>& genParticles,
                      std::vector<const GenParticle*>& leptons) {
    for (const GenParticle* particle : genParticles) {
      // Add PT>0 condition
      if (particle->PT <= 0) continue;
      
      // Select leptons with status==23
      if (particle->Status == 23) {
        if (isElectron(particle) || isMuon(particle)) {
          leptons.push_back(particle);
        } 
        // select hadronic tau
        else if (isTau(particle) && isHadronicTau(particle)) {
          leptons.push_back(particle);
        }
      }
    }
  }
  
  // Identify partons
  void identifyPartons(const std::vector<GenParticle*>& genParticles,
                      std::vector<const GenParticle*>& partons) {
    for (const GenParticle* particle : genParticles) {
      // Add PT>0 condition
      if (particle->PT <= 0) continue;
      
      // Select particles with status==23
      if (particle->Status == 23) {
        int pid = abs(particle->PID);
        // Collect quarks and gluons
        if ((pid >= 1 && pid <= 6) || pid == 21) {
          partons.push_back(particle);
        }
      }
    }
  }
  
  // Sort particles by PT (descending)
  template <typename T>
  std::vector<T> sortByPT(const std::vector<T>& particles) {
    std::vector<T> sorted = particles;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
      return a->PT > b->PT;
    });
    return sorted;
  }
  
    // Refactored Ghost Matching method, processing all jets at once
    std::vector<JetGhostContent> performGhostMatching(
      const std::vector<Jet*>& jets,
      const std::vector<const GenParticle*>& bHadrons,
      const std::vector<const GenParticle*>& cHadrons,
      const std::vector<const GenParticle*>& leptons,
      const std::vector<const GenParticle*>& partons) {
      
      std::vector<JetGhostContent> results(jets.size());
      
      // 1. Prepare all FastJet inputs, including constituent particles of all jets and ghost particles
      std::vector<fastjet::PseudoJet> inputs;
      
      // Add constituent particles for each jet
      for (size_t jetIdx = 0; jetIdx < jets.size(); ++jetIdx) {
        Jet* jet = jets[jetIdx];
        
        // Add jet constituent particles
        for (int i = 0; i < jet->Constituents.GetEntriesFast(); ++i) {
          TObject* constituent = jet->Constituents.At(i);
          if (constituent->IsA() == ParticleFlowCandidate::Class()) {
            ParticleFlowCandidate* pfcand = (ParticleFlowCandidate*)constituent;
            TLorentzVector p4 = pfcand->P4();
            fastjet::PseudoJet pj(p4.Px(), p4.Py(), p4.Pz(), p4.E());
            pj.set_user_index(jetIdx); // Store which jet this particle belongs to
            inputs.push_back(pj);
          }
        }
      }
      
      // Add all leptons as ghost particles
      for (size_t i = 0; i < leptons.size(); ++i) {
        const GenParticle* lepton = leptons[i];
        fastjet::PseudoJet ghost(lepton->Px, lepton->Py, lepton->Pz, lepton->E);
        ghost *= ghostRescaling_;
        // Use GHOST_LEPTON as type marker and store index
        ghost.set_user_index(-GHOST_LEPTON * 10000 - i);
        inputs.push_back(ghost);
      }
      
      // Add all B/C hadrons as ghost particles
      for (size_t i = 0; i < bHadrons.size(); ++i) {
        const GenParticle* hadron = bHadrons[i];
        fastjet::PseudoJet ghost(hadron->Px, hadron->Py, hadron->Pz, hadron->E);
        ghost *= ghostRescaling_;
        // Use GHOST_BHADRON as type marker and store index
        ghost.set_user_index(-GHOST_BHADRON * 10000 - i);
        inputs.push_back(ghost);
      }
      
      for (size_t i = 0; i < cHadrons.size(); ++i) {
        const GenParticle* hadron = cHadrons[i];
        fastjet::PseudoJet ghost(hadron->Px, hadron->Py, hadron->Pz, hadron->E);
        ghost *= ghostRescaling_;
        // Use GHOST_CHADRON as type marker and store index
        ghost.set_user_index(-GHOST_CHADRON * 10000 - i);
        inputs.push_back(ghost);
      }
      
      // Add all partons as ghost particles
      for (size_t i = 0; i < partons.size(); ++i) {
        const GenParticle* parton = partons[i];
        fastjet::PseudoJet ghost(parton->Px, parton->Py, parton->Pz, parton->E);
        ghost *= ghostRescaling_;
        // Use GHOST_PARTON as type marker and store index
        ghost.set_user_index(-GHOST_PARTON * 10000 - i);
        inputs.push_back(ghost);
      }
      
      // 2. Perform jet clustering
      fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, jetR_);
      fastjet::ClusterSequence cs(inputs, jetDef);
      std::vector<fastjet::PseudoJet> ghostJets = cs.inclusive_jets();
      
      // 3. Find corresponding reconstructed ghostJet for each original jet
      for (size_t jetIdx = 0; jetIdx < jets.size(); ++jetIdx) {
        Jet* originalJet = jets[jetIdx];
        JetGhostContent& content = results[jetIdx];
        
        // Find the ghostJet that best matches the original jet
        fastjet::PseudoJet* matchedJet = nullptr;
        double minDR = 999.0;
        
        for (auto& ghostJet : ghostJets) {
          double dR = deltaR(originalJet->Eta, originalJet->Phi, ghostJet.eta(), ghostJet.phi());
          if (dR < 0.2 && dR < minDR) { // Use small matching radius to ensure correct match
            minDR = dR;
            matchedJet = &ghostJet;
          }
        }
        
        // 4. Calculate ghost particles in each jet
        if (matchedJet) {
          // Store matched jet for debugging
          content.ghostMatchedJet = *matchedJet;
          content.hasValidGhostJet = true;
          content.deltaR_to_original = minDR;
          content.pt_ratio = matchedJet->pt() / originalJet->PT;
          content.eta_diff = matchedJet->eta() - originalJet->Eta;
          content.phi_diff = deltaPhi(matchedJet->phi(), originalJet->Phi);
          content.mass_ratio = (matchedJet->m() > 0 && originalJet->Mass > 0) ? 
                             (matchedJet->m() / originalJet->Mass) : 0.0;
          
          std::vector<fastjet::PseudoJet> constituents = matchedJet->constituents();
          
          for (const auto& constituent : constituents) {
            int userIndex = constituent.user_index();
            
            // Skip normal particles
            if (userIndex >= 0) continue;
            
            // Parse ghost type and index
            int ghostType = -userIndex / 10000;
            int ghostIndex = -userIndex % 10000;
            
            // Handle leptons
            if (ghostType == GHOST_LEPTON && ghostIndex < (int)leptons.size()) {
              const GenParticle* lepton = leptons[ghostIndex];
              content.matchedLeptons.push_back(lepton);
              
              // Calculate deltaR between lepton and jet
              double dr = deltaR(matchedJet->eta(), matchedJet->phi(), lepton->Eta, lepton->Phi);
              content.particleDistances.emplace_back(lepton, dr, "Lepton");
              
              if (isElectron(lepton)) {
                content.nElectron++;
              } else if (isMuon(lepton)) {
                content.nMuon++;
              } else if (isTau(lepton)) {
                content.nTau++;
              }
            }
            // Handle B hadrons
            else if (ghostType == GHOST_BHADRON && ghostIndex < (int)bHadrons.size()) {
              const GenParticle* hadron = bHadrons[ghostIndex];
              content.nBHadron++;
              content.matchedBHadrons.push_back(hadron);
              
              // Calculate deltaR between B hadron and jet
              double dr = deltaR(matchedJet->eta(), matchedJet->phi(), hadron->Eta, hadron->Phi);
              content.particleDistances.emplace_back(hadron, dr, "B-Hadron");
            }
            // Handle C hadrons
            else if (ghostType == GHOST_CHADRON && ghostIndex < (int)cHadrons.size()) {
              const GenParticle* hadron = cHadrons[ghostIndex];
              content.nCHadron++;
              content.matchedCHadrons.push_back(hadron);
              
              // Calculate deltaR between C hadron and jet
              double dr = deltaR(matchedJet->eta(), matchedJet->phi(), hadron->Eta, hadron->Phi);
              content.particleDistances.emplace_back(hadron, dr, "C-Hadron");
            }
            // Handle partons
            else if (ghostType == GHOST_PARTON && ghostIndex < (int)partons.size()) {
              const GenParticle* parton = partons[ghostIndex];
              content.matchedPartons.push_back(parton);
              
              // Calculate deltaR between parton and jet
              double dr = deltaR(matchedJet->eta(), matchedJet->phi(), parton->Eta, parton->Phi);
              content.particleDistances.emplace_back(parton, dr, "Parton");
              
              int pid = parton->PID;
              switch (abs(pid)) {
                case 5:  // b quark
                  (pid > 0) ? content.nBParton++ : content.nBbarParton++;
                  break;
                case 4:  // c quark
                  (pid > 0) ? content.nCParton++ : content.nCbarParton++;
                  break;
                case 3:  // s quark
                  (pid > 0) ? content.nSParton++ : content.nSbarParton++;
                  break;
                case 2:  // u quark
                  (pid > 0) ? content.nUParton++ : content.nUbarParton++;
                  break;
                case 1:  // d quark
                  (pid > 0) ? content.nDParton++ : content.nDbarParton++;
                  break;
                case 21: // gluon
                  content.nGParton++;
                  break;
              }
            }
          }
          
          // Sort particle distances by deltaR
          std::sort(content.particleDistances.begin(), content.particleDistances.end());
          
          // 5. Determine hadron flavor label - CMS style
          content.hadronFlavorLabel = determineHadronFlavor(content);
          content.hadronFlavor = getHadronFlavorCode(content.hadronFlavorLabel);
          
          // 6. Determine parton flavor label - based on all matched particles
          content.partonFlavorLabel = determinePartonFlavor(content);
          content.partonFlavor = getPartonFlavorCode(content.partonFlavorLabel);
        }
      }
      
      return results;
    }

  
  // Determine hadron flavor label - CMS style
  std::string determineHadronFlavor(const JetGhostContent& content) {
    // CMS style hadron flavor:
    // 1. If there's at least one B hadron, it's a b jet
    if (content.nBHadron > 0) {
      return "b";
    }
    
    // 2. If there's at least one C hadron, it's a c jet
    if (content.nCHadron > 0) {
      return "c";
    }
    
    // 3. Otherwise it's a light jet
    return "light";
  }
  
  // Determine parton flavor label - based on all matched particles
  std::string determinePartonFlavor(const JetGhostContent& content) {
    // Combine all leptons and partons for parton flavor determination
    std::vector<const GenParticle*> allParticles;
    
    // Add all matched particles
    allParticles.insert(allParticles.end(), content.matchedLeptons.begin(), content.matchedLeptons.end());
    allParticles.insert(allParticles.end(), content.matchedPartons.begin(), content.matchedPartons.end());
    
    // If no particles matched, return Invalid
    if (allParticles.empty()) {
      return "Invalid";
    }
    
    // Sort all particles by PT
    std::vector<const GenParticle*> sortedParticles = sortByPT(allParticles);
    
    // Priority 1: Lepton-based tags - check patterns
    // Check for lepton pairs if we have at least 2 particles
    if (sortedParticles.size() >= 2) {
      const GenParticle* p1 = sortedParticles[0];
      const GenParticle* p2 = sortedParticles[1];
      
      // Check for electron pair
      if (isElectron(p1) && isElectron(p2) && p1->PID * p2->PID < 0) {
        return "epem";
      }
      
      // Check for muon pair
      if (isMuon(p1) && isMuon(p2) && p1->PID * p2->PID < 0) {
        return "mpmm";
      }
      
      // Check for tau pair
      if (isTau(p1) && isTau(p2) && p1->PID * p2->PID < 0) {
        return "tauhptauhm";
      }
      
      // Check for b quark pair
      if (abs(p1->PID) == 5 && abs(p2->PID) == 5 && p1->PID * p2->PID < 0) {
        return "bbbar";
      }
      
      // Check for c quark pair
      if (abs(p1->PID) == 4 && abs(p2->PID) == 4 && p1->PID * p2->PID < 0) {
        return "ccbar";
      }
      
      // Check for s quark pair
      if (abs(p1->PID) == 3 && abs(p2->PID) == 3 && p1->PID * p2->PID < 0) {
        return "ssbar";
      }
      
      // Check for d quark pair
      if (abs(p1->PID) == 1 && abs(p2->PID) == 1 && p1->PID * p2->PID < 0) {
        return "ddbar";
      }
      
      // Check for u quark pair
      if (abs(p1->PID) == 2 && abs(p2->PID) == 2 && p1->PID * p2->PID < 0) {
        return "uubar";
      }
      
      // Check for gluon pair
      if (p1->PID == 21 && p2->PID == 21) {
        return "gg";
      }
    }
    
    // If no pair match, use the leading particle
    const GenParticle* leadingParticle = sortedParticles[0];
    
    // Check for leptons
    if (isElectron(leadingParticle)) {
      return (leadingParticle->PID > 0) ? "em" : "ep";
    } else if (isMuon(leadingParticle)) {
      return (leadingParticle->PID > 0) ? "mm" : "mp";
    } else if (isTau(leadingParticle)) {
      return (leadingParticle->PID > 0) ? "tauhm" : "tauhp";
    }
    
    // Check for partons
    switch (leadingParticle->PID) {
      case 5: return "b";
      case -5: return "bbar";
      case 4: return "c";
      case -4: return "cbar";
      case 3: return "s";
      case -3: return "sbar";
      case 2: return "u";
      case -2: return "ubar";
      case 1: return "d";
      case -1: return "dbar";
      case 21: return "g";
    }
    
    // Default label
    return "Invalid";
  }
};

#endif


