#ifndef GenPartProcessor_h
#define GenPartProcessor_h

#include <iostream>
#include <cassert>
#include <unordered_set>
#include <utility>
#include "TClonesArray.h"
#include "Math/LorentzVector.h"
#include <Math/Vector4D.h>
#include "classes/DelphesClasses.h"

#include "ParticleID.h"
#include "ParticleInfo.h"

class GenPartProcessor {

public:
    struct genPartData {
        std::vector<float> pt;
        std::vector<float> eta;
        std::vector<float> phi;
        std::vector<float> energy;
        std::vector<int> pid;
        std::vector<int> status;
        int user_index;
    };
    genPartData& getData() { return data_; }
    void clearResult() {
        data_.pt.clear();
        data_.eta.clear();
        data_.phi.clear();
        data_.energy.clear();
        data_.pid.clear();
        data_.status.clear();
        data_.user_index = 0;
    }

public:
    GenPartProcessor() {}
    GenPartProcessor(bool debug) : debug_(debug) {}

    virtual ~GenPartProcessor() {}

    void process(TString mode, const TClonesArray *branchParticle) {
        // Process the gen particles with the given mode

        genParticles_.clear();
        for (Int_t i = 0; i < branchParticle->GetEntriesFast(); ++i) {
            genParticles_.push_back((GenParticle *)branchParticle->At(i));
        }
        processed_.clear();

        if (debug_) {
        //   printGenInfoHeader();
        //   for (unsigned ipart = 0; ipart < genParticles_.size(); ++ipart) {
        //     printGenParticleInfo(genParticles_[ipart], ipart);
        //   }
        }

        clearResult();
        for (const auto *gp : genParticles_) {
            if (processed_.count(gp))
                continue;
            processed_.insert(gp);

            if (mode == "wcbana") {
                findTopDecay(gp);
                findWZDecay(gp);
            }
            else
                throw std::invalid_argument("[GenPartProcessor::process] Invalid mode!");
        }

        if (genParticles_.size() != processed_.size())
            throw std::logic_error("[GenPartProcessor] Not all genParticles are processed!");
    }

private:

    void findTopDecay(const GenParticle *particle) {
        // Identify the top quark and store top + W + W daus particles

        auto pdgid = std::abs(particle->PID);
        if (pdgid == ParticleID::p_t) {
            auto final = getFinal(particle);
            storeGenParticle(final); // save the top quark

            auto daus = getDaughters(final);
            GenParticle *b = nullptr, *w = nullptr;
            for (const auto & dau: daus){
                auto dpid = std::abs(dau->PID);
                if (dpid == ParticleID::p_b)  b = const_cast<GenParticle*>(dau);
                if (dpid == ParticleID::p_Wplus || dpid == ParticleID::p_Hplus)  w = const_cast<GenParticle*>(dau); // special treatment: also identify top->bH+ decay
            }
            storeGenParticle(b); // save the b quark
            storeGenParticle(w); // save the W boson

            if (w) {
                auto wfinal = getFinal(w);
                auto wdaus = getDaughters(wfinal);
                for (const auto & wdau: wdaus){
                    storeGenParticle(wdau); // save the W daughters
                    if (std::abs(wdau->PID) == ParticleID::p_b)
                        getData().user_index = 1; // found W->cb decay
                }
            }
            else {
                std::cerr << "[GenPartProcessor::find_top] W boson not found!" << std::endl;
                storeGenParticle(nullptr);
                storeGenParticle(nullptr);
            }
        }
    }

    void findWZDecay(const GenParticle *particle) {
        // Identify the W boson and store W/Z + W/Z daus particles

        auto pdgid = std::abs(particle->PID);
        if (pdgid == ParticleID::p_Wplus || pdgid == ParticleID::p_Z0) {
            auto final = getFinal(particle);
            storeGenParticle(final); // save the W/Z boson

            auto daus = getDaughters(final);
            for (const auto & dau: daus){
                storeGenParticle(dau); // save the W/Z daughters
                if (pdgid == ParticleID::p_Wplus && std::abs(dau->PID) == ParticleID::p_b)
                    getData().user_index = 1; // found W->cb decay
            }
        }
    }

    void storeGenParticle(const GenParticle *particle) {
        // Store the gen particles
        data_.pt.push_back(particle->PT);
        data_.eta.push_back(particle->Eta);
        data_.phi.push_back(particle->Phi);
        data_.energy.push_back(particle->P4().Energy());
        data_.pid.push_back(particle->PID);
    }

private:
    void printGenInfoHeader() const {
        using namespace std;
        cout << right << setw(6) << "#"
            << " " << setw(10) << "pdgId"
            << "  "
            << "Chg"
            << "  " << setw(10) << "Mass"
            << "  " << setw(48) << " Momentum" << left << "  " << setw(10) << "Mothers"
            << " " << setw(30) << "Daughters" << endl;
    }

    void printGenParticleInfo(const GenParticle *genParticle, const int idx) const {
        using namespace std;
        cout << right << setw(3) << genParticle->Status;
        cout << right << setw(3) << idx << " " << setw(10) << genParticle->PID << "  ";
        cout << right << "  " << setw(3) << genParticle->Charge << "  "
             << TString::Format("%10.3g", genParticle->Mass < 1e-5 ? 0 : genParticle->Mass);
        cout << left << setw(50)
             << TString::Format("  (E=%6.4g pT=%6.4g eta=%7.3g phi=%7.3g)",
                                genParticle->P4().Energy(),
                                genParticle->PT,
                                genParticle->Eta,
                                genParticle->Phi);

        TString mothers;
        if (genParticle->M1 >= 0) {
            mothers += genParticle->M1;
        }
        if (genParticle->M2 >= 0) {
            mothers += ",";
            mothers += genParticle->M2;
        }
        cout << "  " << setw(10) << mothers;

        TString daughters;
        for (int iDau = genParticle->D1; iDau <= genParticle->D2; ++iDau) {
            if (daughters.Length())
                daughters += ",";
            daughters += iDau;
        }
        cout << " " << setw(30) << daughters << endl;
    }

    const GenParticle *getFinal(const GenParticle *particle) {
        // will mark intermediate particles as processed
        if (!particle)
            return nullptr;
        processed_.insert(particle);
        const GenParticle *final = particle;

        while (final->D1 >= 0) {
            const GenParticle *chain = nullptr;
            for (int idau = final->D1; idau <= final->D2; ++idau) {
                if (genParticles_.at(idau)->PID == particle->PID) {
                    chain = genParticles_.at(idau);
                    processed_.insert(chain);
                    break;
                }
            }
            if (!chain)
                break;
            final = chain;
        }
        return final;
    }

    bool isHadronic(const GenParticle *particle, bool allow_gluon = false) const {
        // particle needs to be the final version before decay
        if (!particle)
            throw std::invalid_argument("[GenPartProcessor::isHadronic()] Null particle!");
        for (const auto *dau : getDaughters(particle)) {
            auto pdgid = std::abs(dau->PID);
            if (pdgid >= ParticleID::p_d && pdgid <= ParticleID::p_b)
                return true;
            if (allow_gluon && pdgid == ParticleID::p_g)
                return true;
        }
        return false;
    }

    std::vector<const GenParticle *> getDaughters(const GenParticle *particle) const {
        std::vector<const GenParticle *> daughters;
        for (int idau = particle->D1; idau <= particle->D2; ++idau) {
            daughters.push_back(genParticles_.at(idau));
        }
        return daughters;
    }

    std::vector<const GenParticle *> getDaughterQuarks(const GenParticle *particle, bool allow_gluon = false) {
        std::vector<const GenParticle *> daughters;
        for (int idau = particle->D1; idau <= particle->D2; ++idau) {
            const auto *dau = genParticles_.at(idau);
            auto pdgid = std::abs(dau->PID);
            if (pdgid >= ParticleID::p_d && pdgid <= ParticleID::p_b) {
                daughters.push_back(dau);
            }
            if (allow_gluon && pdgid == ParticleID::p_g) {
                daughters.push_back(dau);
            }
        }
        return daughters;
    }

    std::pair<std::vector<const GenParticle*>, int> getTauDaughters(const GenParticle* particle) {
        const auto tau = getFinal(particle);
        auto daughters = getDaughters(tau);
        for (const auto & dau: daughters){
            auto pdgid = std::abs(dau->PID);
            if (pdgid == ParticleID::p_eminus)  return std::make_pair(daughters, 0);
            if (pdgid == ParticleID::p_muminus)  return std::make_pair(daughters, 1);
        }
        return std::make_pair(daughters, 2); // hadronic mode
    }


private:
    bool debug_ = false;
    std::vector<const GenParticle *> genParticles_;
    std::unordered_set<const GenParticle *> processed_;

    genPartData data_;
};

#endif