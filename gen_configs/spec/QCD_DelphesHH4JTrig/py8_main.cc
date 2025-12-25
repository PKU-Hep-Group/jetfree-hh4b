// main03.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: basic usage; process selection; command file; python; matplotlib;

// This is a simple test program.
// It illustrates how different processes can be selected and studied.
// All input is specified in the main03.cmnd file.
// Also illustrated output to be plotted by Python/Matplotlib/pyplot.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

using namespace Pythia8;

// This is the minimal interface needed to access FastJet.
// A more sophisticated interface is demonstrated in main72.cc.
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

//==========================================================================

int main() {

  Pythia8ToHepMC toHepMC("events.hepmc");

  // Generator.
  Pythia pythia;

  // Shorthand for the event record in pythia.
  Event& event = pythia.event;

  // Read in commands from external file.
  pythia.readFile("py8.dat");

  // Extract settings to be used in the main program.
  int nEvent = pythia.mode("Main:numberOfEvents");
  int nAbort = pythia.mode("Main:timesAllowErrors");

  // Initialize.
  pythia.init();

  // Fastjet analysis - select algorithm and parameters
  double Rparam = 0.4;
  fastjet::Strategy               strategy = fastjet::Best;
  fastjet::RecombinationScheme    recombScheme = fastjet::E_scheme;
  fastjet::JetDefinition         *jetDef = NULL;
  jetDef = new fastjet::JetDefinition(fastjet::antikt_algorithm, Rparam,
                                      recombScheme, strategy);
  // Fastjet input
  std::vector <fastjet::PseudoJet> fjInputs;

  // Begin event loop.
  int iAbort = 0;
  int nEventWrite = 0;
  bool firstEvent = true;
    
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // Generate events. Quit if many failures.
    if (!pythia.next()) {
      if (++iAbort < nAbort) continue;
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // Reset Fastjet input
    fjInputs.resize(0);

    // Event filter
    bool pass = false;
    double genHT = 0;
    for (int i = 0; i < pythia.event.size(); ++i) {
      if (pythia.event[i].isFinal()) {
        genHT += pythia.event[i].pT();
        // cout << i << " " << pythia.event[i].status() << " pT = " << pythia.event[i].pT()
        //      << " GeV/c and eta = " << pythia.event[i].eta() << endl;
        // No neutrinos
        if (pythia.event[i].idAbs() == 12 || pythia.event[i].idAbs() == 14 ||
            pythia.event[i].idAbs() == 16)     continue;
        // Only |eta| < 3.6
        if (abs(pythia.event[i].eta()) > 3.6) continue;

        // Store as input to Fastjet
        fjInputs.push_back( fastjet::PseudoJet( pythia.event[i].px(),
          pythia.event[i].py(), pythia.event[i].pz(), pythia.event[i].e() ) );
      }
    }
    if (fjInputs.size() == 0) {
       cout << "Error: event with no final state particles" << endl;
       continue;
    } 

    // Run Fastjet algorithm
    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequence clustSeq(fjInputs, *jetDef);

    // For the first event, print the FastJet details
    if (firstEvent) {
      cout << "Ran " << jetDef->description() << endl;
      cout << "Strategy adopted by FastJet was "
           << clustSeq.strategy_string() << endl << endl;
      firstEvent = false;
    }

    // Extract inclusive jets sorted by pT (note minimum pT of 15.0 GeV)
    inclusiveJets = clustSeq.inclusive_jets(15.0);
    sortedJets    = sorted_by_pt(inclusiveJets);

    int HT_3jets = 0;

    int jetCount20 = 0, jetCount30 = 0;
    int jetCount35 = 0, jetCount50 = 0, jetCount70 = 0, HT = 0;
    for (unsigned int i = 0; i < sortedJets.size(); i++) {
      // Only count jets that have |eta| < 3.5
      if (abs(sortedJets[i].rap()) > 3.5) continue;
      if (sortedJets[i].perp() > 20.0) {
          jetCount20++;
          HT += sortedJets[i].perp();
          if (i <= 2)
              HT_3jets += sortedJets[i].perp();
      }
      if (sortedJets[i].perp() > 30.0)
          jetCount30++;
      if (sortedJets[i].perp() > 35.0)
          jetCount35++;
          // HT += sortedJets[i].perp();
      if (sortedJets[i].perp() > 50.0)
          jetCount50++;
      if (sortedJets[i].perp() > 70.0)
          jetCount70++;
    }

    if (genHT > 350 && HT > 240 && HT_3jets > 180 && jetCount35 >= 4 && jetCount50 >= 2 && jetCount70 >= 1) pass = true;

    // Write to HepMC
    if (pass) {
      ++nEventWrite;
      toHepMC.writeNextEvent(pythia);
    }
  // End of event loop.
  }

  // Final statistics. Normalize and output histograms.
  pythia.stat();
  cout << "Number of events written: " << nEventWrite << endl;

  // Done.
  return 0;
}
