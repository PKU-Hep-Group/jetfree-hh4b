/*
 *  Delphes: a framework for fast simulation of a generic collider experiment
 *  Copyright (C) 2012-2014  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <signal.h>

#include "TApplication.h"
#include "TROOT.h"

#include "TDatabasePDG.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TObjArray.h"
#include "TParticlePDG.h"
#include "TStopwatch.h"

#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "classes/DelphesHepMC2Reader.h"
#include "modules/Delphes.h"

#include "ExRootAnalysis/ExRootProgressBar.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

using namespace std;

//---------------------------------------------------------------------------

static bool interrupted = false;

void SignalHandler(int sig)
{
  interrupted = true;
}

//---------------------------------------------------------------------------

bool passParticleFilter(TObjArray *stableParticleOutputArray)
// This filter acts before Delphes processing
{
  bool pass = false;
  double genHT = 0;
  for(int i = 0; i < stableParticleOutputArray->GetEntries(); ++i)
  {
    Candidate *candidate = static_cast<Candidate *>(stableParticleOutputArray->At(i));
    if(candidate->Status == 1) { // note: stable particles are all status 1
      genHT += candidate->Momentum.Pt();
    }
  }
  if (genHT > 300) {
    pass = true;
  }
  // std::cout << "//" << pass << "// genHT = " << genHT << std::endl;

  return pass;
}

bool passDelphesObjectFilter(Delphes *modularDelphes) {
    TObjArray *jets = modularDelphes->ImportArray("JetEnergyScalePUPPI/jets");
    
    double jHT = 0;
    std::vector<double> jetPts;
    
    for (int i = 0; i < jets->GetEntries(); ++i) {
        Candidate *candidate = static_cast<Candidate *>(jets->At(i));
        
        if (candidate->Momentum.Pt() > 30 && std::abs(candidate->Momentum.Eta()) < 2.5) {
            jHT += candidate->Momentum.Pt();
            jetPts.push_back(candidate->Momentum.Pt());
        }
    }
    
    std::sort(jetPts.begin(), jetPts.end(), std::greater<double>());
    
    if (jetPts.size() < 4) {
        return false;
    }
    
    return (jHT > 330 && 
            jetPts[0] > 75 && 
            jetPts[1] > 60 && 
            jetPts[2] > 45 && 
            jetPts[3] > 40);
}

int main(int argc, char *argv[])
{
  char appName[] = "DelphesHepMC2";
  stringstream message;
  FILE *inputFile = 0;
  TFile *outputFile = 0;
  TStopwatch readStopWatch, procStopWatch;
  ExRootTreeWriter *treeWriter = 0;
  ExRootTreeBranch *branchEvent = 0, *branchWeight = 0;
  ExRootConfReader *confReader = 0;
  Delphes *modularDelphes = 0;
  DelphesFactory *factory = 0;
  TObjArray *stableParticleOutputArray = 0, *allParticleOutputArray = 0, *partonOutputArray = 0;
  DelphesHepMC2Reader *reader = 0;
  Int_t i, maxEvents, skipEvents;
  Long64_t length, eventCounter, genSelectedCounter, fillCounter;

  if(argc < 3)
  {
    cout << " Usage: " << appName << " config_file"
         << " output_file"
         << " [input_file(s)]" << endl;
    cout << " config_file - configuration file in Tcl format," << endl;
    cout << " output_file - output file in ROOT format," << endl;
    cout << " input_file(s) - input file(s) in HepMC format," << endl;
    cout << " with no input_file, or when input_file is -, read standard input." << endl;
    return 1;
  }

  signal(SIGINT, SignalHandler);

  gROOT->SetBatch();

  int appargc = 1;
  char *appargv[] = {appName};
  TApplication app(appName, &appargc, appargv);

  try
  {
    outputFile = TFile::Open(argv[2], "CREATE");

    if(outputFile == NULL)
    {
      message << "can't create output file " << argv[2];
      throw runtime_error(message.str());
    }

    treeWriter = new ExRootTreeWriter(outputFile, "Delphes");

    branchEvent = treeWriter->NewBranch("Event", HepMCEvent::Class());
    branchWeight = treeWriter->NewBranch("Weight", Weight::Class());

    confReader = new ExRootConfReader;
    confReader->ReadFile(argv[1]);

    maxEvents = confReader->GetInt("::MaxEvents", 0);
    skipEvents = confReader->GetInt("::SkipEvents", 0);

    if(maxEvents < 0)
    {
      throw runtime_error("MaxEvents must be zero or positive");
    }

    if(skipEvents < 0)
    {
      throw runtime_error("SkipEvents must be zero or positive");
    }

    modularDelphes = new Delphes("Delphes");
    modularDelphes->SetConfReader(confReader);
    modularDelphes->SetTreeWriter(treeWriter);

    factory = modularDelphes->GetFactory();
    allParticleOutputArray = modularDelphes->ExportArray("allParticles");
    stableParticleOutputArray = modularDelphes->ExportArray("stableParticles");
    partonOutputArray = modularDelphes->ExportArray("partons");

    reader = new DelphesHepMC2Reader;

    modularDelphes->InitTask();

    i = 3;
    do
    {
      if(interrupted) break;

      if(i == argc || strncmp(argv[i], "-", 2) == 0)
      {
        cout << "** Reading standard input" << endl;
        inputFile = stdin;
        length = -1;
      }
      else
      {
        cout << "** Reading " << argv[i] << endl;
        inputFile = fopen(argv[i], "r");

        if(inputFile == NULL)
        {
          message << "can't open " << argv[i];
          throw runtime_error(message.str());
        }

        fseek(inputFile, 0L, SEEK_END);
        length = ftello(inputFile);
        fseek(inputFile, 0L, SEEK_SET);

        if(length <= 0)
        {
          fclose(inputFile);
          ++i;
          continue;
        }
      }

      reader->SetInputFile(inputFile);

      ExRootProgressBar progressBar(length);

      // Loop over all objects
      eventCounter = 0;
      genSelectedCounter = 0;
      fillCounter = 0;
      treeWriter->Clear();
      modularDelphes->Clear();
      reader->Clear();
      readStopWatch.Start();
      while((maxEvents <= 0 || eventCounter - skipEvents < maxEvents) && reader->ReadBlock(factory, allParticleOutputArray, stableParticleOutputArray, partonOutputArray) && !interrupted)
      {
        if(reader->EventReady())
        {
          ++eventCounter;

          readStopWatch.Stop();

          if(eventCounter > skipEvents && passParticleFilter(stableParticleOutputArray))
          {
            ++genSelectedCounter;
            procStopWatch.Start();
            modularDelphes->ProcessTask();
            procStopWatch.Stop();

            reader->AnalyzeEvent(branchEvent, eventCounter, &readStopWatch, &procStopWatch);
            reader->AnalyzeWeight(branchWeight);

            if (passDelphesObjectFilter(modularDelphes)) {
              treeWriter->Fill();
              ++fillCounter;
            }

            treeWriter->Clear();
          }

          modularDelphes->Clear();
          reader->Clear();

          readStopWatch.Start();
        }
        progressBar.Update(ftello(inputFile), eventCounter);
      }

      fseek(inputFile, 0L, SEEK_END);
      progressBar.Update(ftello(inputFile), eventCounter, kTRUE);
      progressBar.Finish();

      if(inputFile != stdin) fclose(inputFile);

      ++i;
    } while(i < argc);

    modularDelphes->FinishTask();
    treeWriter->Write();

    cout << "** Exiting..." << endl;
    cout << "** Processed " << eventCounter << " events" << endl;
    cout << "** GEN-selected " << genSelectedCounter << " events" << endl;
    cout << "** Filled " << fillCounter << " events" << endl;

    delete reader;
    delete modularDelphes;
    delete confReader;
    delete treeWriter;
    delete outputFile;

    return 0;
  }
  catch(runtime_error &e)
  {
    if(treeWriter) delete treeWriter;
    if(outputFile) delete outputFile;
    cerr << "** ERROR: " << e.what() << endl;
    return 1;
  }
}
