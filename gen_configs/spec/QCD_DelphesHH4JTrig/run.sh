#!/bin/bash

# a wrapper to generate hepmc file

# this is a customized version for qcd event generator, using pythia8 only

NEVENT=$1
MACHINE=$2
if [ -z $MACHINE ]; then
    MACHINE=farm
fi

# basic configuration
if [[ $MACHINE == "ihepel9" ]]; then
    MG5_PATH=/publicfs/cms/user/licq/utils/MG5_aMC_v2_9_18
    PYTHIA_PATH=$MG5_PATH/HEPTools/pythia8
    HEPMC_PATH=$MG5_PATH/HEPTools/hepmc
    ZLIB_PATH=$MG5_PATH/HEPTools/zlib
    FASTJET_PATH=/scratchfs/cms/tyyang99/fastjet-install
    LOAD_ENV_PATH=/publicfs/cms/user/licq/pheno/anomdet/gen/condor/load_custom_el9_env.sh
elif [[ $MACHINE == "lxplusel9" ]]; then
    MG5_PATH=/afs/cern.ch/user/c/coli/work/utils/pheno_utils/MG5_aMC_v2_9_18
    PYTHIA_PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/pythia8/309-36906/x86_64-el9-gcc13-opt
    HEPMC_PATH=/cvmfs/sft.cern.ch/lcg/releases/HepMC/2.06.11-d5a39/x86_64-el9-gcc13-opt
    ZLIB_PATH=/cvmfs/sft.cern.ch/lcg/releases/zlib/1.2.11-8af4c/x86_64-centos7-gcc13-opt
    FASTJET_PATH=/cvmfs/sft.cern.ch/lcg/releases/fastjet/3.4.1-5af57/x86_64-el9-gcc13-opt
    LOAD_ENV_PATH=/afs/cern.ch/user/c/coli/work/gen/load_lcg_el9_env.sh
elif [[ $MACHINE == "remote" ]]; then
    MG5_PATH=../MG5_aMC_v2_9_18
    PYTHIA_PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/pythia8/309-36906/x86_64-el9-gcc13-opt
    HEPMC_PATH=/cvmfs/sft.cern.ch/lcg/releases/HepMC/2.06.11-d5a39/x86_64-el9-gcc13-opt
    ZLIB_PATH=/cvmfs/sft.cern.ch/lcg/releases/zlib/1.2.11-8af4c/x86_64-centos7-gcc13-opt
    FASTJET_PATH=/cvmfs/sft.cern.ch/lcg/releases/fastjet/3.4.1-5af57/x86_64-el9-gcc13-opt
    LOAD_ENV_PATH=../load_lcg_el9_env.sh
fi

# the MG process dir
# MDIR=proc

## load environment
source $LOAD_ENV_PATH > /dev/null 2>&1

# cd into current genpack's dir
cd "$( dirname "${BASH_SOURCE[0]}" )"

# read the custom py8 parameters
# sample one line from the paramter file if py8_params.dat exists
if [ -f py8_params.dat ]; then
    RANDOM_PARAMS=$(shuf -n 1 py8_params.dat)
    IFS=' ' read -ra PARAMS <<< $RANDOM_PARAMS
fi

# step1: compile py8 program
if [ ! -f py8_main ]; then
    echo "Comple py8 program"
    g++ py8_main.cc -o py8_main -w -I$PYTHIA_PATH/include -ldl -fPIC -lstdc++ -std=c++11 -O2 -DHEPMC2HACK -DGZIP -I$ZLIB_PATH/include -L$ZLIB_PATH/lib -Wl,-rpath,$ZLIB_PATH/lib -lz -L$PYTHIA_PATH/lib -Wl,-rpath,$PYTHIA_PATH/lib -lpythia8 -ldl -I$FASTJET_PATH/include \
         -I$HEPMC_PATH/include -L$FASTJET_PATH/lib -Wl,-rpath,$FASTJET_PATH/lib -lfastjet -L$HEPMC_PATH/lib -Wl,-rpath,$HEPMC_PATH/lib -lHepMC
fi

# step2: generate event

## write py8.dat
cp -f py8_templ.dat py8.dat
## if py8_params.dat exists, replace $1, $2, ... by the customized params
if [ -f py8_params.dat ]; then
    for i in `seq 1 ${#PARAMS[@]}`; do
        sed -i "s/\$${i}/${PARAMS[$((i-1))]}/g" py8.dat
    done
fi

sed -i "s/\$NEVENT/$NEVENT/g" py8.dat # initialize nevent
sed -i "s/\$SEED/$RANDOM/g" py8.dat # initialize seed, this is not working for pythia as we use pythia's own seed

## generate hepmc events (events.hepmc)
./py8_main
