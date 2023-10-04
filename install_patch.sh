#!/bin/bash

TDDFA_REPO="https://github.com/cleardusk/3DDFA_V2.git"
TDDFA_WDIR="TDDFA_V2"
SYNERGYNET_REPO="https://github.com/choyingw/SynergyNet.git"
SYNERGYNET_WDIR="SynergyNet"


# Check if repository directory exist, otherwise
if [ ! -d "$TDDFA_WDIR" ]; then
        git clone "$TDDFA_REPO"
        mv 3DDFA_V2 $TDDFA_WDIR
        cp patch/FAwKF_3DDFA_V2_Interface.py $TDDFA_WDIR
fi

# Check if repository directory exist, otherwise
if [ ! -d "$SYNERGYNET_WDIR" ]; then
        git clone "$SYNERGYNET_REPO"
        cp patch/FAwKF_SynergyNet_Interface.py $SYNERGYNET_WDIR
        cp patch/synergy3DMM.py $SYNERGYNET_WDIR
        cd $SYNERGYNET_WDIR
        pip install --upgrade gdown
        gdown --id 1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8 -O 3dmm_data.zip
        unzip -o 3dmm_data.zip -d .
        cd ..
fi

