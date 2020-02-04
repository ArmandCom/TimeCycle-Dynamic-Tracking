#!/bin/bash
# Aquest script fa tot lo necessari perque no sigui un penyazo correr el tracker
cd /home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask
echo "Ara esta a "$PWD
export SiamMask=$PWD
source /home/ppalau/sisiam/bin/activate
bash make.sh
echo "Adding project to PYTHONPATH"
export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH



