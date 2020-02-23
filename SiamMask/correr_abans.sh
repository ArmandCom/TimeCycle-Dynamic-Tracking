#!/bin/bash
# Aquest script fa tot lo necessari perque no sigui un penyazo correr el tracker

cd /home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask
echo "Ara esta a "$PWD
export SiamMask=$PWD
echo "Adding project to PYTHONPATH"
cd /home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/experiments/siammask_sharp/
export PYTHONPATH=$PWD:$SiamMask





