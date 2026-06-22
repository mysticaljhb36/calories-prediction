@echo off

echo =====================================
echo Creating Conda Environment
echo =====================================

conda env create -f environment.yml

echo =====================================
echo Activating Environment
echo =====================================

call conda activate calories-prediction

echo =====================================
echo Python Version
echo =====================================

python --version

echo =====================================
echo Environment Ready
echo =====================================

pause