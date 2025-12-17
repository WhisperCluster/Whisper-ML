# Whisper-ML
Whisper Machine (Deep) Learning algorithms
Provided by : 
LPC2E - Laboratoire de Physique et Chimie de l'Environnement et de l'Espace (CNRS, Université d'Orléans, CNES)

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is the repo for the code associated with the scientific publication by Valliere et al.,"XXX*" on journal "XXX".
Whisper data is available at CSA, Cluster Science Archive, https://csa.esac.esa.int/csa-web/#search
* 

## Technologies
This is created with:
  PYTHON, programming language
  CEFLIB, the python version to handle CLUSTER spacecraft data:
  https://ceflib.irap.omp.eu/
  KERAS, deep learning API 
  https://keras.io/


## Setup
To install this project:
   1) Install The python  CEFLIB version https://ceflib.irap.omp.eu/ 
      This software is intended to run on linux disbritbitions (for Windows building CEFLIB is possible but needs some tweaking)
   2) Install python 3.10 (Tested with )
   3) Install Python environement requirements
      pip install -r requirements.txt
## Run a model with an example script
  Examples scripts of trainning and prediction with ML models are available: 
    natural_model.py : 
    fp_model.py :
    sw_ms_model.py :

   To run these scripts : 
   - Download the Whisper demo dataset: The dataset includes 2 days of natural,active and electron density for trainning and 1 day for testing
   - Check the path to the CEFLIB in the script to be run
   - Chek the path to the demo data files in the script you want to run and then 
   run 
  
   
