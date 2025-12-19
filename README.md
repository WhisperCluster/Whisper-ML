# Whisper-ML

**Whisper-ML** is a collection of Machine Learning and Deep Learning algorithms developed for the analysis of **WHISPER** instrument (Décréau et al.) data from the **CLUSTER** mission (Escoubet el al.)
- **Provided by:**  
**LPC2E** – Laboratoire de Physique et Chimie de l’Environnement et de l’Espace
(LPC2E, OSUC, Univ Orleans, CNRS, CNES, F-45071 Orleans, France)
- **Authors**: Emmanuel De Leon, Maxime Vandevoorde, Nicolas Gilet, Xavier Vallières

*Décréau et al.  https://doi.org/10.1023/A:1004931326404
*Escoubet et al. vers https://doi.org/10.1023/A:1004923124586

## Table of contents
- [General Information](#general-information)
- [Data Availability](#data-availability)
- [Technologies](#technologies)
- [Setup](#setup)
- [Running the Models](#running-the-models)

## General Information

This repository contains the source code associated with the scientific publication:

> **Vallière et al., "XXX"**, *Journal: XXX*

The code implements Machine Learning models designed to process and analyze data from the **WHISPER** instrument aboard the **CLUSTER** spacecraft.


## Data Availability

All Cluster data, including the WHISPER electric field spectra and the thermal electron density dataset is public and available at the ESA Cluster Science Archive (CSA) (https://csa.esac.esa.int)
DOI for WHISPER key science datasets at CSA is https://doi.org/10.5270/esa-6stdo07.

A **demo dataset** is provided for testing and demonstration purposes.
[> Link ZENODO](https://zenodo.org/records/17977835)


## Technologies
This project is developed using the following technologies:

- **Python** – Main programming language  
- **CEFLIB** – Python library for handling CLUSTER spacecraft data  
  https://ceflib.irap.omp.eu/
- **Keras** – High-level Deep Learning API  
  https://keras.io/


## Setup
To install this project:
  1. **Install Python** 3.10 (This version has been tested with the current code.)
  2. **Install CEFLIB (Python version)**  version https://ceflib.irap.omp.eu/page=install 
      > ⚠️ CEFLIB is intended to run on **Linux distributions**.  
      > Windows builds are possible but require additional configuration.
  
  3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

## Running the Models
 1. Download the Whisper demo dataset and uncompress Whisper-ML_demo_dataset.tar.gz
  https://zenodo.org/records/17977835
    - Includes:
        * 2 days of data for training (natural, active, electron density)
        * 1 day of data for testing

 3. Verify the CEFLIB installation path
    - Open the model script you want to run
    - Check that the CEFLIB path is correctly set

 4. Update dataset paths
    - Edit the script to point to the local demo dataset files

 5. Run one of the example models
- Training and testing of a region selection model with natural data
```bash
python natural_model.py
```
- Training and testing of a region selection model with natural and acrtive data
```bash
python sw_ms_model.py
```
- Training and testing a model to find the plasma frequency from acrtive data
```bash
python fp_model.py
```

The result obtained (plot image) by running the demo scripts is available in the zenodo archive in the folder Whisper-ML_results.tar.gz
