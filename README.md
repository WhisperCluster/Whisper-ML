# Whisper-ML

**Whisper-ML** is a collection of Machine Learning and Deep Learning algorithms developed for the analysis of **WHISPER** instrument data from the **CLUSTER** mission
- **Provided by:**  
**LPC2E** – *Laboratoire de Physique et Chimie de l’Environnement et de l’Espace*  
(CNRS, Université d’Orléans, CNES)
- **Authors**: Emmanuel De Leon, Maxime Vandevoorde, Nicolas Gilet, Xavier Vallières

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

WHISPER data are publicly available via the **ESA Cluster Science Archive (CSA)**:
https://csa.esac.esa.int/csa-web/#search

A **demo dataset** is provided for testing and demonstration purposes.
> Link ZENODO

## Technologies
This project is developed using the following technologies:

- **Python** – Main programming language  
- **CEFLIB** – Python library for handling CLUSTER spacecraft data  
  https://ceflib.irap.omp.eu/
- **Keras** – High-level Deep Learning API  
  https://keras.io/


## Setup
To install this project:
  1. **Install CEFLIB (Python version)**  version https://ceflib.irap.omp.eu/page=install 
      > ⚠️ CEFLIB is intended to run on **Linux distributions**.  
      > Windows builds are possible but require additional configuration.
  2. **Install Python** 3.10 (This version has been tested with the current code.)
  3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

## Running the Models
 1. Download the Whisper demo dataset
    - Includes:
        * 2 days of data for training (natural, active, electron density)
        * 1 day of data for testing

 2. Verify the CEFLIB installation path
    - Open the model script you want to run
    - Check that the CEFLIB path is correctly set

 3. Update dataset paths
    - Edit the script to point to the local demo dataset files

 4. Run one of the example models
```bash
python natural_model.py
```
```bash
python fp_model.py
```
```bash
python sw_ms_model.py
```