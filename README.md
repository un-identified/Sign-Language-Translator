https://devfolio.co/projects/sl-playground-a801
# Sign Language Translator

## Project Overview

This project implements a sign language translator that can recognize and translate both American Sign Language (ASL) and Indian Sign Language (ISL) into text. The project utilizes deep learning models trained on datasets of sign language videos to achieve this translation. 

## Features

- **ASL Alphabet Recognition:** The project includes a model trained to recognize the alphabet of American Sign Language.
- **ASL Word Recognition:** The project includes a model trained to recognize individual words in American Sign Language.
- **ISL Alphabet Recognition:** The project includes a model trained to recognize the alphabet of Indian Sign Language.
- **ISL Word Recognition:** The project includes a model trained to recognize individual words in Indian Sign Language.

## Installation

This project uses Python and relies on a number of Python libraries.


## Usage

The project is organized into two main directories: `ASL` and `ISL`.  Each directory contains separate models and training scripts for the corresponding sign language.


**To test the trained models:**

1. Navigate to the appropriate directory (`ASL` or `ISL`).
2. Run `python test_isl_words.py` or `python test_isl_alpha.py` for ISL.
3. Run `python test_asl_words.py` or `python test_asl_alpha.py` for ASL.
