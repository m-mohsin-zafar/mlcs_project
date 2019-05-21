# CIS-597 | Machine Learning in Cyber Security
Malware Classification Project | Mircrosoft Malware Classification Challenge (Big 2015)


## Introduction
The subject project completed as partial requirement to successful completion of course 'Machine Learning in Cyber Security' at Pakistan Institute of Engineering and Applied Sciences (PIEAS) was started with objectives to explore domain of Applied Machine Learning (as a beginner) while also getting insight to Cyber Security Research areas pertaining to Machine Learning.
Project was started from the very basics to ensure a complete understanding of Machine Learning Pipeline. Started with Data Collection, did Feature Extraction, partitioned data for Training and Testing, tried different ML Classifiers, tuned Hyper Parameters by employing Manual Grid Search and Stratified K-Fold Cross Validation Strategy, used different Metrics to Evaluate Models and found the best out of them and finally visualized the data/results collected during the whole process.

## Environment Setup

#### Operating System
* Microsoft Windows 10 / Ubuntu 18.04 

#### Software 
* Anaconda 3 (Spyder) / JetBrains PyCharm Professional

#### Language
* Python 3.7

#### Requisite Python Libraries/Modules
* All requisite modules are mentioned in [requirements.txt](requirements.txt)

## Data Collection
* Data for Malware Files is available at [Kaggle](https://www.kaggle.com/c/malware-classification/data)
* Data can be downloaded by using following command (Pre Requisite is Kaggle API Installation)
```
kaggle competitions download -c malware-classification
```
* It is important to note that data downloaded from Kaggle will contain 'train.7z' and 'test.7z' folders. Out of these 'train.7z' should be considered and then unzipped. After unzipping all '.asm' files should be separated and placed in folder at following position;
```
--current_working_directoy
  |-----feature_ext
        |-----train_asm
```
* After placing '.asm' files here, these are to be archived in '.gzip' format.

## Project Structure
* Overall Structure is as under;
```
|----root
     |----data
          |----dumps
          |----original
          |----per_class_distribution
          |----plots
          |----processed
          |----saved_models
     |----feature_ext
          |----dataset
               |----train_asm
          |----io_content
               |----feature_csvs
               |----*** (multiple files)
          |----*** (multiple python files for performing feature extraction)
     |----*** (multiple python files for performing Classification, Metric Evaluations & Plotting)
```



