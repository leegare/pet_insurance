# pet_insurance
NLP analysis and classification of a dataset of claims


## Goals
---
The goal is to build a binary classifier to predict the “PreventiveFlag” label using the text features provided. This model can be used to automate the detection of ineligible line items. The expected output are prediction probabilities for rows 10001 through 11000, where the labels are currently null.

## Questions for Exploration
---

## Initial Data Cleaning Approach and Exploratory Findings
---
![**Figure 1. Statistics on the whole dataset**](graphics/dataset_stats.png)

## EDA
--- 
Class 0 stats: 
![**Figure 2. **](graphics/class0_stats.png)
Class 1 stats: 
![**Figure 3. **](graphics/class1_stats.png)

### Text Analysis

## Model Selection
---
I've selected 3 Models based on their performance with their stratified datasets:
1. Logistic regression handles the dataset with 10% of the class 0 samples (and 100% of the class 1 samples)

3. SVM with SGD deals with the 76% of the class 0 samples 

![**Figure 3. **](graphics/SetWithDT_stats.png)

