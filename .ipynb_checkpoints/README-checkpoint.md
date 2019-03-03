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
Class 0 Item descritpion fields have a normal like distribution The diagnosis however has a few outliers with a character length exceeding 200 Some with over more than 6 sentences and words above 50 count.

Class 1 stats: 
![**Figure 3. **](graphics/class1_stats.png)
Class 1 Char count: Diagnosis has an outlier with more than 800 characters in length Must be the one having over 10 sentences and 170 words Let's see which instances have a diagnosis word count of more than 30 On the ItemDescription side the fields seem to have in average 6 words and no more than 50 characters

### Text Analysis

## Model Selection
---
I've selected 3 Models based on their performance with their stratified datasets:
1. Logistic regression handles the dataset with 10% of the class 0 samples (and 100% of the class 1 samples)

3. SVM with SGD deals with the 76% of the class 0 samples 

![**Figure 3. **](graphics/SetWithDT_stats.png)

## Results: 
![**Figure 4. **](graphics/comparissonRFvsDT.png)
RF beats dt in terms of c1's precision and the recall difference is not that much. But when it comes to voting soft DT beats RF in AUC (0.916) and c1 precision but RF wins in the same extent in c1 recall
In voting hard DT beats RF in AUC (0.871) as well as c1's recall and precision. 
In conclusion DT shows better results. 

