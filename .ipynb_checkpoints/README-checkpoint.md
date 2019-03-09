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

I found out that the data is unbalanced. 
0.0678 Label 1 (678) vs 0.9322 Label 0(9322)

Class 0 stats: 
![**Figure 2. **](graphics/class0_stats.png)
Class 0 Item descritpion fields have a normal like distribution The diagnosis however has a few outliers with a character length exceeding 200 Some with over more than 6 sentences and words above 50 count.

Class 1 stats: 
![**Figure 3. **](graphics/class1_stats.png)
Class 1 Char count: Diagnosis has an outlier with more than 800 characters in length Must be the one having over 10 sentences and 170 words Let's see which instances have a diagnosis word count of more than 30 On the ItemDescription side the fields seem to have in average 6 words and no more than 50 characters

If we count every stem of non-stop words or stems of lemmas of non-stop words, the feature set would be at least (4129+1628) 5757 long. 

### Text Analysis and preprocessing

Remove the name of the pet. 
Remove duplicates. 

## Resampling the Dataset

There are some approaches to counteract the effect of classifying an imbalanced set


In addition, the number of features (words) is larger than the number of samples. So it is likely there's a high risk of overfitting. 

## Model Selection
---
![**Figure 4. **](graphics/StemmingAUCdistributions.png)

I did a series of tests with the set of classifiers (with default parameters) over different number of features, ngrams and the options of stemming, lemmatizing or both. 

The figure above shows the distribution of the performance score AUC of each of the classifiers. I divided the results by n-grams and by their dimensionality (low, mid and high). The 3rd column (showing the distributions with the number of features above 3250) doesn't show the distribution with 1-grams because the maximum number of 1-grams extracted from the training set was below the limit of 3250. However the number of combinations with 2-grams and 3-grams exceeded that limit. 

The results are a bit ambiguous, since there is a tight decision on whether to use stemming or not. 
The above figure shows the results where the training set was stemmed in the preprocessing step. 

The random forest along with the SVM with gradient descent perform best at low dimensions and 1-grams. The decision trees classifier outperforms them but (Im guessing that since it has default parameters) its performance decreases as the dimensionality increases. 

The gaussian, on the other hand, stabilizes around .858 and 2-grams as the number of dimensions increase. This clf also shows a competitive AUC score above 80%. 
As for the Adaboost classifier, shows a high score in the low dimensions with the more n-grams the better. 

So there might be a tradeoff between dims and ngrams. 
The RF and SGD showing low performance (below 80%) must leave the race. 
The GNB performs better with higher dims
AB performs better with higher n-grams
DT performs better with low dims and low ngrams. 
Meeting in the middle gives n-grams=2, dims=mid. 
Lets tune the GNB, DT and Adaboost, see if they perform better. 

Stemming or Lemmatizing?

Soft Voting with the default classifiers attains a score of 87.1% with stemming and 1-grams and low dim (fig = nGrams_distribution(df, 'Val ', 2, 'AUC', 1))
with lemmatizing it has 0.867

---


---

I've selected 3 Models based on their performance with their stratified datasets:
1. Logistic regression handles the dataset with 10% of the class 0 samples (and 100% of the class 1 samples)

3. SVM with SGD deals with the 76% of the class 0 samples 

![**Figure 3. **](graphics/SetWithDT_stats.png)

## Results: 

![**Figure 4. **](graphics/comparissonMetricsRFvsDT.png)
DT shows the highest c1 recall (0.884) but c1 precision is the lowest (34%) and RF's c1 precision is 51% (signs of **overfitting**), however 
Voting soft with DT yields a higher C1 recall (0.875), voting hard favores DT too. 
Also Voting Soft shows the maximum value for AUC with DT. 
An important note is that SGD trains really badly the c1 samples (only 30%) but regularizes well and jumps to 58% with DT and with RF at 66%

![**Figure 5. **](graphics/comparissonRFvsDT.png)
RF beats dt in terms of c1's precision and the recall difference is not that much. But when it comes to voting soft DT beats RF in AUC (0.916) and c1 precision but RF wins in the same extent in c1 recall
In voting hard DT beats RF in AUC (0.871) as well as c1's recall and precision. 
In conclusion DT shows better results. 



## Future Work
1. Readjust the comparisson within the customized vocabulary by stemming and lemmatizing it. 
2. Seems the performance of the model with a dataset containing the original duplicates has a higher AUC and Recall of the minority class. 
3. Refine the choice of top classifiers
