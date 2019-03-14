from random import shuffle
from time import time
from nltk import pos_tag

import re
import nltk
import statistics # involved in hard voting
import numpy as np # used in resample_datasets
import pandas as pd # Used in voting

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns


from nltk import word_tokenize, sent_tokenize # used in eda_stemming_lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist # used in eda_stemming_lemmatizing
from nltk.corpus import stopwords # Used in feature selection analysis

from sklearn import preprocessing # plot_dimensionality_series
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, \
                            mean_squared_error, accuracy_score, f1_score, \
                            precision_score, recall_score
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
# nltk.download('wordnet')

# ====================================================
#                Director Functions
# ====================================================

def trupanion(data, clfs, flags, mod):

    # ---------------------------
    #  Script
    # ---------------------------
    t0 = time()

    ## --- Preprocessing
    Xtrain, Xval, y_train, y_val, feature_names, vCounter, vTfidf = m_preprocess(data, flags, 0)

    ## --- Feature selection
    X_train, X_val, fn = get_best_features(flags['optimalK'], Xtrain, y_train, Xval, feature_names)

    ## --- Training and Prediction
    if mod == 0:
        ytr_pred,yval_pred,duration = m_training_prediction_biz(X_train, y_train, X_val, clfs, flags)
        df = get_results_biz(clfs,y_train,ytr_pred,X_train,y_val,yval_pred,X_val,0,0,flags,duration)

    if mod == 1:
    # And/Or Voting

        yval_soft, duration = voting_biz(clfs, X_train, y_train, X_val, [], 'soft')
        yval_hard, duration = voting_biz(clfs, X_train, y_train, X_val, duration, 'hard')
        df = get_results_biz(0,y_train,0,X_train,y_val,0,X_val,yval_soft,yval_hard,flags, duration)

    if mod == 2:
        ytr_pred,yval_pred,duration = m_training_prediction_biz(X_train, y_train, X_val, clfs, flags)
        yval_soft, duration = voting_biz(clfs, X_train, y_train, X_val, duration, 'soft')
        yval_hard, duration = voting_biz(clfs, X_train, y_train, X_val, duration, 'hard')
        ## --- Results
        df = get_results_biz(clfs,y_train,ytr_pred,X_train,y_val,yval_pred,X_val,yval_soft,yval_hard,flags,duration)

    # if mod == 3: # Just train:

    # df.to_csv(path+'/results/OptKBaselineVotingNonVoc.csv')
    print('Time spent: ', round(time()-t0,1))
    # df.sort_values(['AUC','F1_C1', 'Prec_C1', 'Recal_C1'], ascending=False)
    return df


def m_preprocess(data, f, test):

    # Remove contradictions (specific case)
    data = data.loc[~data.index.isin([7850, 7861])]

    # Remove outliers
    c0 = data.loc[data.PreventiveFlag==0.0]
    c0d_lengths = [len(item[1].Diagnosis) for item in c0.iterrows()]
    outliers = c0.iloc[np.where(np.array(c0d_lengths) > 200)].index
    data = data.loc[~(data.index.isin(outliers))]

    print('Preprocessing...')
    data_features, data = preprocess(data, f) ## --- Preprocessing: ~ 17s

    if not isinstance(test, pd.core.frame.DataFrame):

        # Splitting into validation and training sets:
        print('Splitting datasets into training and validation sets...')
        train, validation, y_train, y_val = train_test_split(data_features,
                                                             data.PreventiveFlag.values,
                                                             test_size=f['trainValRatio'],
                                                           random_state=f['rndseed'])
    else: # Name the test set validation
        f['keepDupes'] = True
        validation, test = preprocess(test, f)
        train = data_features
        y_train = data.PreventiveFlag.values
        y_val = [0]


    # Vectorizing with CountVectorizer and TFIDFVectorizer:
    # remove english stop words
    print('Vectorizing...')
    X_train, vc, vt, fn = vectorize(True, train, False, False, f['n-grams'], f['customVocabulary'])

    X_val, vc, vt = vectorize(False, validation, vc, vt, f['n-grams'], f['customVocabulary'])

    try:
        print('Data shape, Training and validation sizes:{}, {}, {},{},{}'.\
              format(data.shape, X_train.shape,len(y_train), X_val.shape, len(y_val)))
    except:
        print(type(data), type(X_train), type(y_train), type(X_val), type(y_val))
    return X_train, X_val, y_train, y_val, fn, vc, vt


def m_training_prediction_biz(Xtr, ytr, X_val, clfs, f):

#     y_val_proba = {}
    y_val_pred = {}
    y_tr_pred = {}

    n_models = f['N_Models']
    duration = []

    for i in range(1,n_models+1):
        t0 =time()
        # Fit
        clfs['clf_' + str(i)].fit(Xtr, ytr)
        # Predict probabilities
#         y_val_proba['clf_' + str(i)] = clfs['clf_' + str(i)].predict_proba(X_val)
        # Predict classes
        y_val_pred['clf_' + str(i)] = clfs['clf_' + str(i)].predict(X_val)
        # These predictions can then be used to evaluate the classifier:
        y_tr_pred['clf_' + str(i)] = cross_val_predict(clfs['clf_' + str(i)],
                                                       Xtr,
                                                       ytr,
                                                       cv=5)
        # print(round(time()-t0,1))
        duration.append(round(time()-t0,1))

    return y_tr_pred, y_val_pred, list(np.repeat(duration, 2))

####################################
#### Feature engineering functions:
####################################

def feature_engineering_method_1(data, path):
    '''
    Function that receives the binary class dataset:
        - Preprocess the text columns
        - Saves and Returns a dataframe with a set of words that are present in all of the instances.
    '''

    start= time()
    # Which words are the most present in all/most of the instances for class 1
    f_voc = analyze_feats(preprocess(data[data.PreventiveFlag==1.0], flag=5)) # ~ 103 s
    # Which words are the most present in all/most of the instances for class 0
    f_voc0 = analyze_feats(preprocess(data[data.PreventiveFlag==0.0], flag=5)) # ~ 108 Min
    f_vocab = f_voc0[1:]+[f_voc0[0][0]]
    vocabulary = list(set(f_vocab).union(set(f_voc)))
    v = pd.DataFrame({'Vocabulary':vocabulary})
    v.to_csv(path+'/data/vocabulary.csv')
    print(time()-start)
    return v

def getKey(item):
    return item[1]

def analyze_feats(corpus):
    '''Function:
        - receives an np array with sentences
        - Removes stop words and returns those words that as a set have their presence in
          every instance of the corpus.
    '''
    stop_words = stopwords.words('english')
    voc = []
    # Remove stop words
    for i in range(len(corpus)):
        voc += word_tokenize(' '.join(list(filter(lambda c: c not in stop_words, corpus[i].split()))))
    # Vocab has the unique words as keys and their repeats as values.
    vocab = [(k,v) for k,v in FreqDist(voc).items()]

    print("Total number of non-stop words: {}".format(len(voc)))
    print("There are {} unique non_stop words, numbers ".format(len(vocab)))

    # Build the count matrix with instances as rows and vocabulary words as columns.
    com = pd.DataFrame()
    for v in sorted(vocab, key=getKey, reverse=True):
    #     print('Freq word:',v[0])
        iv = []
        for row in range(len(corpus)):
            cnt = 0
    #         print(desc_corpus[row])
            if v[0] in word_tokenize(corpus[row]):
                cnt += 1
            iv.append(cnt)
        com[v[0]] = iv

    com['Presence_'] = com.sum(axis=1)

    ## --- Feature Selection.
    vocabulary = []
    # Need the words that appear in those instances that have only 1 correspondance with the vocabulary
    if len(com.loc[com.Presence_==1].idxmax(axis=1).values) > 0:
        # Consider the case where there might be none...
        vocabulary.append(com.loc[com.Presence_==1].idxmax(axis=1).values)
        # Drop those rows
        com = com[com.Presence_ != 1]

    # Start with the most present word to the least present
    for i in range(len(com.columns)):
        vocabulary.append(com.columns[i])
        com = com[com[com.columns[i]] != 1]
    #     print(com.shape)
        if com.shape[0] == 0:
            print('New feature set length:',len(vocabulary))
            break
    return vocabulary

def get_best_features(k, X_train, y_train, X_val, feature_names):
    if k >= X_train.shape[1]:
        k = X_train.shape[1]-2

    print("Extracting %d best features by a chi-squared test" %k)
    t0 = time()
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    X_val = ch2.transform(X_val)

    # keep selected feature names
    # X.columns[selector.get_support(indices=True)]
    feature_names = [feature_names[i] for i
                     in ch2.get_support(indices=True)]
    print('New reshaped sets: ',X_train.shape, X_val.shape)

    return X_train, X_val, feature_names

##############################
# FUNCTIONS FOR EDA
##############################
def eda_remove_mess(corpus_all):
    analyzer = CountVectorizer(stop_words='english', token_pattern='[a-zA-Z0-9]{2,}').build_analyzer()  # Added
    vectorizer_count = CountVectorizer(ngram_range=(1,1), analyzer=analyzer)
    vectorizer_count.fit_transform(corpus_all)
    print("There are {} different non-stop words and numbers in the Item Description".format(len(vectorizer_count.get_feature_names())))
    return vectorizer_count.get_feature_names()

def eda_stemming_lemmatizing(iDesc, Diag):
    '''Function that gets 2 series corresponding to the 2 columns of text
        Each word is converted to lower case and stripped from symbols of punctuation
        Prints the number of words, lemmas and stems
    '''

    # General word count:
    voc_i = []
    voc_d = []
    for i in range(0,len(iDesc.values)):
    #     print(iDesc.values[i])
        voc_i += word_tokenize(iDesc.values[i])
        voc_d += word_tokenize(Diag.values[i])

    print("There are {} different words, numbers and symbols in the Item Description".format(len(FreqDist(voc_i).keys())))
    print("There are {} different words, numbers and symbols in the Diagnosis".format(len(FreqDist(voc_d).keys())))

    # Remove stop words and symbols
    idesc = eda_remove_mess(iDesc.values)
    diagn = eda_remove_mess(Diag.values)

    # Lemmatize
    lidesc = [get_lemma(w) for w in idesc]
    ldiagn = [get_lemma(w) for w in diagn]
    print("There are {} different lemmas, numbers and symbols in the Item Description".format(len(FreqDist(lidesc).keys())))
    print("There are {} different lemmas, numbers and symbols in the Diagnosis".format(len(FreqDist(ldiagn).keys())))

    # Stemm
    stemmer = PorterStemmer()
    didesc = [stemmer.stem(w) for w in idesc]
    ddiagn = [stemmer.stem(w) for w in diagn]
    print("There are {} different stems & numbers in the Item Description".format(len(FreqDist(didesc).keys())))
    print("There are {} different stems & numbers in the Diagnosis".format(len(FreqDist(ddiagn).keys())))


    # Lemming -> Stemming
    dlidesc = [stemmer.stem(w) for w in lidesc]
    dldiagn = [stemmer.stem(w) for w in ldiagn]
    print("There are {} different stems of lemmas & numbers in the Item Description".format(len(FreqDist(dlidesc).keys())))
    print("There are {} different stems of lemmas & numbers in the Diagnosis".format(len(FreqDist(dldiagn).keys())))

##############################
# FUNCTIONS FOR Preprocess
##############################

def join_measurements(word):  # Version beta: Needs tailoring.
    # For example: Heidi: Hema CBC/S-Chem/T4/UA/HW K9 SA710 VMG
    # Turns into : hema cbc schem t4ua hw k9sa710vmg
    # Join measurements
    pattern = r"\d{1,4}\s\w{1,3}"
    abbrvs = re.findall(pattern, word)
    for n in range(len(abbrvs)):
#         print(abbrvs[n])
        abbrv = re.search(pattern, word)
        a1 = abbrv.span()[0]
        a2 = abbrv.span()[1]
        # Find the space
        space = re.search(r"\s", word[a1:a2])
        space_ind = space.span()[1] + a1 - 1
        # remove it
        word = word[:space_ind] + word[space_ind+1:]
    return word

def text_analysis(i_desc):
    '''Function that receives a string and:
        - Converts to lower case,
        - Removes the pet's name
        - Separate words joined by a slash: word1/word2
        - Remove punctuation except for &
        - Join words separated by a period: R.I.P
        '''
    # TO DO:
#         - Correct some words like mcg to mg (if mcg is a spelling error)
    # Remove pet name:
    try:
        ind = i_desc.index(":")
        i_desc = i_desc[ind+1:]
    except Exception as e:
#         print(repr(e), 'in pet name removal.')
        pass

    pattern = r"/" # Separate words joined by a slash: word1/word2
    idesc = re.sub(pattern, " ", i_desc)
    # Remove punctuation except for &    ### Need to work on this
    pattern = r"\." # Join words separated by a period: R.I.P
    idesc = re.sub(pattern, "", idesc)
    pattern = r"[^\w\s]"
    idesc = re.sub(pattern, ' ', idesc)
    # Join measurements
    idesc = join_measurements(idesc)
    # Lower case: necessary to lemmatizing
    # Case examples: id 4013 with 9.9-22lb/4.5-10kg
    # # text_analysis:
    #     consider removing dates for i.e.:
    # data.loc[data.index.isin([7295,9815]),['Diagnosis']]
    return idesc.strip().lower()

def get_pos(word):
    return nltk.pos_tag([word])[0][1][0].lower()

def get_lemma(word):

    lemmer = WordNetLemmatizer()
    # Example:
    # w = "lbs"
    # w = 'ruptured'
    # lemmer = WordNetLemmatizer()
    # lemmer.lemmatize(w, get_pos(w))
    # print('Word',word)

    try:
        word_lemmed = lemmer.lemmatize(word, get_pos(word))
        # print('Lemmed:',word_lemmed)
        return word_lemmed
    except Exception as e:
        # print(repr(e))
    #     # There was no valid pos tag for lemmer
        return word

def preprocess(df, f):
    '''Preprocess data:
        - Calls function text_analysis:
            - Converts the corpus in lower case
        - If flag=0 or 2, it performs lemmatization
        - If flag=1 or 2, it performs stemming
        - If flag=3 Just preprocess the item description
        - If flag=4 Just preprocess the Diagnosis
        - If flag=5 Preprocess the item description and Diagnosis
       Returns a list of the whole vocabulary stemmed and/or lemmatized or not
    '''

    df['ItemDescription'] = df.ItemDescription.apply(lambda word: text_analysis(word))
    df['Diagnosis'] = df.Diagnosis.apply(lambda word: text_analysis(word))

    # Remove duplicates and consolidate the text in 1 column
    if not f['keepDupes']:
        df = df.drop_duplicates()

    if f['lemStem'] == 0 or f['lemStem'] == 2:
        # Lemmatization: converts into single synonyms and converts plural to singular
        print('Lemmatizing...')

        f_lemmatize = lambda text: " ".join([get_lemma(word) for word in text.split()])

        df['ItemDescription'] = df.ItemDescription.apply(f_lemmatize)
        df['Diagnosis'] = df.Diagnosis.apply(f_lemmatize)

    if f['lemStem'] == 1 or f['lemStem'] == 2:
        # Stemming
        print('Stemming...')
        stemmer = PorterStemmer()
        f_stemming = lambda text: " ".join([stemmer.stem(word) for word in text.split()])

        df['ItemDescription'] = df.ItemDescription.apply(f_stemming)
        df['Diagnosis'] = df.Diagnosis.apply(f_stemming)

    if f['lemStem'] == 3:
        return df.ItemDescription.values
    elif f['lemStem'] == 4:
        return df.Diagnosis.values
    else:
        return df.ItemDescription.values + " " + df.Diagnosis.values, df

def vectorize(phase, corpus_all, vectorizer_count, vectorizer_tf, n, fVoc):
    # Transforms text into a sparse matrix of n-gram counts.
    if phase: # Training data

        if fVoc:
            # vocabulary = feature_engineering_method_1(data, path) # ~110 min
            df_vocabulary = pd.read_csv(path+'/data/vocabulary.csv', index_col=0)
            vocabulary = df_vocabulary.Vocabulary.values
            print('Using customized Vocabulary')
        else:
            print('Using default vocabulary')
            vocabulary = None

        print("Extracting features from the training data using a sparse vectorizer")
        analyzer = CountVectorizer(ngram_range=(1,n), stop_words='english', token_pattern='[a-zA-Z]{2,}').build_analyzer()  # Added
        vectorizer_count = CountVectorizer(analyzer=analyzer, vocabulary=vocabulary)
        one_hot = vectorizer_count.fit_transform(corpus_all)

        vectorizer_tf = TfidfTransformer()
        freq = vectorizer_tf.fit_transform(one_hot)
        feature_names = vectorizer_count.get_feature_names()
        return freq.toarray(), vectorizer_count, vectorizer_tf, feature_names

    else: # Validation data or test data

        print("Extracting features from the test data using the same vectorizer")
        # vectorizer_count._validate_vocabulary()
        one_hot = vectorizer_count.transform(corpus_all)
        freq = vectorizer_tf.transform(one_hot)
        return freq.toarray(),vectorizer_count, vectorizer_tf


##############################
# FUNCTIONS FOR non segmented datasets
##############################

def voting_biz(c, Xt, yt, Xv, dur, mode):

    vc = VotingClassifier(
        estimators = [(k,v) for k,v in c.items()],
        voting=mode)

    t0 = time()
    vc.fit(Xt,yt)
    yval = vc.predict(Xv)
    t = round(time()-t0,1)
    dur += [t]

    return yval, dur



def get_results_biz(clfs, ytr, ytr_pred,Xtr,y_val,yval_pred,X_val,yval_soft,yval_hard,f, duration):

    # ytr = y_train
    # ytr_pred = ytr_pred
    # Xtr = X_train
    # f = flags
    # yval_soft = 0
    # yval_hard = 0

    res = pd.DataFrame()
    if isinstance(clfs, dict):
        for i in range(1,len(clfs.keys())+1):
            # Get training metrics
            classifier = "clf_"+str(i)
            res = get_metrics_biz(res, ytr, ytr_pred[classifier], Xtr.shape,"Tr-"+str(clfs[classifier]).split('(')[0])
            # Get validation metrics
            res = get_metrics_biz(res, y_val, yval_pred[classifier], X_val.shape,'Val-'+str(clfs[classifier]).split('(')[0])

    if isinstance(yval_soft, np.ndarray):
        res = get_metrics_biz(res, y_val, yval_soft, X_val.shape, 'Val Soft')

    if isinstance(yval_hard, np.ndarray):
        res = get_metrics_biz(res, y_val, yval_hard, X_val.shape, 'Val Hard')

    res.reset_index(inplace=True)
    res.columns = ['Phase','AUC','F1_C0','F1_C1','MSE','Prec_C0','Prec_C1','Recal_C0','Recal_C1','db_size']
    res['LemmVsStem'] = f['lemStem']
    res['TrValRatio'] = f['trainValRatio']
    res['nGrams'] = f['n-grams']
    res['Balanced'] = not f['resample']
    res['Vocabulary'] = f['customVocabulary']
    res['Time'] = duration
    if f['resample']:
        res['Ratio'] = f['optimalRatio']
    else:
        res['Ratio'] = '1:1'
    return res

def get_metrics_biz(df, y, yp, dbsize, ind_name):
# Evaluation Metrics:

    d_roc = round(roc_auc_score(y, yp),3)
    d_mse = round(mean_squared_error(y, yp),3)
    d_prec = precision_score(y, yp, average=None)
    d_recl = recall_score(y, yp, average=None)
    d_f1 = f1_score(y, yp, average=None)
#     d_report = metrics.classification_report(y, yp, target_names=['class 0', 'class 1'])
    n_rows, n_cols = dbsize

    return pd.concat([df, pd.DataFrame({
                            'Prec_C0':round(d_prec[0],3),
                            'Prec_C1':round(d_prec[1],3),
                            'Recal_C0':round(d_recl[0],3),
                            'Recal_C1':round(d_recl[1],3),
                            'F1_C0':round(d_f1[0],3),
                            'F1_C1':round(d_f1[1],3),
                            'AUC':d_roc,
                            'MSE':d_mse,
                            'db_size':str(dbsize)}, index=[ind_name])])

####################################
###  Plotting functions
####################################

def plot_baseline(f, axes, pl_df, mode):

    if pl_df.shape[0] == 0:
        print('No data available for plotting..')
        return 0

    barWidth = 0.12
    bar = {}
    cols = pl_df.iloc[:,1:9].columns.values
    for i, row in enumerate(pl_df.iterrows()):
        bar[row[1].Phase]=row[1][cols].values

    r = [list(np.arange(len(cols)))]
    for i in range(len(bar.keys())-1):
        rx = [x + barWidth for x in r[i]]
        r.append(rx)

    for i, clf in enumerate(bar.keys()):
        axes.barh(r[i], bar[clf], height=barWidth,  label=clf[3:])

    axes.set_yticks([y + barWidth for y in range(len(cols))])
    axes.set_yticklabels(cols)

    axes.legend(loc = 'lower left', fontsize=18, bbox_to_anchor=(1, 0))

    axes.grid()
    axes.set_title(mode, fontsize=20)
    plt.show()

def compare_performance_vs_dimensionality(x0, xf, xi, X_train, y_train, X_val, clfs, path, lab):
    chronos = []
    performance = []
    ks = []
    for kx in np.linspace(x0, xf, xi):
        k= int(kx) #2610 #5000 # 2610 #1115 #
        t0 = time()
        ch2 = SelectKBest(chi2, k=k)
        Xtr = ch2.fit_transform(X_train, y_train)
        Xval = ch2.transform(X_val)
        # keep selected feature names
        ftr_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
        yval_soft, duration = voting_biz(clfs, Xtr, y_train, Xval, [], 'soft')
        res = get_metrics_biz(pd.DataFrame(), y_val, yval_soft, Xval.shape, 'Val Soft')
        ks.append(k)
        performance.append(res.AUC.values[0])
        chronos.append(round(time()-t0,1))
#         print(Xtr.shape, Xval.shape, round(time()-t0,1))

#     print("Performance list created!")
    best_K = pd.DataFrame({'Performance':performance}, index = [int(ki) for ki in np.linspace(x0, xf, xi)])
    best_K['MovingMean'] = best_K['Performance'].rolling(10,center=True).mean()
    best_K.to_csv(path+"/data/ChoosingK"+lab+".csv")

    f, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(20,5))
#     print('Call plot function')
    plot_performance_vs_dimensionality(best_K)
#     print('Saving plot')
    f.savefig(path+"/graphics/choosingK"+lab+".png", dpi=f.dpi)

    return best_K['Performance'].argmax(), best_K['MovingMean'].argmax()

def compare_2_classifiers(exp1, exp2, fl):
    # Compare to experiments

#     dt_config['Phase'][2] = "Tr-clfX"
#     dt_config['Phase'][3] = "Val-clfX"
#     rf_config['Phase'][2] = "Tr-clfX"
#     rf_config['Phase'][3] = "Val-clfX"


    f, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(20,34))
    barWidth = 0.35
    cols = ['MSE', 'AUC', 'F1_C0', 'F1_C1', 'Prec_C0', 'Prec_C1','Recal_C0', 'Recal_C1']
    plot_comparisson_metrics(f,axes, exp1, exp2, cols, barWidth)
    f.savefig(fl['path']+'/graphics/comparissonMetrics.png', dpi=f.dpi)

    f1, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(20,34))
    comparisson_clfs(f1,axes, exp1, exp2, cols, barWidth)
    f1.savefig(fl['path']+'/graphics/comparisson.png', dpi=f.dpi)

def comparisson_clfs(f,axes, exp1, exp2, cols, barWidth):
    axes = axes.ravel()
    phases = exp1.Phase.unique()

    for idx, ax in enumerate(axes):

        bar1 = exp1.loc[exp1.Phase==phases[idx], cols].values[0]
        bar2 = exp2.loc[exp2.Phase==phases[idx], cols].values[0]
        # Set position of bar on X axis
        r1 = np.arange(len(bar1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.barh(r1, bar1, color='royalblue', height=barWidth,  label='exp1')
        ax.barh(r2, bar2, color='coral', height=barWidth, label='exp2')

        for idx2,i in enumerate(ax.patches):

            x0 = i.get_width()-.1
            leg_loc = 'upper left'

            if idx2 >= 8:
                colr = 'royalblue'
            else:
                colr = 'gold'

            ax.text(x0, i.get_y(), \
                str(round(i.get_width(),3)), fontsize=16, color=colr)


        # Add xticks on the middle of the group bars
        ax.grid()
        if idx%2==0:
            ax.set_yticks([r + barWidth for r in range(len(bar1))])
            ax.set_yticklabels(cols)
            ax.legend(loc=leg_loc, fontsize=18)
        else:
            ax.get_yaxis().set_ticks([])
        ax.set_xticks(np.linspace(0, 1, num=6))
        # Create legend & Show graphic
        ax.set_title(phases[idx], fontsize=20)

    plt.show()

def plot_comparisson_metrics(f,axes, exp1, exp2, cols, barWidth):
    axes = axes.ravel()

    for idx, ax in enumerate(axes):

        bar1 = exp1[[cols[idx]]].values
        bar2 = exp2[[cols[idx]]].values
        # Set position of bar on X axis
        r1 = np.arange(len(bar1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.barh(r1, bar1, color='royalblue', height=barWidth,  label='Exp 1')
        ax.barh(r2, bar2, color='coral', height=barWidth, label='Exp 2')

        for idx2,i in enumerate(ax.patches):

            if idx2 >= 8:
                colr = 'royalblue'
            else:
                colr = 'gold'

            if idx == 0: # MSE
                x0 = i.get_width()+.02
                leg_loc = 'upper right'
                colr = 'black'
            else:
                x0 = i.get_width()-.1
                leg_loc = 'upper left'


            ax.text(x0, i.get_y(), \
                str(round(i.get_width(),3)), fontsize=16, color=colr)

        # Add xticks on the middle of the group bars
        ax.grid()
        if idx%2==0:
            ax.set_yticks([r + barWidth for r in range(len(bar1))])
            # ax.set_yticklabels(model_c.Phase.unique())
            ax.legend(loc=leg_loc, fontsize=18)
        else:
            ax.get_yaxis().set_ticks([])
        # Create legend & Show graphic
        ax.set_title(cols[idx], fontsize=20)
        ax.set_xticks(np.linspace(0, 1, num=6))

    plt.show()

def stats_overall(data, f, axarr):
    # Length of ItemDescription in number of characters
    ItemDescription_lengths = [len(item[1].ItemDescription) for item in data.iterrows()]
    Diagnosis_lengths = [len(item[1].Diagnosis) for item in data.iterrows()]

    # Distribution of number of sentences in Item Descriptions:
    sentence_count_per_item = [len(sent_tokenize(item[1].ItemDescription)) for item in data.iterrows()]
    sentence_count_per_diagnosis = [len(sent_tokenize(item[1].Diagnosis)) for item in data.iterrows()]

    # Word count in ItemDescription
    ItemDescription_word_count = [len(word_tokenize(item[1].ItemDescription)) for item in data.iterrows()]
    Diagnosis_word_count = [len(word_tokenize(item[1].Diagnosis)) for item in data.iterrows()]

    # create 3 subplots (horizontally stacked)    ~ 893 s
    start = time()

    sns.distplot(ItemDescription_lengths, bins=10, rug=True, ax=axarr[0, 0], label="length")
    axarr[0, 0].set_title('Char count')
    sns.distplot(sentence_count_per_item, bins=10, rug=True, ax=axarr[0, 1], label="sentence count")
    axarr[0, 1].set_title('Sentence count')
    sns.distplot(ItemDescription_word_count, bins=10, ax=axarr[0,2], rug=True, label="word count")
    axarr[0,2].set_title('Word count')

    sns.distplot(Diagnosis_lengths, bins=10, rug=True, ax=axarr[1,0], label="length")
    axarr[1,0].set_title('Diagnosis Char count')
    sns.distplot(sentence_count_per_diagnosis, bins=10, rug=True, ax=axarr[1,1], label="sentence count")
    axarr[1,1].set_title('Diagnosis Sentence count')
    sns.distplot(Diagnosis_word_count, bins=10, ax=axarr[1,2], rug=True, label="word count")
    axarr[1,2].set_title('Diagnosis Word count')

    plt.show()
    # print(time()-start)


def stats_class(data, f, axarr, target):
    # Lets check the Item description and Diagnosis of instances of class 0
    c0 = data.loc[data.PreventiveFlag==target]
    # Length in characters
    c0id_lengths = [len(item[1].ItemDescription) for item in c0.iterrows()]
    c0d_lengths = [len(item[1].Diagnosis) for item in c0.iterrows()]
    # Distribution of number of sentences in Item Descriptions:
    c0id_sentence = [len(sent_tokenize(item[1].ItemDescription)) for item in c0.iterrows()]
    c0d_sentence = [len(sent_tokenize(item[1].Diagnosis)) for item in c0.iterrows()]
    # Word count in ItemDescription
    c0id_word = [len(word_tokenize(item[1].ItemDescription)) for item in c0.iterrows()]
    c0d_word = [len(word_tokenize(item[1].Diagnosis)) for item in c0.iterrows()]

    print("Average: {}".format(''.join(str(np.mean([c0id_lengths,
                     c0d_lengths,
                     c0id_sentence,
                     c0d_sentence,
                     c0id_word,
                     c0d_word], axis=1)))))
    print("Median: {}".format(''.join(str(np.median([c0id_lengths,
                         c0d_lengths,
                         c0id_sentence,
                         c0d_sentence,
                         c0id_word,
                         c0d_word], axis=1)))))
    # create 3 subplots (horizontally stacked)    ~ 581 s ~ 10 min

    sns.distplot(c0id_lengths, bins=10, rug=True, ax=axarr[0, 0], label="length")
    axarr[0, 0].set_title('Char count')
    sns.distplot(c0id_sentence, bins=10, rug=True, ax=axarr[0, 1], label="sentence count")
    axarr[0, 1].set_title('Sentence count')
    sns.distplot(c0id_word, bins=10, ax=axarr[0,2], rug=True, label="word count")
    axarr[0,2].set_title('Word count')

    sns.distplot(c0d_lengths, bins=10, rug=True, ax=axarr[1,0], label="length")
    axarr[1,0].set_title('Diagnosis Char count')
    sns.distplot(c0d_sentence, bins=10, rug=True, ax=axarr[1,1], label="sentence count")
    axarr[1,1].set_title('Diagnosis Sentence count')
    sns.distplot(c0d_word, bins=10, ax=axarr[1,2], rug=True, label="word count")
    axarr[1,2].set_title('Diagnosis Word count')

    plt.show()

def plot_performance_vs_dimensionality(best_K):
    sns.lineplot(data=best_K, palette="tab10", linewidth=2.5)
    axes.vlines(best_K['Performance'].argmax(),     # Plot black line at mean
               ymin=0.8,
               ymax=best_K['Performance'].max(),
               linewidth=5.0, color='brown', linestyle=':', label='max', alpha=0.6)
    axes.vlines(best_K['MovingMean'].argmax(),     # Plot black line at mean
               ymin=0.8,
               ymax=best_K['MovingMean'].max(),
               linewidth=5.0, color='gold', linestyle=':', label='mean max', alpha=0.6)

    axes.set_ylabel('Performance')
    axes.set_xlabel('# of features')
    axes.legend(fontsize=18)
    axes.grid()
    axes.set_title('Performance vs # of features', fontsize=20)
    plt.show()

def plot_dimensionality_series(f, axes, df1, cols):
    axes = axes.ravel()

    temp = df1.loc[df1.Phase.str.startswith('Tr')].groupby('nFeats').sum()
    # Normalize the sums
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(temp.values)
    temp = pd.DataFrame(x_scaled, columns=temp.columns, index = temp.index)
    temp['Phase'] = 'Tr-Sum'
    df1 = pd.concat([df1,temp.reset_index()])


    temp = df1.loc[df1.Phase.str.startswith('Val')].groupby('nFeats').sum()
    # Normalize the sums
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(temp.values)
    temp = pd.DataFrame(x_scaled, columns=temp.columns, index = temp.index)
    temp['Phase'] = 'Val-Sum'
    df1 = pd.concat([df1,temp.reset_index()])

    for i,ax in enumerate(axes):
        if i%2==0:
            c = [p for p in df1.Phase.unique() if p.startswith('Tr')]
            ax.set_title('Training', fontsize=22)
            ax.set_ylabel(cols[i], fontsize=20)
        else:
            c = [p for p in df1.Phase.unique() if p.startswith('Val')]
            ax.set_title('Validation', fontsize=22)

        df2 = df1.pivot(index='nFeats', columns='Phase', values=cols[i])

        if i%2==0:
            ax.fill_between(df2.index, 0, df2.loc[:,'Tr-Sum'].values, interpolate=True, color='red', alpha=0.15)
        else:
            ax.fill_between(df2.index, 0, df2.loc[:,'Val-Sum'].values, interpolate=True, color='red', alpha=0.15)

        df2.loc[:,c].plot(linewidth=4, ax=ax)
        ax.vlines(3100,     # Plot black line at mean
           ymin=0,
           ymax=1,
           linewidth=3.0, color='coral', linestyle='--', label='best k', alpha=0.8)
        ax.grid()

        if i < 2 or i > 13:
            ax.legend(loc = 'lower left', fontsize=15, framealpha=0.3, fancybox=True)
            ax.set_xlabel('# Features',fontsize=20)
        else:
            ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_title('')
    plt.show()


def nGrams_distribution(df, dataset_type, num_clfs, metric, lst):

    '''Function that divides the results of the performance of the set of classifiers
        into 3 quartiles by the number of features into low, mid and high.
        Plots the distribution of the score (variable called metric)
        It has the option to just plot the first n (num_clfs) classifiers (n>1)
        The distribution of the score is also divided in the value of n-grams.
        The mean of the metric is also ploted for every ngrams value in the legend

        The parameter lst refers to the level of lemmatizing (=0), stemming (1), or both (2)

        I made this plot to see what is the difference between using different values of ngrams.
    '''

    ## Example: Plot single classifier:
    # single_clf = df.loc[(df.nGrams==1)&\
    #                     (df.Phase=='Val-AdaBoostClassifier')&\
    #                     (df.LemmVsStem==lst)&\
    #                     (df.nFeats<=df.nFeats.quantile(.3))]
    #
    # sns.distplot(single_clf.AUC, bins=10, kde_kws={"lw": 3}, label="ng=1, µ="+ld)
    # plt.show()


    models = df.loc[df.Phase.str.startswith(dataset_type)].Phase.unique()
    models = np.repeat(models[:num_clfs],3)
    ng = range(1,4)

    lw_lim = np.quantile(df.nFeats.unique(),0.3)
    hi_lim = np.quantile(df.nFeats.unique(),0.6)

    plt.rcParams['xtick.labelsize'] = 10
    f, axes = plt.subplots(nrows = num_clfs, ncols = 3, figsize=(20,4.5*num_clfs))
    axes = axes.ravel()

    for i,ax in enumerate(axes):

        for n in ng:

            df_sec = df.loc[(df.nGrams==n)&\
                    (df.Phase==models[i])&\
                    (df.LemmVsStem==lst)]

            if i%3 == 0: # First column: Print low dim

                lowDim = df_sec.loc[df_sec.nFeats<=lw_lim]
                ld = str(round(lowDim[metric].mean(),3))
                lab = "ng=" + str(n) + " µ=" + ld #+ ' ' +str(lw_lim) + str (hi_lim)

                sns.distplot(lowDim[metric], ax=ax, hist=False, rug=True, kde_kws={"lw": 3, "shade": True}, label=lab)
                ax.set_ylabel(models[i][4:], fontsize=16)
                if i==0:
                    ax.set_title('< ' + str(lowDim['nFeats'].max()))

            if (i-1)%3 == 0: # 2nd Column

                midDim = df_sec.loc[(df_sec.nFeats>lw_lim)&(df_sec.nFeats<=hi_lim)]
                md = str(round(midDim[metric].mean(),3))
                lab = "ng=" + str(n) + " µ=" + md # + ' ' +str(lw_lim) + str (hi_lim)
                sns.distplot(midDim[metric], ax=ax, hist=False, rug=True, kde_kws={"lw": 3, "shade": True}, label=lab)
                if i==1:
                    ax.set_title(str(midDim['nFeats'].min()) + ' <= x ' + '< ' + str(midDim['nFeats'].max()))

            if (i-2)%3 == 0: # 3rd column

                highDim = df_sec.loc[df_sec.nFeats>hi_lim]
                hd = str(round(highDim[metric].mean(),3))
                lab = "ng=" + str(n) + " µ=" + hd #+ ' ' +str(lw_lim) + str (hi_lim)
                sns.distplot(highDim[metric],ax=ax, hist=False, rug=True, kde_kws={"lw": 3, "shade": True}, label=lab)
                if i==2:
                    ax.set_title(str(highDim['nFeats'].min()) + ' <=')

        # Print the xlabel on the last row
        if i >= (num_clfs-1) * 3:
            ax.set_xlabel(metric)
        else:
            ax.set_xlabel("")

        ax.grid()
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()
    return f


def plot_curves(fig, axarr, y_train, y_train_pred, y_val='0', y_val_pred='0'):
    # create 2 subplots (horizontally stacked)

    average_precision = average_precision_score(y_train, y_train_pred)
    precision, recall, f = precision_recall_curve(y_train, y_train_pred)

    axarr[0].set_title('Training Metrics', fontsize=18)
    axarr[0].step(recall, precision, color='royalblue', alpha=0.6,where='post', linewidth=4)
    axarr[0].fill_between(recall, precision, alpha=0.8, color='gold')
    axarr[0].legend(loc="top", fontsize=16)
    axarr[0].grid()
    axarr[0].set_xlabel('Recall')
    axarr[0].set_ylabel('Precision')
    axarr[0].set_ylim([0.0, 1.05])
    axarr[0].set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision), fontsize=18)

    if len(y_val) > 1:
        fpr_v, tpr_v, thresholds = roc_curve(y_val, y_val_pred)
        axarr[1].plot(fpr_v, tpr_v, linewidth=9, color='gold',label='Validation')

    fpr_t, tpr_t, thresholds = roc_curve(y_train, y_train_pred)
    axarr[1].plot(fpr_t, tpr_t, linewidth=9, linestyle=':', color='royalblue',label='Training')
    axarr[1].plot([0, 1], [0, 1], color='darkgray',linestyle='--',linewidth=7)
    axarr[1].axis([0, 1, 0, 1])
    axarr[1].set_xlabel('FPR')
    axarr[1].set_ylabel('TPR')
    axarr[1].legend(loc="lower right", fontsize=16)
    axarr[1].grid()
    axarr[1].set_title('ROC curve', fontsize=18)

    plt.show()
