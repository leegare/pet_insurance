from random import shuffle
from time import time
import re
import nltk
from nltk import pos_tag
import numpy as np # used in resample_datasets
import pandas as pd # Used in voting
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist # used in eda_stemming_lemmatizing
from nltk import word_tokenize, sent_tokenize # used in eda_stemming_lemmatizing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, \
                            mean_squared_error, accuracy_score, f1_score, \
                            precision_score, recall_score
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
# from sklearn.ensemble import VotingClassifier
# nltk.download('wordnet')
from sklearn.model_selection import train_test_split
import statistics # involved in hard voting


def trupanion(data, clfs, flags):
    start = time()

    ## ---- Preprocessing

    X_train, y_train, X_val, y_val = m_preprocess(data, flags)


    ## ---- Resampling imbalanced dataset

    # The dataset is imbalanced and to solve this I ensembled different resampled datasets. The datasets were resampled for each of the selected models using an optimal ratio.
    #     - Classifier 1 will use 14% of the training samples of the abundant class (class 1) plus 100% of the training samples of the minority class.
    #     - Classifier 2 uses 10% of the abundant class plus 100% of the minority class samples.
    #     - Classifier 3 (being the one that showed a fair consistency in the evaluation metrics) uses 100% of the training set.
    Xtr, ytr = resample_datasets(X_train, y_train, flags['optimalRatio'])


    ## ---- Model Selection

#     clfs = {
#         'clf_1':LogisticRegression(random_state=seed),
#         'clf_2':RandomForestClassifier(random_state=seed), #DecisionTreeClassifier(random_state=seed),
#         'clf_3':SGDClassifier(loss='log',random_state=seed)}


    ## ---- Training and prediction

    # The following script assumes 2 things:
    # 1. The top models have been selected (hyperparametrized or not).
    # 2. The dataset is imbalanced
    # Trains the selected models and outputs the predicted values for the training and validation sets.
    # In addition a voting method aids the prediction of the classifiers.

    print('Training...')
    ytr, ytr_pred, yval_pred, yval_soft, yval_hard = m_training_prediction(Xtr, ytr, X_val, clfs, flags)

    ## ---- Gather metrics

    t_time = time()-start
    df = get_results(ytr, ytr_pred, yval_pred, y_val, yval_soft, yval_hard, clfs, flags)
    df['db_size'] = str(X_train.shape)
    df['time'] = round(t_time,1)
    df.to_csv('results/'+flags['name']+'.csv')
    print("\nDone in {:.0f} seconds\n".format(t_time))


def m_preprocess(data, f):
    '''Joins stages of preprocessing, vectorizing
    and returns the training and validation sets'''
    data_features = preprocessing(data, f['lemStem']) ## --- Preprocessing: ~ 17s

    # Splitting into validation and training sets:
    # Val and train hold about the same ratio of imbalance.
    train, validation, y_train, y_val = train_test_split(data_features,
                                                         data.PreventiveFlag.values,
                                                         test_size=f['trainValRatio'],
                                                       random_state=f['rndseed'])

    # Vectorizing with CountVectorizer and TFIDFVectorizer:
    # remove english stop words
    X_train, vc, vt = vectorize(True, train, False, False, f['n-grams'])
    X_val, vc, vt = vectorize(False, validation, vc, vt, f['n-grams'])

    print('Training and Validation set sizes: ',X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val

def m_training_prediction(Xtr, ytr, X_val, clfs, f):

    '''Performs full training as well as soft and hard VotingClassifier
    returns the predictions
    '''
    # start = time()
    # DT 17s - 201s (with ng = 2) - 628s with ng=3, 827 with ng=4
    # RF 799s ng = 3
    # Xtr, ytr = resample_datasets(X_train, y_train, optimal_ratios)

    ### Training and Evaluation:
    ytr_pred, yval_pred, yval_proba = train_and_predict(Xtr, ytr, X_val, clfs, f['N_Models'])

    # Soft voting: predict the class with the highest class probability, averaged over all the individual classifiers.
    yval_soft = voting(yval_proba, 'soft')
    # Hard voting
    yval_hard = voting(yval_pred, 'hard')

    # t_time = "{:.0f} seconds".format(time()-start)

    return ytr, ytr_pred, yval_pred, yval_soft, yval_hard

def get_3_sets(n, pr):
    '''Function that receives an integer and a list with 2 floats
        Returns the intervals of 3 sections of a list of size n
        Those sections have a ratio specified by pr
        * Should be generalized*
    '''
    bucket = [i for i in range(n)]
    shuffle(bucket)
    n = len(bucket)
    h1_size = int(n*pr[0])
    h2_size = int(n*pr[1])
    h3_size = h1_size+h2_size
    # print(h1_size,h2_size,h3_size)
    m1 = bucket[:h1_size]
    m2 = bucket[h1_size:h3_size]
    m3 = bucket[h3_size:]
#     print(len(m1),len(m2),len(m3),sep='\n')
# print(m1,m2,m3,sep='\n')
    return [m1,m2,m3]


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
    # TO DO:
#         - Join single letters as a result of splitting accronyms joined by a symbol such as: U/A
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
    return idesc.lower()





def eda_remove_mess(corpus_all):
    analyzer = CountVectorizer(stop_words='english', token_pattern='[a-zA-Z]{2,}').build_analyzer()  # Added
    vectorizer_count = CountVectorizer(ngram_range=(1,1))
    vectorizer_count.fit_transform(corpus_all)
    print("There are {} different words and numbers in the Item Description".format(len(vectorizer_count.get_feature_names())))
    return vectorizer_count.get_feature_names()


def get_pos(word):
    return nltk.pos_tag([word])[0][1][0].lower()

def get_lemma(word):

    lemmer = WordNetLemmatizer()

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



def preprocessing(df, flag=0):
    '''Preprocess data:
        - Calls function text_analysis:
            - Converts the corpus in lower case
        - If flag=0 or 2, it performs lemmatization
        - If flag=1 or 2, it performs stemming
       Returns a list of the whole vocabulary stemmed and/or lemmatized
    '''

    df['ItemDescription'] = df.ItemDescription.apply(lambda word: text_analysis(word))
    df['Diagnosis'] = df.Diagnosis.apply(lambda word: text_analysis(word))

    if flag == 0 or flag == 2:
        # Lemmatization: converts into single synonyms and converts plural to singular
        print('Lemmatizing...')

        f_lemmatize = lambda text: " ".join([get_lemma(word) for word in text.split()])

        df['ItemDescription'] = df.ItemDescription.apply(f_lemmatize)
        df['Diagnosis'] = df.Diagnosis.apply(f_lemmatize)

    if flag == 1 or flag == 2:
        # Stemming
        print('Stemming...')
        stemmer = PorterStemmer()
        f_stemming = lambda text: " ".join([stemmer.stem(word) for word in text.split()])

        df['ItemDescription'] = df.ItemDescription.apply(f_stemming)
        df['Diagnosis'] = df.Diagnosis.apply(f_stemming)

    return df.ItemDescription.values + " " + df.Diagnosis.values

def vectorize(train, corpus_all, vectorizer_count, vectorizer_tf, n):
    # Transforms text into a sparse matrix of n-gram counts.
    if train:
        analyzer = CountVectorizer(stop_words='english', token_pattern='[a-zA-Z]{2,}').build_analyzer()  # Added
        vectorizer_count = CountVectorizer(ngram_range=(1,n), analyzer=analyzer)
        one_hot = vectorizer_count.fit_transform(corpus_all)
    else:
        vectorizer_count._validate_vocabulary()
        one_hot = vectorizer_count.transform(corpus_all)


    # Transform a count matrix to a normalized tf or tf-idf representation:

    # TFIDF
    # Term frequency: this summarizes how often a given word appears within a document
    # Inverse document frequency: This downscales words that appear a lot across documents
    # Transforming the matrix based on the learnt frequencies or weights


    if train:
        vectorizer_tf = TfidfTransformer()
        freq = vectorizer_tf.fit_transform(one_hot)
    else:
        freq = vectorizer_tf.transform(one_hot)

    return freq.toarray(), vectorizer_count, vectorizer_tf

def resample_datasets(X_train, y_train, resample_ratio=[0.1,0.3]):

    '''Divide the train set of the abundant class in 3 parts with different ratios '''

    # Due to imbalanced dataset: Ensemble different resampled datasets:
    # Get the instance of the minority class and majority class

    min_x_train = X_train[(y_train == 1)]
    maj_x_train = X_train[(y_train == 0)]

    # print(maj_x_train.shape, min_x_train.shape)

    # 0.07% of the # of samples of class 0 ~ about the same as the # of samples of class 1
    m = get_3_sets(len(maj_x_train), resample_ratio)

    X_train1 = np.concatenate((maj_x_train[m[0]],min_x_train))
    y_train1 = np.concatenate((y_train[m[0]],y_train[(y_train == 1)]))
    X_train2 = np.concatenate((maj_x_train[m[1]],min_x_train))
    y_train2 = np.concatenate((y_train[m[1]],y_train[(y_train == 1)]))
    X_train3 = np.concatenate((maj_x_train[m[2]],min_x_train))
    y_train3 = np.concatenate((y_train[m[2]],y_train[(y_train == 1)]))

    print('Datasets sizes:\n',X_train1.shape,X_train2.shape,X_train3.shape)
    # print(len(X_train1)+len(X_train2)+len(X_train3),len(maj_x_train)+3*len(min_x_train))

    X_dico = {'1':X_train1, '2':X_train2, '3':X_train3}
    y_dico = {'1':y_train1, '2':y_train2, '3':y_train3}

    return X_dico, y_dico



def train_and_predict(Xtr, ytr, X_val, clfs, n_models):
    '''Function that takes as input:
        - the classifiers (a dictionary called: clfs)
        - The training set: Xtr
        - With its labels: ytr
        - The validation set: X_val

        and returns its predictions on:
        - The training set: y_tr_pred
        - The validation set: y_val_pred and y_val_proba

    '''
    y_val_proba = {}
    y_val_pred = {}
    y_tr_pred = {}

    for i in range(1,n_models+1):

        # Fit
        clfs['clf_' + str(i)].fit(Xtr[str(i)], ytr[str(i)])
        # Predict probabilities
        y_val_proba['clf_' + str(i)] = clfs['clf_' + str(i)].predict_proba(X_val)
        # Predict classes
        y_val_pred['clf_' + str(i)] = clfs['clf_' + str(i)].predict(X_val)
        # These predictions can then be used to evaluate the classifier:
        y_tr_pred['clf_' + str(i)] = cross_val_predict(clfs['clf_' + str(i)],
                                                       Xtr[str(i)],
                                                       ytr[str(i)],
                                                       cv=10)
    return y_tr_pred, y_val_pred, y_val_proba

def voting(y_hat, vType):
    # Convert y_probabilities in a dataframe for better analysis
    n_models = len(y_hat.keys())
    my_dict = pd.DataFrame({'clf': list(y_hat.keys()),    # score = range(len(tst.valuesof sgd))
                'tags': list(y_hat.values())}, columns = ['clf', 'tags'])
    tags = my_dict['tags'].apply(pd.DataFrame)
    df_temp = pd.DataFrame()

    for i,d in enumerate(y_hat.keys()):
    #     print(type(d))
        if vType == 'soft':
            tags[i].columns = [str(d)+':-1',str(d)+':+1']
        else:
            tags[i].columns = [str(d)]
        df_temp = pd.concat([df_temp, tags[i]], axis=1)
    if vType == 'soft':
        df_temp["0"] = df_temp.iloc[:,range(0,n_models*2,2)].apply(lambda x: sum(x)/n_models, axis=1)
        df_temp["1"] = df_temp.iloc[:,range(1,n_models*2,2)].apply(lambda x: sum(x)/n_models, axis=1)
        df_temp['Label'] = df_temp.loc[:,['0','1']].idxmax(axis=1)

    else:
        df_temp['Label'] = df_temp.apply(lambda x: int(statistics.mode(x)), axis=1)
    return df_temp.Label.values.astype(float)


def get_results(ytr, ytr_pred, yval_pred, y_val, yval_soft, yval_hard, clfs, f):

    res = get_metrics(pd.DataFrame(), ytr["1"], ytr_pred["clf_1"], "Tr-"+str(clfs['clf_1']).split('(')[0])
    res = get_metrics(res, y_val, yval_pred["clf_1"], 'Val-'+str(clfs['clf_1']).split('(')[0])
    res = get_metrics(res, ytr["2"], ytr_pred["clf_2"], 'Tr-'+str(clfs['clf_2']).split('(')[0])
    res = get_metrics(res, y_val, yval_pred["clf_2"], 'Val-'+str(clfs['clf_2']).split('(')[0])
    res = get_metrics(res, ytr["3"], ytr_pred["clf_3"], 'Tr-'+str(clfs['clf_3']).split('(')[0])
    res = get_metrics(res, y_val, yval_pred["clf_3"], 'Val-'+str(clfs['clf_3']).split('(')[0])
    res = get_metrics(res, y_val, yval_soft, 'Val Soft')
    res = get_metrics(res, y_val, yval_hard, 'Val Hard')
    res.reset_index(inplace=True)
    res.columns = ['Phase','AUC','F1_C0','F1_C1','MSE','Prec_C0','Prec_C1','Recal_C0','Recal_C1']
    res['LemmVsStem'] = f['lemStem']
    res['TrValRatio'] = f['trainValRatio']
    res['nGrams'] = f['n-grams']

#     return res.loc[res.Phase.str.startswith('Validation'),['Phase','AUC','F1_C1']]
    return res

def get_metrics(df, y, yp, ind_name):
# Evaluation Metrics:

    d_roc = round(roc_auc_score(y, yp),3)
    d_mse = round(mean_squared_error(y, yp),3)
    d_prec = precision_score(y, yp, average=None)
    d_recl = recall_score(y, yp, average=None)
    d_f1 = f1_score(y, yp, average=None)
#     d_report = metrics.classification_report(y, yp, target_names=['class 0', 'class 1'])


    return pd.concat([df, pd.DataFrame({
                            'Prec_C0':round(d_prec[0],3),
                            'Prec_C1':round(d_prec[1],3),
                            'Recal_C0':round(d_recl[0],3),
                            'Recal_C1':round(d_recl[1],3),
                            'F1_C0':round(d_f1[0],3),
                            'F1_C1':round(d_f1[1],3),
                            'AUC':d_roc,
                            'MSE':d_mse}, index=[ind_name])])


### Plotting functions

def compare_2_classifiers(dt_config, rf_config):
    # Compare to experiments

    dt_config['Phase'][2] = "Tr-clfX"
    dt_config['Phase'][3] = "Val-clfX"
    rf_config['Phase'][2] = "Tr-clfX"
    rf_config['Phase'][3] = "Val-clfX"

    model_c = pd.concat([dt_config,rf_config]).sort_index()#values('MSE')

    f, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(20,34))
    barWidth = 0.35
    cols = ['MSE', 'AUC', 'F1_C0', 'F1_C1', 'Prec_C0', 'Prec_C1','Recal_C0', 'Recal_C1']
    plot_comparisson_metrics(f,axes, model_c, cols, barWidth)
    f.savefig(path+'/graphics/comparissonMetricsRFvsDT.png', dpi=f.dpi)

    f1, axes = plt.subplots(nrows = 4, ncols = 2, figsize=(20,34))
    comparisson_clfs(f1,axes, model_c, cols, barWidth)
    f1.savefig(path+'/graphics/comparissonRFvsDT.png', dpi=f.dpi)
    
def plot_comparisson_metrics(f,axes, model_c, cols, barWidth):
    axes = axes.ravel()

    for idx, ax in enumerate(axes):

        bar1 = model_c.loc[model_c.db_size=='(7000, 4700)', cols[idx]].values
        bar2 = model_c.loc[model_c.db_size=='(7000, 4853)', cols[idx]].values
        # Set position of bar on X axis
        r1 = np.arange(len(bar1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.barh(r1, bar1, color='royalblue', height=barWidth,  label='4700 feats')
        ax.barh(r2, bar2, color='coral', height=barWidth, label='4853 feats')

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
            ax.set_yticklabels(model_c.Phase.unique())
            ax.legend(loc=leg_loc, fontsize=18)
        else:
            ax.get_yaxis().set_ticks([])
        # Create legend & Show graphic
        ax.set_title(cols[idx], fontsize=20)
        ax.set_xticks(np.linspace(0, 1, num=6))

    plt.show()

def comparisson_clfs(f,axes, model_c, cols, barWidth):
    axes = axes.ravel()
    phases = model_c.Phase.unique()

    for idx, ax in enumerate(axes):


        bar1 = model_c.loc[(model_c.db_size=='(7000, 4700)')&(model_c.Phase==phases[idx]), cols].values[0]
        bar2 = model_c.loc[(model_c.db_size=='(7000, 4853)')&(model_c.Phase==phases[idx]), cols].values[0]
        # Set position of bar on X axis
        r1 = np.arange(len(bar1))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        ax.barh(r1, bar1, color='royalblue', height=barWidth,  label='4700 feats')
        ax.barh(r2, bar2, color='coral', height=barWidth, label='4853 feats')

        for idx2,i in enumerate(ax.patches):

    #         if idx == 0: # MSE
    #             x0 = i.get_width()-.02
    #             leg_loc = 'upper right'
    #         else:
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
