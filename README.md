# Disease-Prediction-from-Symptoms

This project is about prediction of disease based on symptoms using machine learning. Machine Learning algorithms such as Naive Bayes, Decision Tree and Random Forest are employed on the provided dataset and predict the disease. Its implementation is done through the python programming language. The research demonstrates the best algorithm based on their accuracy. The accuracy of an algorithm is determined by the performance on the given dataset.

Final count of diseases in the dataset were a total of 261 and 500+ symptoms. To multiply the dataset, each disease’s symptoms are picked up, combinations of the symptoms are created and added as new rows in the dataset.

For example, a disease A, having 5 symptoms, now has a total of (2⁵ − 1) entries in the dataset. The dataset, after pre-processing and multiplication, contains around 8835 rows with 489 unique symptoms. This was done to tackle the problem of only having a single row for each disease which results in poor training of data. This idea was inspired by the real-world scenario where a patient even showing some of the symptoms of all the symptoms for a disease can be suffering from that disease, therefore it is a logical extension of the dataset.

# Dataset

The dataset for this problem is downloaded from here: 
```
https://impact.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
```

This dataset has 3 columns:
```
Disease  | Count of Disease Occurrence | Symptom
```

You can either copy paste the whole table from here to an excel sheet or scrape it out using Beautifulsoup.

# Directory Structure

```
|_ dataset/
         |_ training_data.csv
         |_ test_data.csv

|_ saved_model/
         |_ [ pre-trained models ]

|_ main.py [ code for laoding dataset, training & saving the model]

|_ Disease-Prediction-from-Symptoms-checkpoint.ipynb [ IPython Notebook for Loading dataset, training model and Inference ]
```

## Pre-processing of Dataset and Solution Sketch

The scraped symptoms are pre-processed to remove similar symptoms with different names (For example, headache 😨 and pain in the forehead 😨). To do so, symptoms are expanded by appending synonyms of terms in the symptom string and computing 💻 Jaccard Similarity Coefficient for each pair of symptoms.
```
if Jaccard(Symptom1,Symptom2) > threshold:
    Symptom2->Symptom1
```

## User Symptom pre-processing

The system accepts symptom(s) in a single line, separated by comma (,). Subsequently, the following pre-processing steps are involved:
Split symptoms into a list based on comma
Convert the symptoms into lowercase
Removal of stop words
Tokenization of symptoms to remove any punctuation marks
Lemmatization of tokens in the symptoms
The processed symptom list is then used for symptom expansion.
Symptom Expansion, Symptoms Suggestion and Selection
Each user symptom is expanded by appending a list of synonyms of the terms in the synonym string. The expanded symptom query is used to find the related symptoms in the dataset. To find such symptoms, each symptom from the dataset is split into tokens and each token is checked for its presence in the expanded query. Based on this, a similarity score is calculated and if the symptom’s score is more than the threshold value, that symptom qualifies for being similar to the user’s symptom and is suggested to the user.

```
tokenA->tokens(Symptom A)

tokenSyn->tokens(synonym string)

matching->intersect(tokenA,tokenSyn)

score->count(matching)/count(tokenA)

if score>threshold: select Symptom A
```

The user selects one or more symptoms from the list. Based on the selected symptoms, other symptoms are shown to the user for selection which is among the top co-occurring symptoms with the ones selected by the user initially. The user can select any symptom, skip, or stop the symptom selection process. The final list of symptoms is compiled and shown to the user. Figure shows an example of the symptom suggestion and selection process.
Symptom Suggestion and Selection Process

## Training the models

After all the scraping and pre-processing, now it is time to rev up your enginesand do some magic (Not literally doing magic, just some extensive math) to train the machine learning models.

1. A binary vector is computed that consists of 1 for the symptoms present in the user’s selection list and 0 otherwise. A machine learning model is trained on the dataset, which is used here for prediction. The model accepts the symptom vector and outputs a list of top K diseases, sorted in the decreasing order of individual probabilities. As a common practice K is taken as 10. 

2. Multinomial Naïve Bayes, Random Forest, K-Nearest Neighbor, Logistic Regression, Support Vector Machine, Decision Tree were trained and tested with a train-test split of 90:10.

3. Multi layer Perceptron Neural Network was also trained and tested with the same split ratio.

4. You can find the implementation of all these models in main.py file.

5. Out of all these, Logistic Regression performed the best when tested against 5 folds of cross validation.

## Model training

1. Detected diseases using ML models
2. The probability of a disease is calculated as below.
3. ModelAccuracy->accuracy(model used)
4. DisASymp=Symptoms(DiseaseA)
5. match->intersect(DisASymp,userSymp)
6. matchScore->match/count(userSymp)
7. prob(DiseaseA)=matchScore * modelAccuracy
