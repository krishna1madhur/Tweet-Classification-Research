# Tweet-Classification-Research

This is a research project wherein I evaluated different classifiers and different classification techniques to develop a best classifier for the given training data. 

Used Supervised Classification Techniques like Naive bayes, SVM, Logistic Regresssion, Neural Netwroks, KNN etc; for multi-class text classification. 

Performed the tests using 10-Fold cross validation on training data to improve the classifier. Finally performed the tweet classification on test data without the use of it.

Classification Techniques Used for improving F Score: Stratified K Fold, Ensemble methods, Over Sampling and Under Sampling

Best Results: 
Obama- Voting (SGD,SVM, MultinomialNB)
POSITIVE
Precision: 59.5
Recall: 53.26
F Score: 56.21
NEGATIVE
Precision: 65.55
Recall: 55.68
F Score: 60.21
Overall Accuracy: 56.43

Romney- LinearSVC with sampling
POSITIVE
Precision: 46.91
Recall: 61.04
F Score: 53.05
NEGATIVE
Precision: 75.1
Recall: 67.45
F Score: 71.07
Overall Accuracy: 59.42
