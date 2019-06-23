# Credit-card-fraud-detection-using-statistical-analysis-and-machine-learning.

Hi guys! In this blog post today, I will talk about detecting fraudulent transactions made with credit cards! In order to solve this problem of detecting whether or not a given transaction is fraud, we will use various supervised as well as unsupervised Machine Learning algorithms. We will see how accurate each of these models are in determining whether a given transaction is fraudulent or not.

Before we begin with our analysis, let's understand a bit more about the dataset that is provided to us. The dataset that we have can be downloaded from the Kaggle link at <here>. The given dataset contains information about transactions that were made using credit cards in the month of September, 2013. The transaction data is captured over a duration of 2 days. We have 492 cases of fraudulent transactions out of a total number of 284807 number of transactions recorded during those two days. 

The dataset as such is severely imbalanced with the percentage of fraud transactions being 0.172% of the total data. The dataset contains only transformed numerical features which are a result of a PCA transormation. The original data is not provided to us due to security reasons and to protect the identity of the customers. 


Having said that, there are two features which are not transformed using PCA - 'Time' and 'Amount'. These features are given as it is. The 'Time' feature basically says how much time for each transaction has elapsed since the first transaction in the dataset has taken place. The 'Amount' feature gives us information about the transaction amount for each of the transactions. The fraudulent transactions are denoted by class label 1 and the non fraudulent transactions are denoted by class label 0.

What are our main goals for this problem?

1. We will leverage the the very small data that is provided to us.
2. We will implement standard outlier detection algorithms like LOF and see how successful they are in detecting fradulent transactions.
3. For implementing Machine Learning models, we will follow two approaches for resampling ofour data: 
First, we will undersample the data and create a balanced dataset containing equal number of points from both the classes.
Second, we will oversample the dataset by adding synthetic points using SMOTE sampling technique.
4. We will also leverage the concept of Autoencoders and build an encoder-decoder neural network which will be used to learn the low level representations of the transformed PCA data. 
5. Lastly, we will compare all our models and see which models turns out to better than the rest.

### What is the business problem that we are trying to solve?

Credit card fraud refers to a wide range of activities which includes theft of money using either credit cards or debit cards. The theft can be either online or ofline. An ofline theft generally involves withdrawing money from an ATM machine physically using a stolen credit card. An online theft involves any online transaction using the card without the prior consent of the owner. Both as a customer and as a bank, fraudulent credit card transactions can give you nightmares! From a bank's point of view, it's very essential to identify whether a transaction is fraudulent or not because they don't want to lose money or don't want to lose the faith that there customer has entrusted upon them. In such a scenario it becomes a necessity to build a robust system which can be used to determined fraudulent transactions. 

While designing the system we should keep in mind that the cost of misclassification of a fraudulent transaction is very high. We don't to end up with a system which might classify a fradulent transaction as a non-fradulent one. Such a system in machine learning is also called a high recall system. It's important for the bank to know which of the transactions are fraud, at the same time it is important to understand which of the transactions are not fraud. 

### What are the real world business constraints and what metrics we will use to evaluate our model?

The dataset that we have is a real world dataset which is severely imbalanced. This is expected because if you imagine, the number of valid transactions has to be much much greater than the number of fraud transactions in the world, or else everyone would have been bankrupt by now! Due to the severely imbalanced dataset building the best Machine Learning models will be a challenge. But this is a constraint we have to deal with. Since the dataset is imbalanced we will use roc-auc as our key metric.

Another important factor we must keep in mind is the cost of making an incorrect prediction for the fraudulent class is very very high. It's okay if the model classifies a non-fraudulent transaction as a fraudulent one, but classifying a fraudulent transaction as a non-fradulent one is very very costly, because at the end of the day no one want to lose money. Due to this reason we must always keep a close look at the recall metric and make sure that the false negatives are as low as possible. We will print the confusion matrix and generate classification reports for each models and monitor the false positives.

### Mapping the problem to a real world machine problem.

As we have discussed till now, we will focus on building models which can correctly identify fraudulent transactions. We should build a model which maximizes the roc-auc score as well as minimizes the number of false positives. In other words, we want a model which has a high recall. 

### Exploratory data analysis.

Let's look at the data that is provided to us. The CSV file has been downloaded and renamed to 'data.csv'. The dataset contains 284807 transactions out of which only 492 (0.1727%) transactions are fraud. There are 28 PCA transformed features that are provided to us. These PCA features are not at all interpretable. A high level statistics of the dataset reveals that almost 50% transactions involved 22 dollars or less, almost 75% transactions involves amount less than 77 dollars. The highest transaction amount in a fraudulent scenario is 2125 dollars, whereas the highest transaction recorded in case of a non fraud scenario is 25691 dollars. The median values of all the fraudulent transactions is 9 dollars whereas the median values of all the non fraud transactions is 22 dollars.

By looking at the distribution of all the features we can see that there are some features whose distributions are skewed to the left, there are some features whose distributions are skewed to the right and there are some features which appears to have a normal gaussian distribution. Almost all the features have their distributions mean at 0. Some features like V11, V15, V13, V18, V19 has a wider spread compared to other features. Some features like V6, V7, V8 and V28 have a very low spread as compared to other features. 

On EDA of the 'Time' feature, we can see that the number of transactions falls sharply during a particular time interval. There are some regions of time where the number of transactions are very high and there some regions in time where the number of transactions are very low.

On EDA of the 'Amount' features, we see that the distribution is highly skewed towards the left. There are very small number of higher value transactions which happens in the course of 2 days.

Using seaborn we can draw corelation heatmaps which are basically same as corelation matrices. In case of a corelation heatmaps we will use color codings instead of corelation coefficient values to determine whether features have a positive or a negative corelation. A red color indicates the features have a strong positive corelation between them and a blue color indicates that two features have a strong negative corelation between themeselves. 

### Under-sample the dataset to balance the classes

Since the dataset is highly imbalanced there are broadly two strategies we will follow to correctly sample our dataset - under-sampling the dataset and over-sampling the dataset.

In this section we will use a data under-sampling technique where we will sample the data based on the number of instances we have in our minority class. In order to create the final dataset, we will take equal number of sample from both the classes, concatenate them into a single dataset and perform random shuffling to shuffle the data. The resultant dataset will contain 50% points from each of the classes.

Under-sampling helps us get rid of the problem of data imbalanced, but at the same time we are discarding huge amount of data to build our models. We can negate this by using certain data over sampling strategies. In a later section, we will implement something called SMOTE algorithm - a technique used to oversample an imbalanced dataset by adding synthetic points. We will discuss about SMOTE when we implement it.

### Box Plots

What is a box-plot and why is it useful?

Box plot is a very powerful statistical tool which can be used to represent statistical information like median, quantiles and inter-quartile range in a single plot.

### Splitting the data into train and test datasets

Before building our machine learning models, we will split the dataset in such a way that 80% of the undersample data goes to our training set and 20% data from the undersampled class goes to our test set. We will make use of the 'stratify' argument to make sure we have equal distribution of class labels in both the training as well as test sets. 

After the initial splitting, we have 787 points in our training dataset and 197 points in our test dataset. We will build machine learning models using these 787 points and then evaluate the performance of each of our models on the test set. 

### Check the distribution of train and test data after splitting the original dataset

This is a sanity check we need to perform to check if the distribution of class labels is same in both the training as well as the test set. We can see that in both the train as well as the test sets, the class labels are distributed almost equally at 50% data points from each of the classes.

### Dimensionality reduction using TSNE

T-SNE stands for t-distributed Stochastic Neighbor Embedding.

T-SNE is a tool which is used to visualize high dimensional data in 2 or 3 dimensions. In this section we will try to visualize the high dimensional data in a 2D plot. T-SNE tries to preserve the neighborhood distances between each of the data points when we project them onto a lower dimensional space. We will try plotting the T-SNE with various values of perplexities and see if the resultant plot can separate the positive and negative classes well.

#### Observations:

Here we have run the algorithm for various values of perplexities. Perplexity values basically tells the T-SNE algorithm the number of neighborhood distances it should preserve. Here we can see that T-SNE plots are able to accurately cluster the data points at various values of perplexity based on whether or not they are fraudulent transactions. A partial separability suggest that our Machine Learning models should perform well on the given dataset. 

### Functioin to plot Confusion Matrix, Precision Matrix, Recall Matrix

We will use this function to draw the confusion matrix, precision matrix and the recall matrix. We will use the confusion matrix to keep an eye on the false positive values and the recall values. Our main objective of this case study is to build a model which has a high recall value.

Confusion Matrix is a tool which helps us to evaluate the performance of our classification model on unseen data. It's a very important tool to evaluate metrics such as Precision, Recall, Accuracy and Area under the ROC curve using these four values - False Positives (FP), False Negatives (FN), True Positives (TP) and True Negatives (TN).

Let us understand these four metrics in a bit more detail with regards to the given problem. 

True Positives (TP): Here the model has predicted the transaction to be fraudulent and in real life the transaction is fraudulent.

True Negatives (TN): Here the model has predicted a transaction to be a non-fraudulent one and in real life the transaction is non-fraudulent. 

False Positives (FP): Here the model has predicted the transactions to be fraudulent whereas in real life the given transaction is not fraudulent. These are also known as Type 1 errors.

False Negatives (FN): Here the model has predicted the transactions to be non-fraudulent where as in real life the transactions are fraudulent. These are also known as Type 2 errors.

Ideally, for a perfect model, we would want the values of TPs and TNs to be very high and our FPs and FNs to be very low. Also, for this problem it's an absolute necessity to keep the False Negative values as low as possible. In the real world Type 2 errors are much more sever than Type 1 errors. Imagine this scenario - our model predicts a fraudulent transaction as a non fraudulent one. This is much more severe than predicting a non-fraudulent transaction as a fraudulent one.

Recall tells us that out of the total number of actual/correctly classified classes how many did our model predicted to belong to the correctly classified class?

Precision tells us that out of the total number of predictions how many of them are actually predicted to be true?

### Function to plot the ROC-AUC Curve

ROC curve stands for Receiver operating characteristic curve. In machine learning, ROC curves helps us evaluate our models performance at various threshold settings. ROC curves is a probability curve and AUC stands for the area under the ROC curve. Generally a ROC-AUC curves gives us an idea about how well our model is capable of distinguishing between various class labels. IN ROC-AUC curve, the value of the true positive rates and false positive rates are plotted against each other at various threshold settings. Higher the value of an ROC-AUC curve, the better will be our model in predicting a class 0 label as class 0 and class 1 label as class 1. For this case study, class 1 signifies a fraudulent transaction and class 0 signifies a non-fraudulent transaction. 

While plotting the ROC-AUC curve, the TPR is taken in Y-Axis and the FPR is taken at X-axis. TPR is also known ans Recall. Mathematically TPR is defined as (TP/TP+FN), and FPR is defined as (FP/TN+FP). We will have to optimize our Machine Learning models such that they maximize the ROC-AUC score.

### Generic function to run any model and print the classification metrics

This function is used to evaluate our model on unseen data. We will first obtain the best estimator using either grid search or random search. We will use the best estimator from our model to print the roc-auc scores, the accuracy scores, the recall score and the f1 score. F1 score as we know is the harmonic mean between precision and recall scores. We will also use this function to generate the classification report for each of our models. 

### Generic function to print grid/random search results/attributes

This function will be used to print the best estimator obtained using grid search/random search. For each estimator, we will print the best parameters for a given function along with their best scores on the cross validation dataset.

### Undersmapling

                             Accuracy        Recall      ROC-AUC
                            ----------      --------     --------
Logistic Regression        : 88.83%          97.96%       0.9887
KNN Classifier             : 92.89%          87.76%       0.9853
Decision Trees Classifier  : 89.85%          81.63%       0.9445
Linear SVC                 : 91.88%          84.69%       0.9856
Random Forest Classifier   : 93.4%           88.78%       0.9861
XGBoost Classifier         : 94.42%          92.86%       0.9874
Neural Networks            : 95.94%          96.94%       0.9594

### Oversampling


                             Accuracy        Recall      ROC-AUC
                            ----------      --------     --------
Logistic Regression        : 97.42%          90.82%       0.975
KNN Classifier             : 99.65%          85.71%       0.9326
Decision Trees Classifier  : 98.47%          81.63%       0.9513
Random Forest Classifier   : 99.78%           85.71%       0.9787
XGBoost Classifier         : 99.93%          82.65%       0.966

### Summary of the entire project:

Hi guys! In this blog post today, I will talk about detecting fraudulent transactions made with credit cards! In order to solve this problem of detecting whether or not a given transaction is fraud, we will use various supervised as well as unsupervised Machine Learning algorithms. We will see how accurate each of these models are in determing whether a given transaction is fraudulent or not.

Before we begin with our analysis, let's understand a bit more about the dataset that is provided to us. The dataset that we have can be downloaded from the Kaggle link at <here>. The given dataset contains information about transactions that were made using credit cards in the month of September, 2013. The transaction data is captured over a duration of 2 days. We have 492 cases of fraudulent transactions out of a total number of 284807 number of transactions recorded during those two days. 

The dataset as such is severely imbalanced with the percentage of fraud transactions being 0.172% of the total data. The dataset contains only transformed numerical features which are a result of a PCA transormation. The original data is not provided to us due to security reasons and to protect the identity of the customers. 


Having said that, there are two features which are not transformed using PCA - 'Time' and 'Amount'. These features are given as it is. The 'Time' feature basically says how much time for each transaction has elapsed since the first transaction in the dataset has taken place. The 'Amount' feature gives us informatiuon about the transaction amount for each of the transactions. The fraudulent transactions are denoted by class label 1 and the non fraudulent transactions are denoted by class label 0.

What are our main goals for this problem?

1. We will leverage the the very small data that is provided to us.
2. We will implement standard outlier detection algorithms like LOF and see how successful they are in detecting fradulent transactions.
3. For implementing Machine Learning models, we will follow two approaches for resampling ofour data: 
First, we will undersample the data and create a balanced dataset containing equal number of points from both the classes.
Second, we will oversample the dataset by adding synthetic points using SMOTE sampling technique.
4. We will also leverage the concept of Autoencoders and build an encoder-decoder neural network which will be used to learn the low level representations of the transformed PCA data. 
5. Lastly, we will compare all our models and see which models turns out to better than the rest.

What is the business problem that we are trying to solve?

Credit card fraud refers to a wide range of activities which includes theft of money using either credit cards or debit cards. The theft can be either online or ofline. An ofline theft generally involves withdrawing money from an ATM machine physically using a stolen credit card. An online theft involves any online transaction using the card without the prior consent of the owner. Both as a customer and as a bank, fraudulent credit card transactions can give you nightmares! From a bank's point of view, it's very essential to identify whether a transaction is fraudulent or not because they don't want to lose money or don't want to lose the faith that there customer has entrusted upon them. In such a scenario it becomes a neccessity to build a robust system which can be used to determined fraudulent transactions. 


While designing the system we should keep in mind that the cost of misclassification of a fraudulent transaction is very high. We don't to end up with a system which might classify a fradulent transaction as a non-fradulent one. Such a system in machine learning is also called a high recall system. It's important for the bank to know which of the transactions are fraud, at the same time it is important to understand which of the transactions are not fraud. 

What are the real world business constarints and what metrics we will use to evaluate our model?

The dataset that we have is a real world dataset which is severly imabalanced. This is expected because if you imagine, the number of valid transactions has to be much much greater than the number of fraud transactions in the world, or else everyone would have been bankrupt by now! Due to the severely imabalanced dataset building the best Machine Learning models will be a challenge. But this is a constraint we have to deal with. Since the dataset is imabalanced we will use roc-auc as our key metric.

Another important factor we must keep in mind is the cost of making an incorrect prediction for the fraudulent class is very very high. It's okay if the model classifies a non-fraudulent transaction as a fraudulent one, but classifying a fraudulent transaction as a non-fradulent one is very very costly, beause at the end of the day no one want to lose money. Due to this reason we must always keep a close look at the recall metric and make sure that the false negatives are as low as possible. We will print the confusion matrix and generate classification reports for each models and monitor the false positives.

Mapping teh problem to a real world machine problem.

As we have discussed till now, we will focus on building models which can correctly identify fraudulent transactions. We should build a model which maximizes the roc-auc score as well as minimizes the number of false positives. In other words, we want a model which has a high recall. 

Exploratory data analysis.

Let's look at the data that is provided to us. The CSV file has been downloaded and renamed to 'data.csv'. The dataset contains 284807 transactions out of which only 492 (0.1727%) transactions are fraud. There are 28 PCA transformed features that are provided to us. These PCA features are not at all interpretable. A high level statistics of the dataset reveals that almost 50% transactions involved 22$ or less, almost 75% transactions involves amount less than 77$. The highest transaction amount in a fraudulent scenario is 2125$, whereas the highest transaction recorded in case of a non fraud scenario is 25691$. The median values of all the fraudulent transactions is 9$ whereas the median values of all the non fraud transactions is 22$.

By looking at the distribution of all the features we can see that there are some features whose distributions are skewed to the left, there are some features whose distributions are skewed to the right and there are some features which appears to have a normal gaussian distribution. Almost all the features have their distributions mean at 0. Some features like V11, V15, V13, V18, V19 has a wider spread compared to other features. Some features like V6, V7, V8 and V28 have a very low spread as compared to other features. 

On EDA of the 'Time' feature, we can see that the number of transactions falls sharply during a particular time interval. There are some regions of time where the number of transactions are very high and there some regions in time where the number of transactions are very low.

On EDA of the 'Amount' features, we see that the distribution is highly skewed towards the left. There are very small number of higher value transactions which happens in the course of 2 days.

Before proceeding to build ML models we will column standardize both the time as well amount features. In this way we can ensure that both the time and amount features are at scale with the remaining features. 

Corelation Matrices:

A corelation matrix is basically a table which shows the corelation coefficients between pairs of variables. Each cell in a corelation matrix shows the corelation between two variables. If the corelation coefficient amongst two variable is high it means that as the value of one variable increases, the value of the other variable also increases. This is also called a positive corelation. A negative corelation on the other hand means that if the value of one variable increases, the value of the other variable decreases and vice-versa. Thus in simple words a corelation matrix helps us understand which pairs of points has the highest corelation. 

Using seaborn we can draw corelation heatmaps which are basically same as corelation matrices. In case of a corelation heatmaps we will use color codings instead of corelation coefficient values to determine whether features have a positive or a negative corelation. A red color indicates the features have a strong positive corelation between them and a blue color indicates that two features have a strong negative corelation between them.themeselves. 


