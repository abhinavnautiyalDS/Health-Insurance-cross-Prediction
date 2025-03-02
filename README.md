![Uploading image.png…]()

# Health-Insurance-cross-Prediction

An insurance company that offers health insurance to its customers typically provides other insurance products through various marketing channels. In this scenario, we will develop a model to predict whether policyholders from the previous year will also be interested in the company's vehicle insurance.

## Problem Statement
Our client, an insurance company, offers vehicle insurance to its customers. Now, they need your help to build a model that predicts if last year's policyholders would also be interested in buying vehicle insurance.

Insurance works by providing compensation for specific losses, damages, illnesses, or deaths in exchange for regular payments called premiums. For example, you might pay Rs. 2000 each year for health insurance that covers up to Rs. 100,000 in medical costs. The insurance company can afford to cover these costs because, out of many customers paying premiums, only a few will need to use the insurance each year. This spreads the risk among all policyholders.

Vehicle insurance works the same way. Customers pay an annual premium, and in case of an accident, the insurance company compensates them up to a certain amount.

Predicting which customers might want vehicle insurance helps the company target its communication and improve its business strategy and revenue.

To create this prediction model, you have data on demographics (like gender, age, and region), vehicles (age and condition), and policy details (premium and sourcing channel).

## Variable Discription:

**bold text**- **id**: Unique identifier for each customer.

- **Age**: Age of the customer, ranging between 20 and 85 years.

- **Driving_License**: Binary variable indicating if the customer has a driving license (0: No, 1: Yes).

- **Region_Code**: Code representing the region of the customer.

- **Previously_Insured**: Binary variable indicating if the customer already has vehicle insurance (0: No, 1: Yes).

- **Annual_Premium**: The annual premium amount paid by the customer. Note: There are some outlier values as indicated by a sudden jump from the 75th percentile value to the maximum value.

- **Policy_Sales_Channel**: Code for the channel through which the policy was sold.

- **Vintage**: Number of days the customer has been with the company since they purchased insurance.

- **Response**: Binary target variable indicating if the customer is interested in vehicle insurance (0: No, 1: Yes).

## Dataset Information:

- This dataset contains 381109 rows and 12 columns
- There are no duplicate rows present in this dataset and no null values in any feature
- There are some outliers in the annual premium feature, which I addressed by capping them at the upper whisker value.
- I convert the Age values into categorical ranges such as 20-40, 40-60, and 60+. And Vintage column in to Categories like short-term,mid-term and long-term


![download (2)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/8ae5df25-7dda-411e-b394-2cc723c95de5)

## EDA 

### 1. **Distribution of Numerical Features**
   ![download (3)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/29d05a9f-5593-4736-9212-e7d64a007c1a)
   
### 2. **Distribution of Categorical columns**
![download (4)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/1bcbd4c5-9e99-4d38-8cf8-d29f5145af69)

- Most policyholders are aged between 20 and 40, with the fewest being 60+
- 99.8% of policyholders have a driving license.
- Most customers have vehicles that are 1-2 years old or less than 1 year old.
- Half of the policyholders have vehicle damages, while the other half do not.

### 3. **Relation of age_group with Annual prenium , Response and Gender**
![download (1)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/03ce41b0-e73f-44ff-ae21-08c551d36301)

- 60+ age group customers give more anuual prenium in comparison of rest age groups
- 20-40 and 40-60 age group of policyholders are more interested to take vehicle insurance

- Male policyholder are more than female one
- There are more female than male in 20-40 age group,and for rest of the age groups female are lesser in number than man

### 4. **How does Age correlate with Vehicle Age?**
![download (6)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/6de4b337-f112-4a54-91fe-c7d2831b37b7)

1. Policyholders with vehicles less than 1 year old have a median age of 25.
2. Policyholders around 50 years old generally have vehicles between 1 and 2 years old.
3. Individuals aged 55 years or more tend to own vehicles older than 2 years.

### 5. **how many previously insured people will be insured again? and Is there any relation between Vehicle Damage and Response?**
![download (7)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/020bac73-7ecc-4d3e-b938-09cfb306ebfb)

- if a policy holder insured previously then there is very lower chances that he/she will insured again
- Those policyholder having vehical damage are more likely to renewed their insurance policy

### 6. **proportion of individuals within each age group who were previously insured and then insured again**
![download (8)](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/1dfff596-6021-4351-a9e0-eb425a481117)

**Age Group 20-40:**

High Insurance Rate: Most have insurance.
Low Response Rate: Less interested in new offers, likely due to already having coverage.

**Age Group 40-60:**
Balanced Insurance Status: Evenly split between insured and uninsured.
Moderate Response Rate: Mixed interest in offers, influenced by varying needs and commitments.

**Age Group 60+:**
Low Insurance Rate: Few have insurance.
Low Response Rate: Least engaged, possibly due to existing coverage or lack of interest.


## **Handling Imbalanced Dataset**

**What is inbalanced datset**

If for a classification dataset, if the classes in my target feature are not in same proportion than we can say that the dataset is unbalanced.

To Handle this inbalanced dataset i use a Famous Technique called SMOTE stands for synthetic minority oversampling technique, this technique only focuses on minority class , it tried to create new minority sample from the nearest samples using Knn technique.

![image](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/cc7242c6-e542-4857-943a-23eec153bc2c)

## **Categorical Encoding ANd label Encoding**

- To ensure that all features contribute equally to the model training process, I have utilized techniques to standardize or normalize my dataset. One such technique is Min-Max Scaling, which scales the numerical features to a range between 0 and 1.
- I manualy did categorical encoding, without using One hot encoding

## **Feature Selection**

for feature selection i have used sparsity property of lasso regression , Lasso regression is a kind of regularisation and what do we mean by regularistaion , regurisation is a technique 
which is used to reduce overfitting by adding extra term during a training of our model. there is a paramter in lasso regresion called alpha and if we increase the value of alpha then 
it makes the coefficients of less important features to zero , so those features left which are important for prediction
and the second technique i used is Seaquential forward selection , In this method we simply train our model for the whole dataset and calculate the accuracy then iteratively we remove one feature and check accuracy again , so in this way we find the best subset or the best combination of features which are giving are giving our model best accuracy
and i renove that column which is common in the result of both of this feature selection techniques

## **MODEL Training**

I have used 3 Algorithms to train my Model
1. Logistic Reggression
2. DecesionTreeClassifer
3. RandomForestClassifier

Out of all the models That I trained and evaluated, I can conclude that the Random Forest Classifier is the best model for our dataset. The optimal parameters for this model are {'n_estimators': 150}. Here are its performance metrics:
- Accuracy Score: ~0.90

- Precision: 0.88

- Recall: 0.89

- F1 Score: 0.88

- ROC-AUC Score: 0.96

This model demonstrates high accuracy, a high ROC-AUC score, high precision, and high recall compared to other models, making it the best choice for our data.

![2024-06-26 20 08 58](https://github.com/AbhinavNautiyal123/Health-Insurance-cross-Prediction/assets/164336356/48c37eb0-b1f5-4354-b39c-22da058bbcbb)


## **Conclusion**

Initially, we checked our dataset for null values and duplicates. Since there were no null values or duplicates, no further treatment was required. We then performed feature engineering to create new features. Before data processing, we applied feature scaling techniques to normalize the data, ensuring that all features were on the same scale, making it easier for machine learning algorithms to process.

Through Exploratory Data Analysis (EDA), we categorized age into three groups: '20-40', '40-60', and '60+'. We also categorized the vintage into 'Short term', 'Mid term', and 'Long term'. To handle the imbalanced dataset, we used the SMOTE (Synthetic Minority Over-sampling Technique).

Our analysis revealed that younger customers (aged 20-40) are more interested in vehicle insurance. Additionally, customers with vehicles older than two years and those with damaged vehicles are more likely to be interested in vehicle insurance.

For feature selection, we used Lasso regression and sequential forward selection. We found that the feature 'Previously policy insured' was the most important, while 'Driving license' was the least important.

We then applied various machine learning algorithms to predict whether a customer would be interested in vehicle insurance. For logistic regression, we achieved an accuracy score of 83% after hyperparameter tuning. Using decision tree classifier and random forest classifier, we achieved accuracy scores of approximately 88% and 89%, respectively, after hyperparameter tuning.

Based on these results, we selected the random forest classifier as our best model, with an accuracy score of 89%.






