from joblib import dump, load
import numpy as np
model = load('Predict_price.joblib')
print("      ******Enter the following features******")
print(" 1. CRIM      per capita crime rate by town \n 2. ZN        proportion of residential land zoned for lots over 25000 sq. ft. \n 3. INDUS     proportion of non-retail business acres per town \n 4. CHAS      Charles River dummy variable(=1 if tract bounds river 0 otherwise) \n 5. NOX       nitric oxides concentration(parts per 10 million) \n 6. RM        average number of rooms per dwelling  \n 7. AGE       proportion of owner-occupied units built prior to 1940 8. DIS       weighted distances to five Boston employment centres \n 9. RAD       index of accessibility to radial highways \n 10. TAX      full-value property-tax rate per $10, 000  \n 11. PTRATIO  pupil-teacher ratio by town \n 12. B        1000(Bk - 0.63) ^ 2 where Bk is the proportion of blacks by town  \n 13. LSTAT    lower status of the population")

features_arr = list(map(int, input().split()))
features_arr = np.asarray(features_arr)
features_arr = np.reshape(features_arr, (-1, 13))

print(model.predict(features_arr))

# features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
#                       -0.24141041, -1.31238772,  2.61111401, -1.0016859, -0.5778192,
#                       -0.97491834,  0.41164221, -0.86091034]])

# print(model.predict(features))
