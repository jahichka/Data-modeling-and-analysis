import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

#definiranje funkcije koja nam iskljucuje "outlier" vrijednosti
def exclude_outliers(data, column, threshold=2):
    z_scores = stats.zscore(data[column])
    filtered_indices = np.where(np.abs(z_scores) <= threshold)
    filtered_data = data.iloc[filtered_indices]
    return filtered_data

#ucitavanje CSV file-a
data=pd.read_csv("Salary_dataset.csv")

#informacije o dobijenim podacima
print(data.shape)
print(data.info())
print(data.describe())

#racunanje srednje vrijednosti i standardne devijacije
mean_exp=data["YearsExperience"].mean()
mean_sal=data["Salary"].mean()

#provjera da li postoje NaN vrijednosti
#ukoliko postoji, zamjena NaN vrijednosti sa srednjom vrijednoscu
data["YearsExperience"].fillna(mean_exp, inplace=True)
data["Salary"].fillna(mean_exp, inplace=True)

#odstranjivanje "outlier" vrijednosti
filtered_data = exclude_outliers(data, "YearsExperience")

#prikazivanje scatter plota da provjerimo da li je linearna regresija
#model koji je pogodan
sns.relplot(x="YearsExperience", y="Salary", data=data, height=3.8, aspect=1.8, kind="scatter")
sns.set_style("dark")
plt.show()

#razdvajanje dataset-a na dio za testiranje i dio za treniranje 
X = filtered_data.iloc[:,:-1].values #izdvajanje godina iskustva iz dataset-a
y = filtered_data.iloc[:,1].values #izdvajanje plate iz dataset-a

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#fittanje
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#koeficijent
print(regressor.coef_)

plt.figure()
sns.regplot(x='YearsExperience', y='Salary', data=data, scatter_kws={'s':100, 'facecolor':'red'})

#testiranje naucenog seta
y_pred = regressor.predict(X_test)
print(y_pred)

#poredjenje vrijednosti iz seta za testiranje i pretpostavljenih vrijednosti po modelu
comparison_df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(comparison_df)

#razlika - greska
residuals = y_test - y_pred
print(residuals)

#prikaz na scatterplotu
plt.figure()
sns.scatterplot(x=y_test, y = y_pred, s=140)
plt.xlabel("Test data")
plt.ylabel("Predictions")
plt.show()

#Evaluacija modela

#Srednja apsolutna greska
print("MAE: ", mean_absolute_error(y_test,y_pred))

#srednja kvadratna greska
print("MSE: ",mean_squared_error(y_test,y_pred))

#korijen srednje kvadratne greske
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# R^2 - mjera fitanja izmedju predikcije i stvarne vrijednosti {0,1}
r2 = r2_score(y_test,y_pred)
print("R^2: ",r2)

