#-------------------------------------------------------------------------
# AUTHOR: Saul Barajas
# FILENAME: naive_bayes.py
# SPECIFICATION: using the naive bayes model to see class probabilities of instances
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
db =[]
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
outlook = {"Sunny":1,
           "Overcast":2,
           "Rain":3}

temperature ={"Hot":1,
           "Mild":2,
           "Cool":3}

humidity ={"High":1,
           "Normal":2}

wind ={"Strong":1,
           "Weak":2}

tennis = {"Yes":1,
          "No":2}
X=[]
for data in db:
    X.append([outlook[data[1]], temperature[data[2]], humidity[data[3]], wind[data[4]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y=[]
for data in db:
    Y.append(tennis[data[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTesting=[]
dbTestingTransformed = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTesting.append (row)

for data in dbTesting:
    dbTestingTransformed.append([outlook[data[1]], temperature[data[2]], humidity[data[3]], wind[data[4]]])

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i,test in enumerate(dbTestingTransformed):
    print(dbTesting[i])
    print(clf.predict_proba([test])[0])



