import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

df = pd.read_csv('Cleaned_DS_Jobs.csv')

input = df[['Job Title','Rating','Company Name', 'Location', 'Size', 'Type of ownership',
       'Industry', 'Sector', 'Revenue', 'python','excel', 'hadoop', 'spark', 'aws', 'tableau', 'big_data']]
       
output = df[['min_salary','max_salary','avg_salary']]

from sklearn.preprocessing import LabelEncoder

# Encoding

leJobTitle = LabelEncoder()
input['Job Title'] = leJobTitle.fit_transform(input['Job Title'])

leCompanyName = LabelEncoder()
input['Company Name'] = leCompanyName.fit_transform(input['Company Name'])

leLocation = LabelEncoder()
input['Location'] = leLocation.fit_transform(input['Location'])

leSize= LabelEncoder()
input['Size'] = leSize.fit_transform(input['Size'])

leTypeeOfOwnership = LabelEncoder()
input['Type of ownership'] = leTypeeOfOwnership.fit_transform(input['Type of ownership'])

leIndustry = LabelEncoder()
input['Industry'] = leIndustry.fit_transform(input['Industry'])

leSector = LabelEncoder()
input['Sector'] = leSector.fit_transform(input['Sector'])

leRevenue = LabelEncoder()
input['Revenue'] = leRevenue.fit_transform(input['Revenue'])

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(input,output,test_size=0.15)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(x_train,y_train)

# yPred = model.predict(x_test)

# min_salary = yPred[0]
# max_salary = yPred[1]
# avg_salary = yPred[2]


# so I would be outputting the average salary 
data = {"model" :  model , "leJobTitle" : leJobTitle , "leCompanyName" : leCompanyName , "leIndustry" :leIndustry , "leLocation" : leLocation , "leRevenue" : leRevenue , "leSector":leSector , "leSize" : leSize , "leTypeeOfOwnership" : leTypeeOfOwnership}

with open('model_saved.pkl', 'wb') as file:
    pickle.dump(data, file)
    
with open('model_saved.pkl', 'rb') as file:
    data = pickle.load(file)