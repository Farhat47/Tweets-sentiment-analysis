#################
#### imports ####
#################
import pandas as pd
## Importing stopwords
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


########################
#### Help Functions ####
########################
def list_files(directory):
	allfiles = []	
	for dirname, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			allfiles.append(os.path.join(dirname,filename))
	return allfiles

def SentimentsCounter(sentiment):
	if(str(sentiment)=="['negative']"):
		global Negative_Count
		Negative_Count  +=1
	elif(str(sentiment)=="['positive']"):
		global Positive_Count
		Positive_Count +=1
	elif (str(sentiment)=="['neutral']"):
		global Neutral_Count
		Neutral_Count += 1

# ## Loading Data
train_data = pd.read_csv("x_y_train.csv")
test_data = pd.read_csv("x_test.csv")

## Training data
x_train_data = train_data["text"]
y_train_data = train_data["sentiment"]

## Testing data
x_test_data = test_data["text"]


########################
#### Stop Words ########
########################
stop = stopwords.words("english")
## Importing punctuations
punctuations = string.punctuation
## Adding Punctuations to our stop words
stop += punctuations



##########################
#### Train and Test Split ####
##########################

## Splitting in the given training data for our training and testing
x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train_data, y_train_data, 
                                                                            random_state = 0, test_size = 0.25)

##########################
#### Count Vectoriser ####
##########################
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words=stop, 
                             analyzer='word', max_df = 0.8, lowercase = True, use_idf = True, smooth_idf = True)

## Fit transform the training data
train_features = tf_idf_vec.fit_transform(x_train_train)

## Only transform the testing data according to the features which was fit using x_train
test_features = tf_idf_vec.transform(x_train_test)

######################################
######################################
#### Applying various classifiers ####
######################################
######################################

################################
####  Applying SVC #############
################################
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_features, y_train_train)

print("SVC Algorithm")
print(svc.score(test_features, y_train_test))


################################
#### Applying Random Forest ####
################################

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_features, y_train_train)

print("Random Forest Algorithm")
print(rf.score(test_features, y_train_test))

##########################################
#### Applying Multinomial Naive Bayes ####
##########################################

from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB(alpha=0.4)
bayes.fit(train_features, y_train_train)

print("Multinomial Naive Bayes")
print(bayes.score(test_features, y_train_test))


##########################################
#### Applying Multinomial Naive Bayes ####
##########################################
from sklearn.model_selection import GridSearchCV
clf = MultinomialNB()
grid = {"alpha" :[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}
abc = GridSearchCV(clf, grid) 
abc.fit(test_features, y_train_test)

print("Best Estimator")
print(abc.best_estimator_)

pickle.dump(bayes, open(r"C:\Users\moust\Google Drive\New folder\Twitter-Sentiment-Analysis-master\ML_Model_Sentiment_Analysis.pkl", "wb"))

# ## Prediction
#pred_features = tf_idf_vec.transform(x_test_data)
#y_pred = bayes.predict(pred_features)


#Prediction Test
#pred_features = tf_idf_vec.transform(["trade China and thanks"])
#result = bayes.predict(pred_features)
#print(result)


#################
#### Program ####
#################
directory=r'C:\Users\moust\Google Drive\New folder\Twitter-Sentiment-Analysis-master'

input_directory = directory + r'\Filtered Tweets'
allfiles=list_files(input_directory)
output_file = directory + r'\Tweets.txt'

###################
#### Variables ####
###################
Positive_Count= 0
Negative_Count=0
Neutral_Count=0


for tweetFile in allfiles:

	with open(tweetFile, 'r') as f:
		data = f.read()
		# jump to the beginning of the file:
		f.seek(0)

		#Remove Directory
		Txt_Name=tweetFile.replace(input_directory,'')


		#Get Date from Tweet
		Date=f.readline()
		print("---------------Start-----------------")
		print(Date)
		print(Txt_Name)
	
		pred_features = tf_idf_vec.transform([data])
		result = bayes.predict(pred_features)

		SentimentsCounter(result)

		print(result)
		print("-------------End-------------------" + '\n')

					#Write Data to Text File
		outfile = open(output_file, mode = 'a+') 
		outfile.write(Txt_Name+ " ")
		outfile.write(str(result)+ " ")
		outfile.write(str(Date)+ " ")
		outfile.close()
	f.close()

# Make a fake dataset:
plt.title("Trump Tweet Sentiment",fontsize= 16)
height = [Positive_Count, Negative_Count, Neutral_Count]
bars = ('Positive'+' ' + str(Positive_Count),'Negative'+ ' ' +str(Negative_Count),'Neutral'+ ' ' +str(Neutral_Count))
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 # Create names on the x-axis
plt.xticks(y_pos, bars)
# Show graphic
plt.show()

#import numpy as np
#np.savetxt("output.csv", y_pred, fmt='%s')