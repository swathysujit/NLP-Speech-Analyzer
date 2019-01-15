# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:38:53 2018

@author: Swathy Sujit
"""

#Speech Sentiment Analyzer
#This analyzer, first coverts speech to text, analyzes, does prediction of sentiment and 


#!pip install nltk
import os
os.chdir("C:\\Users\\Swathy Sujit\\Documents\\NLP Online Course\\Datasets\\sorted_data_acl\\electronics")

#Library to save model
#import pickle

import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
#from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 

#Turns words into their 'base' forms or 'root' forms
wordnet_lemmatizer = WordNetLemmatizer()

import re
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_more=set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews=BeautifulSoup(open('positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')
#positive_reviews
negative_reviews = BeautifulSoup(open('negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')
#negative reviews
np.random.shuffle(positive_reviews)

positive_reviews = positive_reviews[:len(negative_reviews)]
#len(positive_reviews)

#Takes a sentence(one review at a time) and returns a toenized and relevant words only
def my_tokenizer(s):
    s=s.strip()
    s=s.lower()
    tokens = nltk.tokenize.word_tokenize(s)    
    #Automatically throw out short words since they will mostly not be useful(in, of, the..)
    tokens = [t for t in tokens if len(t.strip())>2]
    #replace by root or base words like jumping becomes jump, dogs becomes dog, etc
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    #Remove stopwords
    tokens = [t for t in tokens if t not in stopwords_more]
    #Remove default stopwords in nltk package
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]
    #Remove punctutations
    tokens = [re.sub('[^a-zA-Z]',' ',t) for t in tokens]
    return tokens        
#word_index_map is to store each new word in the dicstionary and give it an index
#Going through both positive and negative reviews and adding all unique words into dictionary word_index_map

word_index_map = {}
current_index = 0

#Saving tokenized reviews for later since we have to do two passes
positive_tokenized = []
#postive_tokenized = my_tokenizer(positive_reviews)
negative_tokenized = []
#negative_tokenized = my_tokenizer(negative_reviews)

#Need to download these specific nltk packages for certain operations
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download()

#==================================================================================
#Word Frequency Dictionary

word_freq = {}

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    for token in tokens:
        if token not in word_freq:
            word_freq[token] = 1
        else:
            word_freq[token]+=1

#Using same loop as above for negative reviews, to see if thre are more distinct words to be added into the dictionary
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    for token in tokens:
        if token not in word_freq:
               word_freq[token] = 1
        else:
            word_freq[token]+=1

#import pandas as pd
word_frequency = pd.Series(word_freq)
writer = pd.ExcelWriter("Word_Frequency.xlsx")
word_frequency.to_excel(writer,"Sheet 1")
writer.save()
#==================================================================================

#Creating vocabulary dictionary - only taking words that appear more than 3 times
#Looping and adding elements to dictionary with an index
for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            if word_freq[token]>3:
                word_index_map[token] = current_index
                current_index += 1
            
#Using same loop as above for negative reviews, to see if thre are more distinct words to be added into the dictionary
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            if word_freq[token]>3:
                word_index_map[token] = current_index
                current_index += 1
                
#len(word_index_map)
                
#Taking each token and creating a data array which is a bunch of numbers, we discussed about word proportions using raw counts(refer notes)
#Because we want to shuffle train and test sets again, we will put token vector and label in same
def tokens_to_vector(tokens,label):
    x = np.zeros(len(word_index_map)+1) #+!, the last element of the vector is for the label
    for t in tokens:
        if t in word_index_map:
            i = word_index_map[t]
            x[i]+=1
    x = x/x.sum()
    x[-1] = label
    return x

#tokens_to_vector(['toy','bug','defective'],0)
#len(word_index_map)+1

N = len(positive_tokenized)+len(negative_tokenized)

#Creating a matrix data, of negative+positive reviews count as number of rows, 
#and word_index_map count+1(for label) as no of columns
data = np.zeros((N, len(word_index_map)+1))
i=0

#Looping in positive and negative review tokens and appending each feature vector to the data matrix along with label flag as the right most element
for tokens in positive_tokenized:
    xy=tokens_to_vector(tokens,1)
    data[i,:] = xy
    i+=1

for tokens in negative_tokenized:
    xy=tokens_to_vector(tokens,0)
    data[i,:] = xy
    i+=1

#Therefore we have now the training dataset, which has the words raw count proportions and the label(0 or 1)
#We can now split them into training and test sets and perform classification analysis
np.random.shuffle(data)

X = data[:,:-1]
#Last column is labels
Y=data[:,-1]
#Training and test splitting - last 200 rows will be test set
Xtrain = X[:-200,:]
Ytrain = Y[:-200,]
Xtest = X[-200:,]
Ytest = Y[-200:,]

# =============================================================================
# training_set = pd.DataFrame(Xtrain,Ytrain)
# writer = pd.ExcelWriter("Word_Frequency.xlsx")
# training_set.to_excel(writer,"Sheet 1")
# writer.save()
# =============================================================================

#Running logistic regression classification model
#from sklearn.linear_model import LogisticRegression
#model1 = LogisticRegression()
#model1.fit(Xtrain,Ytrain)
#model1.score(Xtrain,Ytrain)

#model = RandomForestClassifier()
model = MLPClassifier(verbose = True,hidden_layer_sizes=(1024,1024),random_state=5,batch_size=40,learning_rate_init=0.0001,max_iter=20)
model.fit(Xtrain,Ytrain)
model.score(Xtrain,Ytrain)

#Running random forest classification model for performance comparison
print("ANN Classification rate:",model.score(Xtest,Ytest))
#Classification rate will vary each tim

#Saving model
#pickle.dump(model, open('Speech_Analyzer_Classifier.sav', 'wb'))

#Loading Model
#model = pickle.load(open('Speech_Analyzer_Classifier.sav', 'rb'))

# =============================================================================
# #Classification using Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# model2 = GaussianNB()
# model2.fit(Xtrain,Ytrain)
# model2.score(Xtrain,Ytrain)
# print("Naive Bayes Classification rate:",model2.score(Xtest,Ytest))
# 
# #Classification using KNN
# from sklearn.neighbors import KNeighborsClassifier
# model3 = KNeighborsClassifier()
# model3.fit(Xtrain,Ytrain)
# model3.score(Xtrain,Ytrain)
# print("KNN Classification rate:",model3.score(Xtest,Ytest))
# =============================================================================

#We have considered logistic regression here as the results are interpretable and we can look at the weights(coeff) of the words
#Now we can look at the weights the words have to see if the sentiment is postive or negative
#We'll look at everything thats far away from 0, not all

#Taking speech input and converting to text
def audio_to_vector(tokens):
    x = np.zeros(len(word_index_map)) #+!, the last element of the vector is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i]+=1
    x = x/x.sum()
    return x

#pip install gTTS
#pip install pygame
from gtts import gTTS #Google API for text to speech conversion
from pygame import mixer #library to play audio file in python

#Save this question input and play before asking user for opinion regarding say restaurant or any other
q = "Please tell us your opinion of the restaurant"
speech = gTTS(text=q, lang='en',slow=False)
speech.save("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/q.mp3")

def speech_analyzer():
    import speech_recognition as sr
    r = sr.Recognizer()    
    
    with sr.Microphone() as source: 
        mixer.init()
        mixer.music.load("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/q.mp3")
        mixer.music.play()    
        print("Speak now:")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)       
            tokens = my_tokenizer(text)
            audio_vector = audio_to_vector(tokens)
            print('You said: {}'.format(text))
            if model.predict([audio_vector]) ==0:
                res = "Sorry to hear that you are unhappy. We will do our best to satisfy you next time"
                print(res)
                response = gTTS(text=res, lang='en',slow=False)
                response.save("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/response.mp3")
                mixer.music.load("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/response.mp3")
                mixer.music.play()
                exit                    
            else:
                res = "Glad to know that you are happy with us! Thank you for the feedback"
                print(res)
                response = gTTS(text=res, lang='en',slow=False)
                response.save("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/response.mp3")
                mixer.music.load("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/response.mp3")
                mixer.music.play()
                exit
        except:
            excep = 'Sorry could not register your feedback! Please try again' 
            print(excep)
            error = gTTS(text=excep, lang='en',slow=False)
            error.save("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/error.mp3")
            mixer.music.load("C:/Users/Swathy Sujit/Documents/NLP Online Course/Speech-to-text-to-speech/error.mp3")
            mixer.music.play()
            exit

#Creating GUI using tkinter
##Tkinter
# initialize the window toolkit along with the two image panels
from tkinter import *
root = Tk()
root.title("Customer Review Assistant")
root.geometry("390x150")
root['bg'] = "white"
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn1 = Button(root, text="Click to Provide Review", command = speech_analyzer,bg = "dark green",fg="white")
btn1.place(relx = 0.05, rely= 0.3)
btn2 = Button(root, text="  Try Again  ", command=speech_analyzer,bg = "dark blue",fg="white")
btn2.place(relx = 0.55, rely= 0.3)
btn3=Button(root, text="  Exit  ", command=root.destroy,bg="maroon",fg="white")
btn3.place(relx = 0.45, rely= 0.6)

# kick off the GUI
root.mainloop()
