#! /usr/bin/python3

import csv
import json
import sys
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

JUNK_DESCRIPTIONS = [
        {"plot", "wrap"},
        {"plot", "unknown"},
        {"add", "plot"},
        {"keep", "wrap"},
        {"film", "set"},
        {"film", "introduction"}
        ]

MISSING_GENRE = [
        "",
        "unknown"
        ]

def is_junk(description):
    for entry in JUNK_DESCRIPTIONS:
        if entry.issubset(description):
            return True
    return False


def tokenize(string): #adjust to taste

    string = string.lower() #lowercase NOTE THAT CHANGING THIS MAY AFFECT OUR ABILITY TO DETECT JUNK DESCRIPTIONS

    string = re.sub(r'[^\w\s]', '', string) #remove punctuation

    tokens = word_tokenize(string) #tokenize

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words] #no more stop words

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens] #lemmatize - NOTE THAT CHANGING THIS MAY AFFECT OUR ABILITY TO DETECT JUNK DESCRIPTIONS

    tokens = [token for token in tokens if not token.isdigit()] #remove pure numbers

    return tokens


def preprocess(csv_file_path):
    csv_reader = csv.DictReader(csv_file)
    dataset = {
            "data": [
                #{
                #"description": ["This", "is", "the", "description"]
                #"genre": ["These", "are", "the", "genres"]
                #},
                ],
            "genres": [] #List of genres as strings
            }
    #known_genres = set()
    known_genres = {}
    for row in csv_reader:
        #description = row["Description"] imdb desc
        description = row["Plot"] #wikipedia desc
        genres = row["Genre"]

        #get genres
        genres = [genre.strip().lower() for genre in genres.split(',')]

        #convert genres to standard forms using genre_mapping.json
        genre_mapping = {}
        with open('genre_mapping.json', 'r') as file:
            genre_mapping = json.load(file)

        clean_genre_list = set()
        for genre in genres:
            standard_genre = genre_mapping.get(genre, [""]) #get the list of genres that a genre maps o
            clean_genre_list.update(standard_genre) # add to set of genres
        genres = list(clean_genre_list)

        #we don't want movies without a genre
        if all(genre in MISSING_GENRE for genre in genres):
                continue

        #clean empty strings
        genres = [genre for genre in genres if genre.strip()]

        description = tokenize(description)

        #Remove junk descriptions like "plot unknown"
        if is_junk(description):
            continue

        dataset["data"].append({
            "description": description,
            "genre": genres
            })

        for genre in genres:
            known_genres[genre] = known_genres.get(genre,0) + 1
        #known_genres.update(genres)

    #dataset["genres"] = list(known_genres)
    dataset["genres"] = known_genres
    return dataset

if __name__ == "__main__":
    dataset = None
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess.py <input> <output>")
    else:
        csv_file_path = sys.argv[1]
        json_file_path = sys.argv[2]
        try:
            #preprocess data
            with open(csv_file_path, 'r') as csv_file:
                dataset = preprocess(csv_file)
            #print(dataset)

            #dump output as a big json
            with open(json_file_path, 'w') as json_file:
                json.dump(dataset, json_file, indent=4)

        except Exception as e:
            print(f"Error: {e}")
