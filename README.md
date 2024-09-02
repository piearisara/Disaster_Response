# Disaster Response Pipeline Project

## To run ETL pipeline
The process_data file takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite.
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

## To train ML model. 
The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
`Note that: This will take around 10 min.`
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```


## Run webapp
The index page includes visualizations using data from the SQLite database.And, when a user inputs a message into the app, the app returns classification results for all 36 categories.
```bash
cd app 
python run.py
```
