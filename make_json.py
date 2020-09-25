import csv
import json

  
# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_json(csvFilePath, jsonFilePath):

    # create a dictionary 
    data = {}
    row_vals = []
    
    # Open a csv reader called DictReader 
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)  
        # Convert each row into a dictionary  
        # and add it to data 
        for rows in csvReader:
            row_vals.append(rows)
    data = {"intents": row_vals}
    
    # Open a json writer, and use the json.dumps()  
    # function to dump data 
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

# name file paths
csvFilePath = 'intents_new.csv'
jsonFilePath = 'intents_new.json'

# Call the make_json function 
make_json(csvFilePath, jsonFilePath)     
# name file paths
csvFilePath = 'intents_new.csv'
jsonFilePath = 'intents_new.json'
  
# Call the make_json function 
make_json(csvFilePath, jsonFilePath)
