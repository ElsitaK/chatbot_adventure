import csv
import json
#import pdb  
  
# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_json(csvFilePath, jsonFilePath): 
		      
    # create a dictionary 
    data = {} 
      
    # Open a csv reader called DictReader 
    with open(csvFilePath, encoding='utf-8') as csvf: 
        csvReader = csv.DictReader(csvf) 
          
        # Convert each row into a dictionary  
        # and add it to data 
        for rows in csvReader:
            # row value in column named 'tag' is the primary key 
            tag = rows['tag'] 
            
            for key, value in data.items():
                #print(key, str("HELLO"), value) #if value = key:
                if key == tag:
                    data[key].update(rows)			
                    print(value) #append somehow
                else:
                    next
            data[tag] = rows             	    
 		
    # Open a json writer, and use the json.dumps()  
    # function to dump data 
    #pdb.set_trace()
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonf.write(json.dumps(data, indent=4)) 
          
# name file paths
csvFilePath = 'intents_new.csv'
jsonFilePath = 'intents_new.json'
  
# Call the make_json function 
make_json(csvFilePath, jsonFilePath)
