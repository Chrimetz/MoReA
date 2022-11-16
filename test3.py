import json
import sys

description = json.load(open("models/"+"RandomForestRegression"+".json", 'rb')) 

def validate_feature(feature, index):
    if not "name" in feature:
        raise KeyError("Name not found in input feature " + str(index))
    elif not "type" in feature:
       raise KeyError("Type not found in input feature " + str(index))

    if feature["type"] != "list" and "shape" not in feature:
        raise KeyError("Shape not found in input feature " + str(index))
    elif feature["type"] != "list" and "shape" in feature:
        if isinstance(feature["shape"], list):
            if not len(feature["shape"]) > 0:
                raise(ValueError("Shape needs to have at least one dimension in input feature " + str(index)))
        else:
            raise(ValueError("Shape is not a list in input feature " + str(index)))
    
    if feature["type"] == "list":
        if not "features" in feature:
            raise KeyError("Features not found in input feature " + str(index))

        j = -1
        for f in feature["features"]:
            j = j + 1
            try:
                validate_feature(f, j)
            except Exception as e:
                raise e

def validate_description(description):
    for s in ["name", "details", "outputs", "input_features"]:
        if not s in description:
            raise KeyError(s + " not found")

    i = -1
    for feature in description['input_features']:
        i = i + 1
        try:
            valid = validate_feature(feature, i)
        except Exception as e:
            raise e
        
def is_description_valid(description) -> bool:
    try:
        validate_description(description)
        return True
    except Exception as e:
        print(e)
        return False

print(is_description_valid(description))

