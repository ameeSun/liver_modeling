import pickle
import numpy as np

#input age
while True:
  try:
    age = float(input("What is the age of the patient?\n"))
    if age>0 and age<130:
      break;
    else:
      print("Provide a valid age...")
  except ValueError:
    print("Provide an integer value...")
    continue

#input gender
while True:
  try:
    gender = input("What is the biological sex of the patient\n").lower()
    if gender == "male":
        gender = int(0);
        break
    elif gender == "female":
      gender = int(1);
      break;
    else:
      print("Gender should be either Male or Female")
  except:
    continue
#input total bilirubin
while True:
  try:
    total_bili = float(input("What is the total bilirubin?\n"))
    break
  except ValueError:
    print("Provide a decimal value...")
    continue
#input direct bilirubin
while True:
  try:
    dir_bili = float(input("What is the direct bilirubin?\n"))
    break
  except ValueError:
    print("Provide a decimal value...")
    continue
# input ALP
while True:
    try:
        ALP = float(input("What is the alkaline phosphatase (ALP) level?\n"))
        break
    except ValueError:
        print("Provide an integer value...")
        continue
# input ALT
while True:
    try:
        ALT = float(input("What is the alanine aminotransferease (ALT) level?\n"))
        break
    except ValueError:
        print("Provide an integer value...")
        continue
# input AST
while True:
    try:
        AST = float(input("What is the alanine aminotransferease (ALT) level?\n"))
        break
    except ValueError:
        print("Provide an integer value...")
        continue
# input total proteins
while True:
    try:
        total_protein = float(input("What is the total protein level?\n"))
        break
    except ValueError:
        print("Provide a decimal value...")
        continue
# input albumin
while True:
    try:
        albumin = float(input("What is the albumin level?\n"))
        break
    except ValueError:
        print("Provide a decimal value...")
        continue
# input A/G ratio
while True:
    try:
        AG_ratio = float(input("What is the albumin/globulin (A/G) ratio?\n"))
        break
    except ValueError:
        print("Provide a decimal value...")
        continue

input_predict = [age,gender,total_bili,dir_bili,ALP,ALT,AST,total_protein,albumin,AG_ratio]

filename = 'rf_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
outcome = loaded_model.predict(np.array(input_predict).reshape((1, -1)))[0]
if outcome == 1:
    print("The patient has liver disease")
elif outcome == 2:
    print("The patient does not have liver disease")