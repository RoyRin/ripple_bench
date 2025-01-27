# load 
import pickle

with open(r"SVDS_0.pkl", "rb") as input_file:
    e = pickle.load(input_file)
    
print(e)
#print(e.shape)