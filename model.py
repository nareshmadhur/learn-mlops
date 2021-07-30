import pickle
from sklearn.model_selection import train_test_split
import sklearn

with open('IDEA_NS.pkl', 'rb') as file:
    df = pickle.load(file)

X = df.date
Y = df.close

xtr, xts, ytr, yts = train_test_split(X, Y, test_size=0.2)
