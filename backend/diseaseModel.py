import pandas as pd
import numpy as np
import sklearn
df = pd.read_csv('C:\\Users\\arpit.rai\\Desktop\\Repos\\med-assist\\backend\\disease-symptom\\DiseaseAndSymptoms.csv')
care = pd.read_csv('C:\\Users\\arpit.rai\\Desktop\\Repos\\med-assist\\backend\\disease-symptom\\Disease precaution.csv')

data2=df.drop_duplicates().reset_index(drop=True)

print(df.head)
import math
all_symptoms=set()
for i in data2.columns:
    if i=='Disease' or pd.isna(i) or not isinstance(i,str):
        continue
    all_symptoms=all_symptoms.union(set(data2[i].str.strip().unique()))

all_symptoms = {x for x in all_symptoms if not isinstance(x, float) or not math.isnan(x)}

all_symptoms_ordered = sorted(list(all_symptoms))

dict={}
for i in range(len(all_symptoms_ordered)):
    dict[list(all_symptoms_ordered)[i]]=i


zero_data = np.zeros((304, 131))

# Create the DataFrame from empty array
new_df = pd.DataFrame(zero_data)
new_df.columns = dict.keys()

new_df['Disease'] = data2['Disease']

#Fill the dataframe based on symptoms
for ind,item in data2.iterrows():
    for i,j in enumerate(item):
        if(i==0):
            continue
        if(pd.isna(j)):
            continue
        new_df.iloc[ind,dict[j.strip()]]=1

P = new_df[["Disease"]]
X = new_df.drop(["Disease"],axis=1)
Y = new_df.drop(["Disease"],axis=1)
from sklearn.preprocessing import LabelEncoder
ohe = LabelEncoder()
P_enc = ohe.fit_transform(P)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,P_enc,test_size=0.2,random_state=42)

from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Flatten, BatchNormalization
import keras

import keras
import tensorflow as tf
class l12Regularizer(keras.regularizers.Regularizer):
    """
    Custom regularizer that applies a penalty based on the L1 and L2 norms of the weights.
    Inherits from `keras.regularizers.Regularizer`.
    """

    def __init__(self, l1=0.06):
        # Store the coefficients for serialization
        self.l1 = l1

    def __call__(self, weights):
        """
        Calculates the regularization penalty for the given weights.
        This method is called by the Keras model during training.

        Args:
            weights (tf.Tensor): The weights of the layer to regularize.

        Returns:
            tf.Tensor: The calculated regularization penalty.
        """
        # Calculate the L1 penalty ### *tf.stop_gradient(tf.convert_to_tensor(1.0/(1.0+0.01*correlation**2),dtype=tf.float32)
        l1_penalty = self.l1 * tf.reduce_sum(tf.sqrt(tf.abs(weights)+1e-6))

        return l1_penalty

    def get_config(self):
        """
        Returns the configuration of the regularizer.
        This is necessary to serialize the regularizer when saving the model.

        Returns:
            dict: A dictionary containing the regularizer's configuration.
        """
        return {'l1': float(self.l1)}

def robustLasso():
    inputs = Input(shape=(131,))
    x = keras.layers.Dense(130,activation='relu',kernel_regularizer=l12Regularizer(0.005))(inputs)
    x = keras.layers.Dense(41,activation='softmax',kernel_regularizer=l12Regularizer(0.005))(x) #l12Regularizer(0.04)
    return keras.Model(inputs=inputs,outputs=x)
    
learning_rate = 0.01
def trainM(model,path,epochs=40,learning_rate=0.1):
  model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),#learning_rate=learning_rate,momentum=0.9,clipvalue=0.9), #,momentum=0.9#,clipvalue=0.01,global_clipnorm=0.01
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])
  model.fit(xtrain,  ytrain, validation_data=(xtest,ytest), epochs=epochs, batch_size=16, verbose=1)
  model.save_weights('symptom_evaluator2')

aLasso = robustLasso()
trainM(aLasso,'test_features',epochs=250,learning_rate=0.1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Assuming you have already trained your model and made predictions
y_pred_probs = aLasso.predict(xtest)
y_pred_original = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded labels back to original labels
y_test_original = ohe.inverse_transform(ytest)
#y_pred_original = ohe.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(ytest, y_pred_original)
precision = precision_score(ytest, y_pred_original, average='weighted')
recall = recall_score(ytest, y_pred_original, average='weighted')
f1 = f1_score(ytest, y_pred_original, average='weighted')

# Print metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')