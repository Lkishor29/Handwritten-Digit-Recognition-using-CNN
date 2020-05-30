#model Accuracy
from keras.models import load_model
model=load_model('hand_wr.h5')
values=model.evaluate(x_test,y_test,verbose=1)
print('Test Loss: ',values[0])
print('Test Accuracy: ',values[1])