#!/usr/bin/env python
# coding: utf-8

# In[205]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil
import cv2
import matplotlib.image as mpimg
import seaborn as sns
# Enable inline plots for Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[3]:


# Rename files to ensure a consistent naming convention
folder = 'brain_tumor_dataset/yes/'
count = 1
for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "Y_" + str(count) + ".jpg"
    os.rename(source, destination)
    count += 1
print("All files in yes dir are renamed")


# In[5]:


# Rename files to ensure a consistent naming convention
folder = 'brain_tumor_dataset/no/'
count = 1
for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "N_" + str(count) + ".jpg"
    os.rename(source, destination)
    count += 1
print("All files in no dir are renamed")


# # EDA - Exploratory Data Analysis

# Plot

# In[203]:


listyes = os.listdir("brain_tumor_dataset/yes/")
number_files_yes = len(listyes)
print(number_files_yes)

listno = os.listdir("brain_tumor_dataset/no/")
number_files_no = len(listno)
print(number_files_no)


# In[18]:


# Create a dictionary with the number of tumorous and non-tumorous images
data = {'tumerous': number_files_yes, 'non-tumerous': number_files_no}

typex = data.keys()
values = data.values()

# Plot the dataset distribution to visualise imbalance
fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="red")
plt.xlabel("Data")
plt.ylabel("No. of Brain Tumor Images")
plt.title("Count of Brain Tumor Images")
plt.show()


# # Data Augmentation
# Dataset: 150(67%), 73(33%)

# In[18]:


import tensorflow as tf
# Import TensorFlow/Keras modules for image augmentation and deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
# Load the VGG19 model as the base for transfer learning
from tensorflow.keras.applications.vgg19 import VGG19
# Optimizers for model training
from tensorflow.keras.optimizers import Adam
# Callbacks for saving model, early stopping, and learning rate adjustments
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[42]:


def augmented_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(rotation_range=10,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       shear_range=0.1,
                       brightness_range=(0.3, 1.0),
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='nearest')
    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir + '/' + filename)
        image = image.reshape((1,) + image.shape)
        save_prefix = 'aug_' + filename[:-4]
        i = 0
        for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = save_to_dir, save_prefix = save_prefix, save_format = "jpg"):
            i += 1
            if i > n_generated_samples:
                break


# In[108]:


yes_path = 'brain_tumor_dataset/yes'
no_path = 'brain_tumor_dataset/no'

augmented_data_path = 'augmented_data/'

augmented_data(file_dir = yes_path, n_generated_samples=6, save_to_dir=augmented_data_path+'yes')
augmented_data(file_dir = no_path, n_generated_samples=13, save_to_dir=augmented_data_path+'no')


# In[109]:


# Function to output summary of dataset
def data_summary(main_path):
    yes_path = 'augmented_data/yes/'
    no_path = 'augmented_data/no/'

    n_pos = len(os.listdir(yes_path))
    n_neg = len(os.listdir(no_path))

    n = (n_pos + n_neg)

    pos_per = (n_pos*100)/n
    neg_per = (n_neg*100)/n

    print(f"Number of samples: {n}")
    print(f"{n_pos}: Number of positive sample in percentage: {pos_per}%")
    print(f"{n_neg}: Number of negative sample in percentage: {neg_per}%")


# In[349]:


data_summary(augmented_data_path)


# In[260]:


listyes = os.listdir("augmented_data/yes/")
number_files_yes = len(listyes)
print(number_files_yes)

listno = os.listdir("augmented_data/no/")
number_files_no = len(listno)
print(number_files_no)


# In[112]:


# Create a dictionary with the number of tumorous and non-tumorous images
data = {'tumerous': number_files_yes, 'non-tumerous': number_files_no}

typex = data.keys()
values = data.values()
# Plot the dataset distribution to visualise new balanced augmented dataset
fig = plt.figure(figsize=(5,7))
plt.bar(typex, values, color="red")
plt.xlabel("Data")
plt.ylabel("No. of Brain Tumor Images")
plt.title("Count of Brain Tumor Images")
plt.show()


# # Data Preprocessing

# In[114]:


import imutils
def crop_brain_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres =cv2.erode(thres, None, iterations = 2)
    thres = cv2.dilate(thres, None, iterations = 2)
    
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return image
        
    c = max(cnts, key = cv2.contourArea)
    
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBot = tuple(c[c[:,:,1].argmax()][0])
    
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]] 

    return new_image


# In[122]:


folder1 = 'augmented_data/no/'
folder2 = 'augmented_data/yes/'
valid_extensions = (".jpg")

for filename in os.listdir(folder1):
    img = cv2.imread(folder1 + filename)
    # Ignore non-image files like .DS_Store
    if not filename.lower().endswith(valid_extensions):
        print(f"⚠️ Skipping Non-Image File: {img_path}")
        continue
    img = crop_brain_tumor(img)
    cv2.imwrite(folder1 + filename, img)
    
for filename in os.listdir(folder2):
    img = cv2.imread(folder2 + filename)
    # Ignore non-image files like .DS_Store
    if not filename.lower().endswith(valid_extensions):
        print(f"⚠️ Skipping Non-Image File: {img_path}")
        continue
    img = crop_brain_tumor(img)
    cv2.imwrite(folder2 + filename, img)


# In[128]:


from sklearn.utils import shuffle

valid_extensions = (".jpg")

def load_data(dir_list, image_size):
    X=[]
    y=[]
    
    image_width, image_height=image_size
    
    for directory in dir_list:
        for filename in os.listdir(directory):
            if not filename.lower().endswith(valid_extensions):
                print(f"⚠️ Skipping Non-Image File: {img_path}")
                continue
            image = cv2.imread(directory + '/' + filename)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation = cv2.INTER_CUBIC)
            image = image/255.00
            X.append(image)
            if directory[-3:] == "yes":
                y.append(1)
            else:
                y.append(0)
    X=np.array(X)
    y=np.array(y)
    
    X,y = shuffle(X,y)
    print(f"Number of example is : {len(X)}")
    print(f"X SHAPE is : {X.shape}")
    print(f"y SHAPE is : {y.shape}")
    return X,y


# In[ ]:


augmented_path = 'augmented_data/'
augmeneted_yes = augmented_path + 'yes'
augmeneted_no = augmented_path + 'no'

IMAGE_WIDTH, IMAGE_HEIGHT = (240,240)

X,y = load_data([augmeneted_yes, augmeneted_no], (IMAGE_WIDTH, IMAGE_HEIGHT))


# # Data splitting
# 80% 10% 10%

# In[321]:


original_dataset_tumorours = os.path.join('augmented_data','yes/')
original_dataset_nontumorours = os.path.join('augmented_data','no/')


# Tumorous

# In[324]:


files = [f for f in os.listdir('augmented_data/yes/') if not f.startswith(".")]
fnames = []
for i in range(0,842):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_tumorours, fname)
    dst = os.path.join(infected_train_dir, fname)
    shutil.copyfile(src, dst)


# In[325]:


files = [f for f in os.listdir('augmented_data/yes/') if not f.startswith(".")]
fnames = []
for i in range(842,946):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_tumorours, fname)
    dst = os.path.join(infected_test_dir, fname)
    shutil.copyfile(src, dst)


# In[327]:


files = [f for f in os.listdir('augmented_data/yes/') if not f.startswith(".")]
fnames = []
for i in range(946,1050):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_tumorours, fname)
    dst = os.path.join(infected_valid_dir, fname)
    shutil.copyfile(src, dst)


# In[330]:


print(f"Train: {len(os.listdir(infected_train_dir))}")
print(f"Test: {len(os.listdir(infected_test_dir))}")
print(f"Validation: {len(os.listdir(infected_valid_dir))}")


# Non-Tumorous

# In[333]:


files = [f for f in os.listdir('augmented_data/no/') if not f.startswith(".")]
fnames = []
for i in range(0,817):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_nontumorours, fname)
    dst = os.path.join(healthy_train_dir, fname)
    shutil.copyfile(src, dst)


# In[334]:


files = [f for f in os.listdir('augmented_data/no/') if not f.startswith(".")]
fnames = []
for i in range(817,919):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_nontumorours, fname)
    dst = os.path.join(healthy_test_dir, fname)
    shutil.copyfile(src, dst)


# In[336]:


files = [f for f in os.listdir('augmented_data/no/') if not f.startswith(".")]
fnames = []
for i in range(919,1021):
    fnames.append(files[i])
for fname in fnames:
    src = os.path.join(original_dataset_nontumorours, fname)
    dst = os.path.join(healthy_valid_dir, fname)
    shutil.copyfile(src, dst)


# In[339]:


print(f"Train: {len(os.listdir(healthy_train_dir))}")
print(f"Test: {len(os.listdir(healthy_test_dir))}")
print(f"Validation: {len(os.listdir(healthy_valid_dir))}")


# # Model Building

# In[20]:


# Define data augmentation transformations for training data
train_datagen = ImageDataGenerator(rescale = 1./255,
                  horizontal_flip=0.4,
                  vertical_flip=0.4,
                  rotation_range=40,
                  shear_range=0.2,
                  width_shift_range=0.4,
                  height_shift_range=0.4,
                  fill_mode='nearest')
# Normalise validation and test data without augmentation
test_data_gen = ImageDataGenerator(rescale=1.0/255)
valid_data_gen = ImageDataGenerator(rescale=1.0/255)


# In[36]:


train_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/train/', 
                                                    batch_size=32, 
                                                    target_size=(240,240), 
                                                    class_mode='categorical',
                                                    shuffle=True, 
                                                    seed = 42, 
                                                    color_mode = 'rgb')


# In[38]:


test_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/test/', 
                                                   batch_size=32, 
                                                   target_size=(240,240), 
                                                   class_mode='categorical',
                                                   shuffle=False, 
                                                   seed = 42, 
                                                   color_mode = 'rgb')


# In[40]:


valid_generator = train_datagen.flow_from_directory('tumorous_and_nontumorous/valid/', 
                                                    batch_size=32, 
                                                    target_size=(240,240), 
                                                    class_mode='categorical',
                                                    shuffle=False, 
                                                    seed = 42, 
                                                    color_mode = 'rgb')


# In[28]:


class_labels = train_generator.class_indices
class_name = {value: key for (key,value) in class_labels.items()}


# In[30]:


class_name


# In[ ]:





# In[356]:


# Load the VGG19 model without the fully connected (top) layers
base_model = VGG19(input_shape=(240,240,3), include_top=False, weights='imagenet')

# Freeze all layers in the base model to retain pretrained weights
for layer in base_model.layers:
    layer.trainable = False

# Define the fully connected classification head
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(1024, activation='relu')(flat)
drop_out = Dropout(0.3)(class_1)  
class_2 = Dense(256, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

# Combine the base model with the custom classification head
model_01 = Model(base_model.input, output)


# In[358]:


model_01.summary()


# In[78]:


# Define callbacks to improve training efficiency
es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint(filepath='model_baseline.h5', monitor='val_loss', verbose=1, save_best_only=True)
lrr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)


# In[362]:


adam = Adam(learning_rate=0.0003)
model_01.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[364]:


# Train the model
history_01 = model_01.fit(
    train_generator, 
    steps_per_epoch=10, 
    epochs=20,  
    callbacks=[es, cp, lrr], 
    validation_data=valid_generator
)


# In[98]:


from tensorflow.keras.models import load_model

model_01 = load_model("model_baseline.h5")


# In[100]:


baseline_valid_eval = model_01.evaluate(valid_generator)
baseline_test_eval = model_01.evaluate(test_generator)

print(f'Validation Loss: {baseline_valid_eval[0]}')
print(f'Validation Acc: {baseline_valid_eval[1]}')
print(f'Testing Loss: {baseline_test_eval[0]}')
print(f'Testing Acc: {baseline_test_eval[1]}')


# In[114]:


results_dict = {"validation": {}, "test": {}}

results_dict["validation"]["baseline"] = baseline_valid_eval
results_dict["test"]["baseline"] = baseline_test_eval


# In[62]:


# Generate predictions from the model
baseline_prediction = model_01.predict(test_generator, steps=len(test_generator), verbose = 1)
y_pred = np.argmax(baseline_prediction, axis=1)
y_true = test_generator.classes # Actual labels


# In[191]:


from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_baseline.png", dpi=300, bbox_inches="tight")
plt.show()


# In[70]:


model_02 = load_model("model_baseline.h5")

set_trainable = False
for layer in model_02.layers:
    if layer.name in ['block5_conv4', 'block5_conv3']:  # Unfreeze these layers
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[72]:


model_02.summary()


# In[74]:


adam2 = Adam(learning_rate=0.00001)
model_02.compile(optimizer=adam2, loss='categorical_crossentropy', metrics=['accuracy'])


# In[86]:


cp = ModelCheckpoint(filepath="model_finetuned_stage1.h5", monitor='val_loss', verbose=1, save_best_only=True)


# In[80]:


history_02 = model_02.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,  
    validation_data=valid_generator,
    callbacks=[es, cp, lrr]
)


# In[106]:


model_02 = load_model("model_finetuned_stage1.h5")


# In[108]:


finetune1_valid_eval = model_02.evaluate(valid_generator)
finetune1_test_eval = model_02.evaluate(test_generator)

print(f'Validation Loss: {finetune1_valid_eval[0]}')
print(f'Validation Acc: {finetune1_valid_eval[1]}')
print(f'Testing Loss: {finetune1_test_eval[0]}')
print(f'Testing Acc: {finetune1_test_eval[1]}')


# In[116]:


results_dict["validation"]["finetune1"] = finetune1_valid_eval
results_dict["test"]["finetune1"] = finetune1_test_eval


# In[118]:


results_dict


# In[120]:


finetune1_prediction = model_02.predict(test_generator, steps=len(test_generator), verbose = 1)
y_pred2 = np.argmax(finetune1_prediction, axis=1)


# In[193]:


cm = confusion_matrix(y_true, y_pred2)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_finetune1.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:





# In[124]:


model_03 = load_model("model_finetuned_stage1.h5")

for layer in model_03.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
    else:
        layer.trainable = False


# In[126]:


adam3 = Adam(learning_rate=0.000005)
model_03.compile(optimizer=adam3, loss='categorical_crossentropy', metrics=['accuracy'])


# In[128]:


cp = ModelCheckpoint(filepath="model_finetuned_stage2.h5", monitor='val_loss', verbose=1, save_best_only=True)


# In[130]:


history_03 = model_03.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,  
    validation_data=valid_generator,
    callbacks=[es, cp, lrr]
)


# In[ ]:





# In[132]:


model_03 = load_model("model_finetuned_stage2.h5")


# In[134]:


finetune2_valid_eval = model_03.evaluate(valid_generator)
finetune2_test_eval = model_03.evaluate(test_generator)

print(f'Validation Loss: {finetune2_valid_eval[0]}')
print(f'Validation Acc: {finetune2_valid_eval[1]}')
print(f'Testing Loss: {finetune2_test_eval[0]}')
print(f'Testing Acc: {finetune2_test_eval[1]}')


# In[135]:


results_dict["validation"]["finetune2"] = finetune2_valid_eval
results_dict["test"]["finetune2"] = finetune2_test_eval


# In[136]:


finetune2_prediction = model_03.predict(test_generator, steps=len(test_generator), verbose = 1)
y_pred3 = np.argmax(finetune2_prediction, axis=1)


# In[195]:


cm = confusion_matrix(y_true, y_pred3)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_finetune2.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:





# In[145]:


model_04 = load_model("model_finetuned_stage2.h5")

for layer in model_04.layers:
    if layer.name.startswith('block4') or layer.name.startswith('block5'):
        layer.trainable = True
    else:
        layer.trainable = False


# In[147]:


adam4 = Adam(learning_rate=0.000002)
model_04.compile(optimizer=adam4, loss='categorical_crossentropy', metrics=['accuracy'])


# In[149]:


cp = ModelCheckpoint(filepath="model_finetuned_stage3.h5", monitor='val_loss', verbose=1, save_best_only=True)


# In[151]:


history_04 = model_04.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10,  
    validation_data=valid_generator,
    callbacks=[es, cp, lrr]
)


# In[153]:


model_04 = load_model("model_finetuned_stage3.h5")


# In[154]:


finetune3_valid_eval = model_04.evaluate(valid_generator)
finetune3_test_eval = model_04.evaluate(test_generator)

print(f'Validation Loss: {finetune3_valid_eval[0]}')
print(f'Validation Acc: {finetune3_valid_eval[1]}')
print(f'Testing Loss: {finetune3_test_eval[0]}')
print(f'Testing Acc: {finetune3_test_eval[1]}')


# In[156]:


results_dict["validation"]["finetune3"] = finetune3_valid_eval
results_dict["test"]["finetune3"] = finetune3_test_eval


# In[157]:


finetune3_prediction = model_04.predict(test_generator, steps=len(test_generator), verbose = 1)
y_pred4 = np.argmax(finetune3_prediction, axis=1)


# In[197]:


cm = confusion_matrix(y_true, y_pred4)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_finetune3.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:





# In[163]:


model_05 = load_model("model_finetuned_stage3.h5")

for layer in model_05.layers:
    layer.trainable = True


# In[165]:


adam5 = Adam(learning_rate=0.000001)
model_05.compile(optimizer=adam5, loss='categorical_crossentropy', metrics=['accuracy'])


# In[168]:


cp = ModelCheckpoint(filepath="model_final.h5", monitor='val_loss', verbose=1, save_best_only=True)


# In[170]:


history_final = model_05.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=10, 
    validation_data=valid_generator,
    callbacks=[es, cp, lrr]
)


# In[172]:


model_05 = load_model("model_final.h5")


# In[174]:


final_valid_eval = model_05.evaluate(valid_generator)
final_test_eval = model_05.evaluate(test_generator)

print(f'Validation Loss: {final_valid_eval[0]}')
print(f'Validation Acc: {final_valid_eval[1]}')
print(f'Testing Loss: {final_test_eval[0]}')
print(f'Testing Acc: {final_test_eval[1]}')


# In[175]:


results_dict["validation"]["final"] = final_valid_eval
results_dict["test"]["final"] = final_test_eval


# In[176]:


final_prediction = model_05.predict(test_generator, steps=len(test_generator), verbose = 1)
y_pred5 = np.argmax(final_prediction, axis=1)


# In[199]:


cm = confusion_matrix(y_true, y_pred5)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_final.png", dpi=300, bbox_inches="tight")
plt.show()


# # Saving data

# In[183]:


import pickle

with open("fine_tuning_results.pkl", "wb") as f:
    pickle.dump(results_dict, f)


# In[185]:


with open("history_finetune_stage1.pkl", "wb") as f:
    pickle.dump(history_02.history, f)

with open("history_finetune_stage2.pkl", "wb") as f:
    pickle.dump(history_03.history, f)

with open("history_finetune_stage3.pkl", "wb") as f:
    pickle.dump(history_04.history, f)

with open("history_finetune_final.pkl", "wb") as f:
    pickle.dump(history_final.history, f)


# In[ ]:




