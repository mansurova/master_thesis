import matplotlib.pyplot as plt  
# from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd 
import numpy as np
import pickle 
import matplotlib.pyplot as plt 


def compute_rates(y_test, y_pred, pos_label=1):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 1
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i], pos_label=pos_label)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def get_data_28_svm():
    file = "uscecchini28.csv"
    df = pd.read_csv(file)

    y = df['misstate']
    y = preprocessing.label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]

    x = df.iloc[:, 9:37]#.values
    
    x = x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    print(x.shape, y.shape)
    return x, y


def adjust_pred(y_pred):
    # return np.sqrt((y_pred - np.mean(y_pred))**2)
    return y_pred


fraud_data = pd.read_csv("fraud_acc.csv").to_numpy()
y_fraud = np.array([[1] ] * fraud_data.shape[0])

true_data = pd.read_csv("true_acc.csv").to_numpy()
y_true = np.array([[0] ]* true_data.shape[0])

print(y_fraud.shape)
print(y_true.shape)
y_test = np.vstack([y_fraud, y_true])
data = np.vstack([fraud_data, true_data])
print(y_test.shape)
print(data.shape)

## GAN
res_dir = "../exp/gan/trial5/"
disc = tf.keras.models.load_model(res_dir + "disc.h5") 
y_pred = disc(data).numpy()
y_pred = adjust_pred(y_pred)
fpr_gan, tpr_gan, roc_auc_gan = compute_rates(y_test, y_pred)


## WGAN 
res_dir = "../exp/raw/wgan/"
disc = tf.keras.models.load_model(res_dir + "disc.h5") 
y_pred = disc(data).numpy()
y_pred = adjust_pred(y_pred)
fpr_wgan, tpr_wgan, roc_auc_wgan = compute_rates(y_test, y_pred, 0)


## Disc (WGAN / ground truth)
res_dir = "../exp/disc/DiscGAN/"
disc = tf.keras.models.load_model(res_dir + "disc.h5") ## discriminator trained on generated (WGAN) and ground truth
y_pred = disc(data).numpy()
y_pred = adjust_pred(y_pred)
fpr_dgan, tpr_dgan, roc_auc_dgan = compute_rates(y_test, y_pred)


## WDisc (GAN / ground truth)
res_dir = "../exp/disc/CriticWGAN/"
disc = tf.keras.models.load_model(res_dir + "disc.h5") ## critic trained on generated (GAN) and ground truth
y_pred = disc(data).numpy()
y_pred = adjust_pred(y_pred)
fpr_wdgan, tpr_wdgan, roc_auc_wdgan = compute_rates(y_test, y_pred)


### SVM CLASSIFIER 28 DIMS SVM mit rbf kernel
pkl_filename = "../binary_models/svm.pkl"
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)
x, y = get_data_28_svm()
y_pred = classifier.decision_function(x)[:, np.newaxis]
y_pred = adjust_pred(y_pred)
fpr_svm_28, tpr_svm_28, roc_auc_svm_28 = compute_rates(y, y_pred)


### SVM CLASSIFIER 28 DIMS logit
pkl_filename = "../binary_models/logit.pkl"
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)
x, y = get_data_28_svm()
y_pred = classifier.decision_function(x)[:, np.newaxis]
y_pred = adjust_pred(y_pred)
fpr_log_28, tpr_log_28, roc_auc_log_28 = compute_rates(y, y_pred)





plt.figure()
lw = 1
plt.plot(fpr_gan[0], tpr_gan[0],  lw=lw, label='GAN (%0.2f)' % roc_auc_gan[0])
plt.plot(fpr_wgan[0], tpr_wgan[0],  lw=lw, label='WGAN (%0.2f)' % roc_auc_wgan[0])
plt.plot(fpr_dgan[0], tpr_dgan[0],  lw=lw, label='DiscGAN (WGAN/true) (%0.2f)' % roc_auc_dgan[0])
plt.plot(fpr_wdgan[0], tpr_wdgan[0], lw=lw, label='CriticWGAN (GAN/true) (%0.2f)' % roc_auc_wdgan[0])
plt.plot(fpr_svm_28[0], tpr_svm_28[0], lw=lw, label='SVM rbf (%0.2f)' % roc_auc_svm_28[0])
plt.plot(fpr_log_28[0], tpr_log_28[0], lw=lw, label='Logit (%0.2f)' % roc_auc_log_28[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

