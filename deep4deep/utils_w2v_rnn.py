import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from deep4deep.utils import simple_time_tracker


@simple_time_tracker
def make_X_check(X_val, y_val, X_val_check, model):
    '''
    arguments:
    : a copy of a X_set with columns as prepared by
    before any preprocessing or embedding
    : the associated targets
    : a model that can be applied to X_val to predict values for y
    returns:
    a dataframe with X_val_check's fields (for inspection) and
    confusion matrix values: true positives (TP), etc.
    '''
    threshold = .5
    X_val_check = pd.DataFrame(X_val_check)
    X_val_check['y_pred'] = model.predict(X_val)
    X_val_check['y_true'] = y_val
    X_val_check['y_pred_binary'] = [1 if item >threshold else 0 for item in X_val_check.y_pred]

    condition_target_1 = (X_val_check.y_true==1.0)
    condition_pred_1 = (X_val_check.y_pred_binary==1)
    X_val_check['TP'] = condition_target_1 & condition_pred_1
    X_val_check['TN'] = ~condition_target_1 & ~condition_pred_1
    X_val_check['FP'] = ~condition_target_1 & condition_pred_1
    X_val_check['FN'] = condition_target_1 & ~condition_pred_1

    return X_val_check

@simple_time_tracker
def my_metrics(X_val_check):
    '''
    takes a dataframe of the form returned by make_X_check
    '''


    total = X_val_check.shape[0]

    accuracy = (X_val_check.TP.sum()+X_val_check.TN.sum())/total
    precision = X_val_check.TP.sum()/(X_val_check.TP.sum()+X_val_check.FP.sum())
    recall = X_val_check.TP.sum()/(X_val_check.TP.sum()+X_val_check.FN.sum())
    f1 = 2*((precision*recall)/(precision+recall))

    print(f"accuracy: {accuracy *100:.2f} %")
    print(f"precision: {precision *100:.2f} %")
    print(f"recall: {recall *100:.2f} %")
    print(f"f1: {f1 *100:.2f} %")

    cm = confusion_matrix(X_val_check.y_true, X_val_check.y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              #display_labels=display_labels
                             )
    disp = disp.plot(include_values=True,
                 cmap='viridis', ax=None, xticks_rotation='horizontal')
    print("Confusion matrix")
    plt.show()

    print(f"{round(accuracy*100)}\t{round(precision*100)}\t{round(recall*100)}\t{round( f1*100)}")
    return {"accuracy": accuracy,"precision": precision,"recall": recall, "f1": f1 }


def plot_loss_accuracy(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()

    return None

def rmse(X_val, y_val, model):
    y_pred = model.predict(X_val)
    model_rmse = ((y_val - y_pred[:,0])**2).mean()**0.5
    print(f"model_rmse: {model_rmse:.2f}, vs original_rmse: {.4325:.2f} and dummy_rmse:{0.5:.2f} ")
    return model_rmse


