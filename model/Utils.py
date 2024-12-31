from genericpath import exists
import torch
import numpy as np
import os
import sklearn.metrics as metrics

def AddContext(x, context, label=False, dtype=float):
    assert context % 2 == 1, "context value error."
    cut = int(context / 2)
    if label:
        tData = x[cut:x.shape[0] - cut]
    else:
        tData = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1]], dtype=dtype)
        for i in range(cut, x.shape[0] - cut):
            tData[i - cut] = x[i - cut:i + cut + 1]
    return tData

def accuracy(output, labels):
    preds = torch.argmax(output.cpu(),1)
    labels = torch.argmax(labels.cpu(),1)
    acc = metrics.accuracy_score(labels.cpu(),preds)
    return acc

def PrintScore(true,pred,savePath,average='macro'):
    if not os.path.exists(savePath):
        with open(savePath, 'w') as saveFile:
            pass
    saveFile = open(savePath, 'a+')
    # Main scores
    F1=metrics.f1_score(true,pred,average=None)
    print("Main scores:")
    print('-------------------------------------------------------',file=saveFile)
    print('Acc\t\tF1S\t\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R',file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'% (metrics.accuracy_score(true,pred),
                                                             metrics.f1_score(true,pred,average='macro'),
                                                             metrics.cohen_kappa_score(true,pred),
                                                             F1[0],F1[1],F1[2],F1[3],F1[4]),
                                                             file=saveFile)
    # Classification report
    print("\nClassification report:",file=saveFile)
    print(metrics.classification_report(true,pred,target_names=['Wake','N1','N2','N3','REM']),file=saveFile)
    # Confusion matrix
    print('Confusion matrix:',file=saveFile)
    print(metrics.confusion_matrix(true,pred),file=saveFile)
    # Overall scores
    print('\n    Accuracy\t',metrics.accuracy_score(true,pred),file=saveFile)
    print(' Cohen Kappa\t',metrics.cohen_kappa_score(true,pred),file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=average),'\tAverage =',average,file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=average),'\tAverage =',average,file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=average),'\tAverage =',average,file=saveFile)
    # Results of each class
    print('\nResults of each class:',file=saveFile)
    print('    F1-Score\t',metrics.f1_score(true,pred,average=None),file=saveFile)
    print('   Precision\t',metrics.precision_score(true,pred,average=None),file=saveFile)
    print('      Recall\t',metrics.recall_score(true,pred,average=None),file=saveFile)
    print('      \n', file=saveFile)

    saveFile.close()

    print('Acc\t\tF1S\t\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R')
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (metrics.accuracy_score(true, pred),
                                                              metrics.f1_score(true, pred, average='macro'),
                                                              metrics.cohen_kappa_score(true, pred),
                                                              F1[0], F1[1], F1[2], F1[3], F1[4]))
    # Classification report
    print("\nClassification report:")
    print(metrics.classification_report(true, pred, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))
    # Confusion matrix
    print('Confusion matrix:')
    print(metrics.confusion_matrix(true, pred))
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred))
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred))
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average)
    # Results of each class
    print('\nResults of each class:')
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None))
    print('   Precision\t', metrics.precision_score(true, pred, average=None))
    print('      Recall\t', metrics.recall_score(true, pred, average=None))

    return

