import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from loaddata_Copy4 import Data
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from statsmodels.tsa.arima_model import ARIMA
import keras.initializers as initializers
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import StandardScaler


class FuseAD:
    def __init__(self, data, number, train_index, test_index, p, d, q,csv, write_path=None):
        self.a_p = p
        self.a_q = q
        self.a_d = d
        self.d = data
        self.N = number
        self.csv = csv
        self.write_path = write_path
        #n = int(len(data) * scale)
        self.output = (self.window(data.iloc[train_index, :], number))
        self.testdata = self.window(data.iloc[test_index, :], number)
        self.test = data.iloc[test_index[50:], :].copy()
        self.arilabel=self.test['label'].values
    
    def window(self, d, N):
        output = []
        for i in range(N, len(d)):
            l = d.iloc[(i-N):i, 1].values.tolist()
            l.extend([d.iloc[i, 1]])
            l.extend([d.iloc[i, 2]])
            output.append(l)
        return pd.DataFrame(output)

    #def plot_arima(self, c, value):
    #    plt.title('arima fit window value,number ' + c)
    #    plt.plot(value)
    #    plt.xlabel('index')
    #    plt.ylabel('value')
    #    plt.show()

    def arima(self):
        # todo finish all window arima
        model = ARIMA(self.output.iloc[0, :-2], order=(self.a_p, self.a_d, self.a_q))
        result = model.fit()
        a_y = result.predict(50)
        self.bias = int(a_y.iloc[0])

    def train(self):
        self.arima()
        # Initialising the CNN
        classifier = Sequential()
        # Step 1 - Convolution
        classifier.add(Conv1D(filters=32, kernel_size=5, input_shape=(self.N, 1), activation='relu', use_bias=True, bias_initializer=initializers.Constant(self.bias)))
        # Step 2 - Pooling
        classifier.add(MaxPooling1D(pool_size=5))
        # Adding a second convolutional layer
        classifier.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=5))
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(units=60, activation='relu'))
        classifier.add(Dense(units=1, activation='relu'))
        # Compiling the CNN
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(optimizer=sgd, loss='mae', metrics=['accuracy'])
        n = int(len(self.output) * .4)  # 40%train 60%test
        x_train = self.output.iloc[:n, :-2]
        y_train = self.output.iloc[:n, -2]
        x_test = self.output.iloc[n:, :-2]
        y_test = self.output.iloc[n:, -2]
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        classifier.fit(x_train, y_train, validation_data=(x_test, y_test))
        self.model = classifier
        self.sc = sc

    def anomaly_detection(self):
        print('***** Anomaly Detection *****')
        x = self.testdata.iloc[:, :-2]
        y = self.testdata.iloc[:, -2].values
        x = self.sc.transform(x)
        x = np.expand_dims(x, axis=2)
        y_pred = self.model.predict(x)
        y_pred=y_pred.reshape(y_pred.shape[0],)
        return abs(y_pred + self.bias - y)

    def plot(self):
        plt.plot(self.d['value'], label='value')
        plt.legend(loc='best')
        plt.xlabel('timestamp')
        plt.ylabel('value')
        plt.title(csv)
        plt.show()

        tp_set = self.test.loc[(self.test['label'] == 1) & (self.test['test_label'] == 1)]
        fp_set = self.test.loc[(self.test['label'] == 0) & (self.test['test_label'] == 1)]
        tn_set = self.test.loc[(self.test['label'] == 0) & (self.test['test_label'] == 0)]
        fn_set = self.test.loc[(self.test['label'] == 1) & (self.test['test_label'] == 0)]
        plt.vlines(tp_set['value'].index, 0, tp_set['value'], label='tp', colors='blue')
        plt.vlines(fp_set['value'].index, 0, fp_set['value'], label='fp', colors='red')
        plt.vlines(tn_set['value'].index, 0, tn_set['value'], label='tn', colors='green')
        plt.vlines(fn_set['value'].index, 0, fn_set['value'], label='fn', colors='orange')
        plt.legend(loc='best')
        plt.xlabel('timestamp')
        plt.ylabel('value')
        plt.title(csv + ' after test')
        plt.show()
    def print_result(self,save_path):
        print('***** Print Result *****')
        #print(len(self.test))
        file_data = pd.DataFrame(self.test, columns=['timestamp','value', 'label', 'score'])
        #print(file_data)
        file_data.to_csv(save_path)
    def evaluate(self, a):
        a.reshape(a.shape[0],)
        self.arilabel.reshape(self.arilabel.shape[0],)
        print(a)
        print(self.arilabel)
        precision, recall, thresholds = precision_recall_curve(self.arilabel,a,pos_label=0)
        #precision, recall, thresholds = precision_recall_curve(ls,l,pos_label=0)
        f1 = 2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1])
        f1[np.isnan(f1)]=0
        m_idx = np.argmax(f1)
        m_thresh = thresholds[m_idx]
        self.test['score']=a
        #abnormal = np.where(a > threshold)[1]
        ##print('eva')
        ##print(abnormal)
        #ab=abnormal
        #test_labe = [0 for i in range(0,len(self.testdata[51].index))]
        ##print(len(self.testdata[51].index))
        #test_label=pd.DataFrame(test_labe)
        #test_label.iloc[ab,:]=1
        ##print(test_label)
        ##print(len(test_label))
        ##for i in self.testdata[51].index:
        ##    if i in abnormal:
        ##        test_label.append(1)
        ##    else:
        ##        test_label.append(0)
        #self.test['test_label'] = test_label.values.tolist()
        #y_true = self.testdata.iloc[:, 51].values
        #y_pred = np.array(test_label)
        #print(y_true.shape)
        #y_pred=y_pred.reshape(y_pred.shape[0])
        #print(y_pred.shape)
        ##print('tpfn')
        ##print(np.equal(y_pred, 1))
        ##print(np.equal(y_true, 0))
        ##print(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
        #tp = int(np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0))))  # true positive
        #fp = int(np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0))))  # false positive
        #tn = int(np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1))))  # true negative
        #fn = int(np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1))))  # false negative
        #precision = 0 if tp + fp == 0 else tp / (tp + fp)
        #recall = 0 if tp + fn == 0 else tp / (tp + fn)
        #f1_score = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        #print('TP:%d FP:%d TN:%d FN:%d' % (tp, fp, tn, fn))
        print('precision:%0.2f recall:%0.2f f1_score:%0.2f' % (precision[m_idx],recall[m_idx], f1[m_idx]))
        with open(self.write_path, 'a') as f:
            #f.writelines('TP:%d FP:%d TN:%d FN:%d\n' % (tp, fp, tn, fn))
            f.writelines('precision:%0.2f recall:%0.2f f1_score:%0.2f\n' % (precision[m_idx],recall[m_idx], f1[m_idx]))
        return precision[m_idx],recall[m_idx],f1[m_idx]

def list_csv(path, csv_list):
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    file = [os.path.join(path, i) for i in lsdir if os.path.isfile(os.path.join(path, i))]
    csv_list.append(file)
    if dirs:
        for i in dirs:
            list_csv(os.path.join(path, i), csv_list)

    
if __name__ == '__main__':
    set_name=['other']#'sogou','ebay',
    set_path = 'testdata/dataset'
    lied = []
    for ki in os.listdir('testdata/fusead_dataset/other'):
        lied.append('testdata/dataset/other/'+ki)
    #'testdata/dataset/sogou/50C2XJ2_AvgCost.csv',
    #lied=['testdata/dataset/sogou/CVT0JD2_AvgCost.csv']
    for name in set_name:
        path=os.path.join(set_path,name)
        csv_list = []
        list_csv(path, csv_list)
        print(csv_list)
        csvs = []
        [csvs.extend(cl) for cl in csv_list]
        for csv in csvs:
            if csv in lied:
                continue
            print(csv)
            with open('testdata/fusead_dataset/evaluate5.txt', 'a') as f:
                f.writelines(csv + '\n')
            data = Data(csv)
            df = data.load()
            kf= KFold(n_splits=10, shuffle=False)
            precisions=[]
            recalls=[]
            scores=[]
            i=0
            if not os.path.exists(os.path.join('testdata/fusead_dataset',name,csv.split('/')[-1])):
                os.makedirs(os.path.join('testdata/fusead_dataset',name,csv.split('/')[-1]))
            for train_index, test_index in (kf.split(df['value'])):
                lag=True
                i=i+1
                fuseAD = FuseAD(df, 50, train_index, test_index, 2, 1, 1,csv, 'testdata/fusead_dataset/evaluate5.txt')
                if len(fuseAD.arilabel[np.isnan(fuseAD.arilabel)]) == fuseAD.arilabel.shape[0]:
                    continue
                fuseAD.arilabel[np.isnan(fuseAD.arilabel)] = 0
                #deepant = DeepAnt(df, 50, train_index, test_index, csv, 'testdata/fusead_dataset/evaluate5.txt')
                try:
                    fuseAD.train()
                except:
                    lag=False
                print('end---------------')
                if lag==False:
                    continue
                a = fuseAD.anomaly_detection()
                #print(a)s
                #print(len(a))
                try:
                    p,r,f=fuseAD.evaluate(a)
                except:
                    lag = False
                if lag==False:
                    continue
                #deepant.train()
                
                #a = deepant.anomaly_detection()
                
                #p,r,f=deepant.evaluate(a, 3)
                precisions.append(p)
                recalls.append(r)
                scores.append(f)
                fuseAD.print_result(os.path.join('testdata/fusead_dataset',name,csv.split('/')[-1],'result'+str(i)+'.txt'))
            ten_fold=pd.DataFrame({'precision':precisions,'recall':recalls,'f1_score':scores})
            with open('testdata/fusead_dataset/evaluate5.txt', 'a') as fw:
                fw.writelines('final:precision:%0.2f recall:%0.2f f1_score:%0.2f\n' % (ten_fold['precision'].mean(),ten_fold['recall'].mean(),ten_fold['f1_score'].mean()))
                fw.writelines('\n')