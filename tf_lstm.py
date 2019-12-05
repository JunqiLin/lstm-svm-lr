#！/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
tf.reset_default_graph()
df=pd.read_csv('./data.csv',encoding='gbk', index_col = 0)

data = df['close']
#data=np.array(df['qty'])
#data = np.array(df.iloc[:-4,2])

data_close=np.array(df['close'])
data_preclose=np.array(df['preclose'])
yd=[]
num_x=len(df)

for i in range(num_x):   
    if data_close[i]>=data_preclose[i]:
        yd.append(1)
    else:
        yd.append(0)


normalize_data = data

normalize_data=(data-np.mean(data))/np.std(data)
normalize_data=normalize_data[:,np.newaxis]

time_step=40
rnn_unit=10
batch_size=128
input_size=1
output_size=1
lr=0.0006
train_x,train_y=[],[]
split = 60


for i in range(len(normalize_data)-time_step-split):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

X=tf.placeholder(tf.float32, [None,time_step,input_size])
Y=tf.placeholder(tf.float32, [None,time_step,output_size])
#
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

def lstm(batch):      #
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
#    saver=tf.train.Saver(tf.global_variables())
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(30):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                if step%50==0:
                    print("d")
                    print(i,step,loss_)
                    print("保存模型：", saver.save(sess,'./model/stock1.model',global_step=i))
                step+=1
def prediction():
    pred,_=lstm(1)      #
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('./model')
        saver.restore(sess, module_file)
        prev_seq=train_x[-1]
        print(prev_seq)
        predict=[]
        for i in range(split): 
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
#        print(predict)
        p_start = len(train_x)+time_step
        
        y_result = []        
        y_p = np.roll(predict,1)
        predict = np.array(predict)
        
        for i in range(len(predict)):
            if predict[i]>y_p[i]:
                y_result.append(1)
            else:
                y_result.append(0)
                
        y_origin = yd[p_start:]
#        print(len(y_origin))
#        print("###")
#        print(len(y_result))
        
        a = 0
        for i in range(len(y_origin)):
            if y_origin[i] == y_result[i]:
                a = a+1
                
#        acc = np.mean(y_origin == y_result)
        print("LSTM classifier accuacy:")
#        print(result2)
        print(a/split)
#        plt.figure()
#        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
#        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
#        plt.show()
#        plt.savefig('./re.png')

train_lstm()
prediction()



