
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1)


# In[2]:



X, Y = load_planar_dataset()


# In[3]:


plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)


# In[4]:



shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  


print ("X的维度为: " + str(shape_X))
print ("Y的维度为: " + str(shape_Y))
print ("数据集里面的数据有：" + str(m) + " 个")


# In[5]:


clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);


# In[6]:


plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y)) #绘制决策边界
plt.title("Logistic Regression") #图标题
LR_predictions  = clf.predict(X.T) #预测结果
print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")


# In[8]:


img=Image.open('images/一个隐藏层.png')
plt.figure(figsize=(20,20))
plt.imshow(img)    


# In[9]:


#定义层
def layer_sizes(X, Y):
    n_x=X.shape[0]#输入层
    n_h=4#隐藏层
    n_y=Y.shape[0]#输出层
    return n_x,n_h,n_y


# In[11]:


#查看
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("输入层: n_x = " + str(n_x))
print("隐藏层: n_h = " + str(n_h))
print("输出层: n_y = " + str(n_y))


# In[14]:


#定义权重
def initialize_parameters(n_x, n_h, n_y):
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[15]:


#查看
n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In[19]:


def forward_propagation(X, parameters):
    #获取参数
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    
    #
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    
    assert(A2.shape==(1,X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[20]:


#查看
X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))


# In[22]:


img=Image.open('images/单隐藏层损失函数.png')
plt.figure(figsize=(15,15))
plt.imshow(img)    


# In[23]:


#计算损失函数
def compute_cost(A2, Y, parameters):
    #样本数量
    m=Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    log_cost=np.multiply(Y,np.log(A2))+np.multiply((1-Y),np.log(1-A2))
    cost=-np.sum(log_cost)/m
    
    
    cost=np.squeeze(cost)
    assert(isinstance(cost, float))
    
    return cost
    
    
    


# In[24]:


#查看
A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# In[29]:


#图片源自Andrew Ng
#用于计算反向传播
img2=Image.open('images/Backpropagation.png')
plt.figure(figsize=(15,15))
plt.imshow(img2)


# In[31]:


#反向传播
def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
 
    A1 = cache['A1']
    A2 = cache['A2']

    
    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[32]:


#查看
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


# In[33]:


#更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
 
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    


# In[34]:


#查看

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In[35]:


#整合
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    
    np.random.seed(3)
    
    #输入与输出层数
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    #根据层数初始化权重！
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #开始迭代
    for i in range(0, num_iterations):
        #前向传播
        A2, cache = forward_propagation(X, parameters)
        
        #计算代价函数
        cost = compute_cost(A2, Y, parameters)
 
        #反向传播
        grads = backward_propagation(parameters, cache, X, Y)
 
        #更新参数
        parameters = update_parameters(parameters, grads)

        #true的时候可以方便查看cost
        if print_cost and i % 1000 == 0:
            print ("迭代后cost %i: %f" % (i, cost))

    return parameters


# In[36]:


#查看
X_assess, Y_assess = nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In[37]:


#预测。
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    
    return predictions


# In[38]:


#查看
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("平均值 = " + str(np.mean(predictions)))


# In[45]:


# 建立模型
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

# 绘图
plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
#plt.title中文不好显示
plt.title("one hidden layer size"+str(4))

predictions = predict(parameters, X)

print("predictions shape"+str(predictions.shape))
print("Y shape"+str(Y.shape))


print ("准确率： %d " % float((np.dot(Y, predictions.T) + np.dot(1 - Y,1 - predictions.T)) / float(Y.size) * 100) +"% ")


# In[49]:



#更改隐藏层的节点，查看一下
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50,100]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    

