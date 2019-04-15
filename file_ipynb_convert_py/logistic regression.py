
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>train_set_x_orig  训练特征</h3>
# <h3>train_set_y     训练标签</h3>
# <h3>test_set_x_orig   测试特征</h3>
# <h3>test_set_y      测试标签</h3>
# <h3>classes        类别
# </h3>

# In[2]:


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# In[3]:


print(train_set_x_orig.shape)
print(train_set_y.shape)
print(test_set_x_orig.shape)
print(test_set_y.shape)
print(classes.shape)


# In[4]:


#查看其中图片
index = 24
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")


# In[5]:


#查看维度
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
#刚刚好是64*64.
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# In[6]:




#64*64*3，再转置，
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


# In[7]:


#预处理。
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# In[8]:


img=Image.open('images/逻辑.png')
plt.figure(figsize=(20,20))
plt.imshow(img)    


# In[9]:


#计算sigmoid函数
def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z


# In[10]:


#查看输出
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(1) = " + str(sigmoid(1)))
print ("sigmoid(2) = " + str(sigmoid(2)))
print ("sigmoid(3) = " + str(sigmoid(3)))
print ("sigmoid(4) = " + str(sigmoid(4)))


# In[11]:


#初始化权重，神经网络不可初始化为0，需要随机数值。
def initialize_with_zeros(dim):
    w=np.zeros(shape=(dim,1))
    b=0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w,b


# In[12]:


dim = 2
w, b = initialize_with_zeros(dim)
print(w.shape)
print ("w = " + str(w))
print ("b = " + str(b))


# <h1>#公式推现图</h1>

# In[13]:


img=Image.open('images/数学推现.png')
plt.figure(figsize=(20,20))
plt.imshow(img)    


# In[14]:


#根据上图推导求
def propagate(w, b, X, Y):
    
    #m为样本个数。
    m=X.shape[1]
    #相加后为一个值
    A=sigmoid(np.dot(w.T,X)+b)
#     print("A shape"+str(A.shape))
#     print(A)
    cost=-(1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))
    
    #多个值相加
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)
    
#     print("dw shape"+str(dw.shape))
#     print("db shape"+str(db.shape))
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    
    
    
    
    


# In[15]:


#查看一下
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("X shape"+str(X.shape))
print("X shape"+str(X.shape))
print("w shape"+str(w.shape))
print("Y shape"+str(Y.shape))
grads, cost = propagate(w, b, X, Y)

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


# In[16]:


#多次迭代
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs=[]
    for i in range(num_iterations):
        
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))            
    params = {"w": w,
              "b": b}    
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
    


# In[17]:


#查看一下
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


# In[18]:


#预测
def predict(w, b, X):
    m=X.shape[1]
    y_pre=np.zeros(shape=(1,m))
    
    z=np.dot(w.T,X)+b
    A=sigmoid(z)
    
    for i in range(A.shape[1]):
        if(A[0,i]>=0.5):
            y_pre[0,i]=1
        else:
            y_pre[0,i]=0
            
    assert(y_pre.shape == (1, m))
    
    return y_pre       


# In[19]:


print("predictions = " + str(predict(w, b, X)))


# In[20]:


#initialize_with_zeros    propagate    optimize    predict
#集合
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b=initialize_with_zeros(X_train.shape[0])
    
    
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    #100-差值，即准确率.
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    
    


# In[21]:


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# In[22]:


classes[int(d["Y_prediction_test"][0,5])]


# In[23]:


#查看一下
index = 9
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))

print("y = " + str(test_set_y[0, index])+ ", you predicted that it is a \"" +str(classes[int(d["Y_prediction_test"][0,5])])+"\" picture.")


# In[24]:


# 0.005下的cost迭代
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[25]:


learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# In[27]:



#放一张自己的图片
my_image = "luffy.jpg"   
##

# 预测图片是否有猫
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


# In[29]:



#放一张自己的图片
my_image = "猫.png"   
##

# 预测图片是否有猫
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

