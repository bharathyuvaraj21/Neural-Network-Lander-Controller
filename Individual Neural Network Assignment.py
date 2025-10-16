#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import numpy as np


# In[28]:


#read the csv file
df=pd.read_csv("Data.csv",header = None)
df.head()


# In[30]:


#check min values in all column
df.min()


# In[ ]:


#check max values in all column
df.max()


# In[29]:


#
df.shape


# In[30]:


#To check the duplicates values
df.duplicated().sum()


# In[31]:


#To drop duplicate values
df=df.drop_duplicates()


# In[32]:


#To drop NaN values
df.dropna(inplace=True)


# In[33]:


# Applying min-max scaling to each column in the DataFrame
# The lambda function takes each column 'x' and scales its values using the formula:
# Scaled Value = (Original Value - Minimum Value) / (Maximum Value - Minimum Value)
df=df.apply(lambda x:(x-x.min())/(x.max()-x.min()))


# In[34]:


#read the first 6 rows
df.head()


# In[35]:


len(df)


# In[36]:


# Determining the training set size
# The training set size is calculated as 70% of the total length of the DataFrame 'df'.
train_size=int(0.7*len(df))


# In[37]:


# The DataFrame 'df' is shuffled using the sample() method with frac=1, meaning all rows are sampled.
Schuffle_df=df.sample(frac=1)


# In[38]:


#The DataFrame 'shuffled_df' is divided into two sets: 'train_set' and 'validation_set'.
train_set=Schuffle_df[:train_size]
validation_set=Schuffle_df[train_size:]


# In[39]:


# Saving the training,validation and Test sets to CSV files
train_set.to_csv("D:/NN_3_DATA/Data_train_data.csv", index=False,header=False)
validation_set.to_csv("D:/NN_3_DATA/Data_validation_data.csv", index=False,header=False)
validation_set.to_csv("D:/NN_3_DATA/Data_Test_data.csv", index=False,header=False)


# In[2]:


#generating input random weights

def input_random_weights(n_neurons,n_input):
    input_rand_weight = []
    for x in range(n_input):
        rand = []
        for y in range(n_neurons):
            rand.append(random.random())
        input_rand_weight.append(rand)
    return input_rand_weight
        


# In[3]:


#generating output random weights


def output_random_weights(n_neurons,n_output):
    output_rand_weight =[]
    for x in range(n_neurons):
        rand =[]
        for y in range(n_output):
            rand.append(random.random())
        output_rand_weight.append(rand)
    return output_rand_weight


# In[4]:


#Generating weight_multiplication

def weight_multiplication(IP,wt):
    
    add = 0
    for x in range(len(IP)):
    
        prod = IP[x]*wt[x]
        add = add+prod
    return add


# In[5]:


#Generating activation_function

def activation_fn(multiplied_wts, lamda):
    
    AF =  1/(1 + math.exp(-multiplied_wts * lamda))
    
    return AF


# In[6]:


#Generating Local_gradient_function

def local_gradient(lamda, output, error):
    
    LG = lamda * output *(1 - output) *error
    
    return LG


# In[7]:


#generating Delta_Weights

def delta_weights(n1,n2,local_gradient, output, eta,dlta_wt_prev,alfa):
    delta_weights =[]
    for x in range(n1):
        
        weight = []
        for y in range(n2):
            weight.append((eta*local_gradient[x]*output[y])+(alfa*dlta_wt_prev[x][y]))
        delta_weights.append(weight)
    return delta_weights


# In[8]:


#generating weight_Transpose

def weight_transpose(output, hidden_nodes,inputs):
    
    wt = []
    for x in range(hidden_nodes):
        wt_temp=[]
        for y in range(inputs):
            wt_temp.append(output[y][x])
        wt.append(wt_temp)
    return wt


# In[9]:


#generating hidden_gradient
def hidden_gradient(lamda, output, summation_val):
    HG = lamda * output *(1-output) * (summation_val)
    return HG


# In[10]:


#Generate new_weights
def new_weights(previous_weight, delta_weight):
    
    wt =[]
    for x in range(len(previous_weight)):
        wt_add = []
        for y in range(len(delta_weight[0])):
            w = previous_weight[x][y] + delta_weight[x][y]
            wt_add.append(w)
        wt.append(wt_add)
    return wt


# In[11]:


#generate RMSE(root_mean_sq_error)
def root_mean_sq_error(err,out): #out
    err_1_total = 0
    err_2_total = 0
    final_err = []
    final_out = []
    for x in range(len(err)):
        err_1_total += (err[x][0])**2 
        err_2_total += (err[x][1])**2 
    mean_1_err = err_1_total/len(err)
    mean_2_err = err_2_total/len(err)
    rmse1=math.sqrt(mean_1_err)
    rmse2=math.sqrt(mean_2_err)
    final_root_mean_sq_error=(rmse1+rmse2)/2
    
  
    return final_root_mean_sq_error


# In[21]:


#variable declaration
no_of_inputs = 2
no_of_outputs = 2
no_of_hidden_nodes = int(input("Enter the number of hidden nodes in the hidden layer\n"))
Lamda = 0.6
Eta = 0.1
alfa=0.9


# In[22]:


#Callng input_random_weights function to generate random weights for input layer to hidden layer
IP_weight = []
IP_weight = input_random_weights(no_of_hidden_nodes,no_of_inputs)
print(IP_weight)


# In[23]:


#Callng output_random_weights function to generate random weights for hidden layer to output layer
OP_weight = []
OP_weight = output_random_weights(no_of_hidden_nodes,no_of_outputs)
print(OP_weight)


# In[24]:


#Previous output Delta Weight
dlta_op_wt_prev = []
for x in range(no_of_hidden_nodes):
    op_wt_1 = [0]*no_of_outputs
    dlta_op_wt_prev.append(op_wt_1)
print(dlta_op_wt_prev)


# In[25]:


#Previous Input Delta Weight
dlta_ip_wt_prev = []
for x in range(no_of_inputs):
    ip_wt_1 = [0]*no_of_hidden_nodes
    dlta_ip_wt_prev.append(ip_wt_1)
print(dlta_ip_wt_prev)


# In[26]:


#starting the EPOC
root_mean_sq_err = []
validation_root_mean_sq_err=[]
Test_root_mean_sq_err=[]
count=0
EP = int(input("enter the number of Epochs to be run"))
for num in range(EP):
  
    training_game_data = open("D:/NN_3_DATA/Train_data.csv", "r")
    y1_and_y2=[]
    e1_and_e2=[]
    for row in training_game_data.readlines():
        
        row_1 = row.rstrip().split(",")
        input_values = [float(row_1[0]),float(row_1[1])]
        output_values = [float(row_1[3]),float(row_1[2])]
        
        #calculation of feedforward values from input to hidden layer
        
        IP_wt_arrange = []
        IP_wt_arrange = weight_transpose(IP_weight,no_of_hidden_nodes,no_of_inputs)
        
        ff_1_output = []
        for x in range(no_of_hidden_nodes):
            ff_1_output.append(activation_fn(weight_multiplication(input_values,IP_wt_arrange[x]),Lamda))
        
            
         #calculation of feedforward values from hidden layer to output layer
        
        OP_wt_arrange = []
        OP_wt_arrange = weight_transpose(OP_weight,no_of_outputs,no_of_hidden_nodes)
        
        ff_2_output = []
        for x in range(no_of_outputs):
            
            ff_2_output.append(activation_fn(weight_multiplication(ff_1_output,OP_wt_arrange[x]),Lamda))
            
        #calculate the error
        
        Err =[]
        for x in range(len(ff_2_output)):
            
            error = output_values[x] - ff_2_output[x]
            Err.append(error)
        
        #break
        
        #calculation of local gradient
        
        LG = []
        for x in range(len(Err)):
            
            LG.append(local_gradient(Lamda, ff_2_output[x], Err[x]))
        
        #calculation of delta weights for output layer to hidden layer
        
        dlta_op_wt_prev_transpose=weight_transpose(dlta_op_wt_prev,no_of_outputs,no_of_hidden_nodes)
        
        delta_weights_o = delta_weights(no_of_outputs, no_of_hidden_nodes,LG,ff_1_output,Eta,dlta_op_wt_prev_transpose,alfa)
        
        #Delta Weights rearrangement for calculating output weights
        delta_weights_o_transpose=weight_transpose(delta_weights_o,no_of_hidden_nodes,no_of_outputs)
        dlta_op_wt_prev=delta_weights_o_transpose
        
        #weight updation by adding delta weights to previous weights
        updated_weight_o = new_weights(OP_weight,delta_weights_o_transpose)
        OP_weight = updated_weight_o

        #break
        
        #Weight rearrangement for calculating hidden gradient
        
        wt_arrange = []
        wt_arrange = weight_transpose(OP_weight,no_of_outputs,no_of_hidden_nodes)
        
        #Calculating summation part of hidden gradient
        
        summation_val=[]
        for i in range(no_of_hidden_nodes):
            x=0
            for j in range(len(LG)):
                y=LG[j]*wt_arrange[j][i]
                x+=y
            summation_val.append(x)
        
        #Calculation of hidden_grdient
        
        HG = []
        for x in range(len(ff_1_output)):
            
            HG.append(hidden_gradient(Lamda,ff_1_output[x],summation_val[x]))
            
        #calculation of delta weights for hidden layer to input layer
        
        dlta_ip_wt_prev_transpose=weight_transpose(dlta_ip_wt_prev,no_of_hidden_nodes,no_of_outputs)
        
        delta_weights_i = delta_weights(no_of_hidden_nodes,no_of_inputs,HG,input_values,Eta,dlta_ip_wt_prev_transpose,alfa)
        
        #Delta Weights rearrangement for calculating hidden weights
        delta_weights_i_transpose=weight_transpose(delta_weights_i,no_of_inputs,no_of_hidden_nodes)
        dlta_ip_wt_prev=delta_weights_i_transpose
        
        #weight updation by adding delta weights to previous weights
        
        updated_weight_i = new_weights(IP_weight,delta_weights_i_transpose)
        IP_weight = updated_weight_i
        
        
        
        #store error values for rmse calculation
        e1_and_e2.append(Err)
        
        #store predicted output for rmse calculation
        y1_and_y2.append(ff_2_output)
        
    training_game_data.close()
    
     #Validation Data
       
    validation_game_data = open("D:/NN_3_DATA/Validation_data.csv", "r") 
    validation_e1_and_e2=[]
    validation_y1_and_y2=[]
    for row in validation_game_data.readlines():
        
        row_1 = row.rstrip().split(",")
        input_values = [float(row_1[0]),float(row_1[1])]
        output_values = [float(row_1[3]),float(row_1[2])]
        
        #calculation of feedforward values from input to hidden layer
        
        IP_wt_arrange = []
        IP_wt_arrange = weight_transpose(IP_weight,no_of_hidden_nodes,no_of_inputs)
        
        ff_1_output = []
        for x in range(no_of_hidden_nodes):
            ff_1_output.append(activation_fn(weight_multiplication(input_values,IP_wt_arrange[x]),Lamda))
        
            
         #calculation of feedforward values from hidden layer to output layer
        
        OP_wt_arrange = []
        OP_wt_arrange = weight_transpose(OP_weight,no_of_outputs,no_of_hidden_nodes)
        
        ff_2_output = []
        for x in range(no_of_outputs):
            
            ff_2_output.append(activation_fn(weight_multiplication(ff_1_output,OP_wt_arrange[x]),Lamda))
        
         #calculate the error
        
        val_Err =[]
        for x in range(len(ff_2_output)):
            
            error = output_values[x] - ff_2_output[x]
            val_Err.append(error)
            
         #store error values for rmse calculation
        validation_e1_and_e2.append(val_Err)
        
        #store predicted output for rmse calculation
        validation_y1_and_y2.append(ff_2_output)
      
    validation_game_data.close()
    
    
    training_game_rmse = root_mean_sq_error(e1_and_e2,y1_and_y2) 
    root_mean_sq_err.append(root_mean_sq_error(e1_and_e2,y1_and_y2)) 
    
    
    validation_game_rmse = root_mean_sq_error(validation_e1_and_e2,validation_y1_and_y2) 
    validation_root_mean_sq_err.append(root_mean_sq_error(validation_e1_and_e2,validation_y1_and_y2))
    
    #EARLY STOPPING
    if validation_root_mean_sq_err[num]>=validation_root_mean_sq_err[num-1]:
        count=count+1
    else:
        count=0
    if count==5:
        print("Early stoping at :",num)
        break

    print("Running Epoch no:",num,"with RMSE_Value of:",round(root_mean_sq_error(e1_and_e2,y1_and_y2),5),"validation_game_rmse = ",round(root_mean_sq_error(validation_e1_and_e2,validation_y1_and_y2),5))
#Test Data

Test_game_data = open("D:/NN_3_DATA/Test_data.csv", "r") 
Test_e1_and_e2=[]
Test_y1_and_y2=[]
for row in Test_game_data.readlines():

    row_1 = row.rstrip().split(",")
    input_values = [float(row_1[0]),float(row_1[1])]
    output_values = [float(row_1[3]),float(row_1[2])]

    #calculation of feedforward values from input to hidden layer

    IP_wt_arrange = []
    IP_wt_arrange = weight_transpose(IP_weight,no_of_hidden_nodes,no_of_inputs)

    ff_1_output = []
    for x in range(no_of_hidden_nodes):
        ff_1_output.append(activation_fn(weight_multiplication(input_values,IP_wt_arrange[x]),Lamda))


     #calculation of feedforward values from hidden layer to output layer

    OP_wt_arrange = []
    OP_wt_arrange = weight_transpose(OP_weight,no_of_outputs,no_of_hidden_nodes)

    ff_2_output = []
    for x in range(no_of_outputs):

        ff_2_output.append(activation_fn(weight_multiplication(ff_1_output,OP_wt_arrange[x]),Lamda))

     #calculate the error

    val_Err =[]
    for x in range(len(ff_2_output)):

        error = output_values[x] - ff_2_output[x]
        val_Err.append(error)

     #store error values for rmse calculation
    Test_e1_and_e2.append(val_Err)

    #store predicted output for rmse calculation
    Test_y1_and_y2.append(ff_2_output)

Test_game_data.close()
    
Test_game_rmse = root_mean_sq_error(Test_e1_and_e2,Test_y1_and_y2) 
Test_root_mean_sq_err.append(root_mean_sq_error(Test_e1_and_e2,Test_y1_and_y2))
print("Test RMSE : ",Test_root_mean_sq_err) 
print( "Input weight : ", IP_weight)
print( "Output weight : ", OP_weight)


# In[39]:


# No of Neurons=4 ,Lamda = 0.6,Eta = 0.01,apoc=300
#Input weight :  [[2.2845455188301016, -1.2068987760484946, -0.8455908604663668, -2.2350503938578776], [2.049730825387576, -1.985089439876428, 4.351948179669111, 1.883898263035134]]
#Output weight :  [[3.960007869790556, 1.4987887756384295], [-3.317086234597661, 1.9056027023463733], [-1.179353046743311, -4.432883731516138], [-3.7629779275406063, 0.6448460000931142]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[71]:


# No of Neurons=6 ,Lamda = 0.6,Eta = 0.001,apoc=100
#Input weight :  [[1.975255147012095, -2.7033934535790087, 0.7829663544238968, 0.9143326273890113, 0.35160317898431526, -1.617199894968194], [0.29492739705109694, 2.365593227319465, 2.592551391369974, -1.2514784300309947, 3.3993168664277937, -1.1346311302235264]]
#Output weight :  [[1.2637296706329473, 1.0409756915974329], [-3.3665680369827404, -0.5964337273336557], [1.015489470541018, -1.6516651196350272], [0.5113687370495776, 2.061366876411054], [0.6902962362837833, -2.6089345141787526], [-3.979880387907353, 1.7553006925794725]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[19]:


#--------------------------------------------------------------------------------------------------------------------------------
# No of Neurons=6 ,Lamda = 0.8,Eta = 0.01,alfa=0.9,apoc=100(testing with different values like aplha =0.01 which landing but not all times)
#Input weight :  [[-10.092803897763975, -2.4837126867576584, -9.219893688015539, 0.4096336574627513, 0.9230816227930417, 7.852798040291203, -2.9558554382285127, -16.163574518730655, 1.8160589675373087, -2.663843588352432], [4.574949898545153, -9.668000223283599, 4.066842926967148, -0.7441423968960126, -18.302141144153378, -9.169237834674743, -9.394652146245344, 6.026079642098439, -7.617302936374368, -9.628715980453801]]
#Output weight :  [[-9.665698724398302, -2.8322208717282527], [-9.784893832096422, 0.0890644900550934], [-8.494803672810617, 0.1734891538938621], [7.376101997966366, -1.8888513492711827], [7.706636114084964, -1.527579924601443], [-6.169556367327769, -2.7207997700547137], [-10.62475917016842, -3.7060226500139652], [12.467346863847254, 0.9847696098761807], [12.258334123705763, 9.01453995896663], [-13.54872233427336, -1.581188655567425]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[26]:


# No of Neurons=7 ,Lamda = 0.4,Eta = 0.001,alfa=0.2,apoc=100(testing with different values)
#Input weight :  [[-10.092803897763975, -2.4837126867576584, -9.219893688015539, 0.4096336574627513, 0.9230816227930417, 7.852798040291203, -2.9558554382285127, -16.163574518730655, 1.8160589675373087, -2.663843588352432], [4.574949898545153, -9.668000223283599, 4.066842926967148, -0.7441423968960126, -18.302141144153378, -9.169237834674743, -9.394652146245344, 6.026079642098439, -7.617302936374368, -9.628715980453801]]
#Output weight :  [[-9.665698724398302, -2.8322208717282527], [-9.784893832096422, 0.0890644900550934], [-8.494803672810617, 0.1734891538938621], [7.376101997966366, -1.8888513492711827], [7.706636114084964, -1.527579924601443], [-6.169556367327769, -2.7207997700547137], [-10.62475917016842, -3.7060226500139652], [12.467346863847254, 0.9847696098761807], [12.258334123705763, 9.01453995896663], [-13.54872233427336, -1.581188655567425]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[20]:


# No of Neurons=10 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=300(Testing for RMSE VALUE) Good weights
#Input weight :  [[-9.1218183025958, -15.612205923631143, 6.71304518987757, -9.065512623885724, -3.078039799903326, -2.8581926458584714, 8.956709560203064, 2.460315264328611, -0.4898781133395387, -3.5666438080696743], [4.161085782954259, 5.820555698190883, -4.10608283684252, 4.160324109394528, 0.9223370274607275, -7.006742578262992, -9.827235381257843, -8.908199118702381, -3.7018451797177816, -6.890101840624964]]
#Output weight :  [[-9.781706927069877, -3.5479167705778805], [14.122837312257571, 0.8372663033604694], [4.755791106374785, -3.9143273348333048], [-9.725330422684472, -3.0816752108324765], [2.683091164978274, 5.6171284316194106], [-12.52286460177695, 0.02706701489173624], [-7.672433851672789, -0.08674912260771667], [8.86130338690394, 3.5177204970383906], [7.532609448844774, 3.285204442306368], [-13.353758307560941, -6.744877202140169]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[19]:


# No of Neurons=10 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=100,landing correctly(This correct weights)
#Input weight :  [[-10.092803897763975, -2.4837126867576584, -9.219893688015539, 0.4096336574627513, 0.9230816227930417, 7.852798040291203, -2.9558554382285127, -16.163574518730655, 1.8160589675373087, -2.663843588352432], [4.574949898545153, -9.668000223283599, 4.066842926967148, -0.7441423968960126, -18.302141144153378, -9.169237834674743, -9.394652146245344, 6.026079642098439, -7.617302936374368, -9.628715980453801]]
#Output weight :  [[-9.665698724398302, -2.8322208717282527], [-9.784893832096422, 0.0890644900550934], [-8.494803672810617, 0.1734891538938621], [7.376101997966366, -1.8888513492711827], [7.706636114084964, -1.527579924601443], [-6.169556367327769, -2.7207997700547137], [-10.62475917016842, -3.7060226500139652], [12.467346863847254, 0.9847696098761807], [12.258334123705763, 9.01453995896663], [-13.54872233427336, -1.581188655567425]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[78]:


# No of Neurons=4 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=100,landing correctly
#Input weight :  [[-4.865807362788303, -8.938154909098616, -2.972125996128372, -16.09611539286542], [2.174042800375683, 2.3562650536215437, 5.595414692862924, 4.3995205263785975]]
#Output weight :  [[0.6551654676008074, 11.043161946575452], [-15.662688843999558, -10.48221925227467], [1.2424463922486357, -4.8080488029896005], [10.178533911147978, 2.5016814117248183]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[19]:


# No of Neurons=6 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=100,(landing correctly)
#Input weight :[[2.1400755894554844, -9.537616130415614, -6.956909784267166, -8.92768650396283, 3.4525897342593295, -16.01420612948682], [-5.091296649894389, 3.9383107196603246, 6.503954675188068, 3.495955596234316, -2.3927702734928733, 5.566471076822742]]  
#Output weight :  [[1.548733927697386, 6.510381242673692], [-11.140118377599144, -3.7727063153754306], [5.867248065681255, 2.142544636783609], [-12.121627573481403, -1.5558043702193651], [-0.9315840998388218, -5.556284772562421], [13.701150267968393, 1.3862643878204626]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[92]:


# No of Neurons=7 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=100,landing correctly
#Input weight :  [[-8.052933020124476, -1.2277438656189783, -1.3156857916395512, 3.94281058452917, -7.393274603571387, -15.047044108250345, 3.948578397531121], [1.9312720731198554, -1.0087474873025362, -0.977019339063514, -6.614820264929256, 1.4539949456525927, 4.120365823881249, -2.978291181924203]]
#Output weight :  [[-13.550256299443827, -6.289809459457787], [6.315273906591865, 4.104493401393332], [6.306134781298726, 4.823008511161286], [-3.52405789048108, 3.034834108604808], [-12.665475102705441, -2.725679844465681], [13.931104070731937, 2.1774063631298324], [-0.7944510831180605, -6.733867414642709]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[99]:


# No of Neurons=6 ,Lamda = 0.7,Eta = 0.1,alfa=0.7,apoc=100
#Input weight :  [[-8.052933020124476, -1.2277438656189783, -1.3156857916395512, 3.94281058452917, -7.393274603571387, -15.047044108250345, 3.948578397531121], [1.9312720731198554, -1.0087474873025362, -0.977019339063514, -6.614820264929256, 1.4539949456525927, 4.120365823881249, -2.978291181924203]]
#Output weight :  [[-13.550256299443827, -6.289809459457787], [6.315273906591865, 4.104493401393332], [6.306134781298726, 4.823008511161286], [-3.52405789048108, 3.034834108604808], [-12.665475102705441, -2.725679844465681], [13.931104070731937, 2.1774063631298324], [-0.7944510831180605, -6.733867414642709]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[114]:


# No of Neurons=8 ,Lamda = 0.9,Eta = 0.1,alfa=0.3,apoc=100
#Input weight :  [[7.379389411453611, -5.183450935985072, 2.283788877016779, -1.9435126775254565, -2.4069140643925557, -0.23021814015673323, 2.2717689247122483, -2.1644571502255556], [0.8502895070218622, 2.5826975808671313, 4.657351441009394, 1.207390124788686, -5.118327172169598, 3.364165749408477, 4.8849282545639765, 10.106367389521557]]
#Output weight :  [[-7.3861027211615085, 1.2356753005254648], [-4.775363924511397, -1.5376746980157832], [4.430948522292508, 0.11980383553223317], [3.8145014908354873, 4.264235225432788], [-6.696717826028951, -2.4628615153758955], [-1.114500803283735, -4.531585342195014], [8.113880115977146, 0.5850442799775281], [-4.5991962023872235, -0.33261272071590353]]
plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[27]:


# No of Neurons=10 ,Lamda = 0.6,Eta = 0.1,alfa=0.9,apoc=250(Stopping Cretria)
#Input weight :   [[6.664042903727861, 8.96266615148241, 2.444371173798272, -15.618634458482871, -3.0078752589730295, -3.02336707131184, -0.48563832160872267, -3.4178511984294646, -9.181234657459704, -9.010146845559797], [-4.093584301495948, -9.834821509520633, -8.923026979022232, 5.824548152186308, -6.986664828641934, 0.8624863120141291, -3.701604572470313, -6.8970903839876545, 4.164291404801679, 4.161945239063598]]
#Output weight :   [[4.726655027176025, -3.9339504466851123], [-7.643022167419843, -0.08138863505946459], [8.857687038707208, 3.6740674798563067], [14.10452444822674, 0.8145523246051584], [-12.944424783095208, -1.4928738373602188], [2.813221864904043, 5.629180927352492], [7.5042054991788865, 3.200735891270768], [-12.964050897693076, -5.181391618716175], [-9.856774985343284, -3.9800897861145152], [-9.686104606318962, -2.5759481201580514]]

plt.plot(root_mean_sq_err)
plt.plot(validation_root_mean_sq_err)
plt.ylabel("RMSE")
plt.show()


# In[ ]:




