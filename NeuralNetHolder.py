import math
import ast
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.IP_weight=    [[6.664042903727861, 8.96266615148241, 2.444371173798272, -15.618634458482871, -3.0078752589730295, -3.02336707131184, -0.48563832160872267, -3.4178511984294646, -9.181234657459704, -9.010146845559797], [-4.093584301495948, -9.834821509520633, -8.923026979022232, 5.824548152186308, -6.986664828641934, 0.8624863120141291, -3.701604572470313, -6.8970903839876545, 4.164291404801679, 4.161945239063598]]  
        self.OP_weight= [[4.726655027176025, -3.9339504466851123], [-7.643022167419843, -0.08138863505946459], [8.857687038707208, 3.6740674798563067], [14.10452444822674, 0.8145523246051584], [-12.944424783095208, -1.4928738373602188], [2.813221864904043, 5.629180927352492], [7.5042054991788865, 3.200735891270768], [-12.964050897693076, -5.181391618716175], [-9.856774985343284, -3.9800897861145152], [-9.686104606318962, -2.5759481201580514]]
        self.Lambda=0.6
        self.x1min=-525.530307
        self.x1max= 524.232533
        self.x2min=65.501678
        self.x2max= 441.681961
        self.y1min= -2.428772
        self.y1max=6.700000
        self.y2min=-3.245970
        self.y2max= 4.181843

        self.no_of_hidden_nodes=10
        self.no_of_inputs=2
        self.no_of_outputs=2
    def activation_fn(self,add):
        AF =  1/(1 + math.exp(-1*(add) * self.Lambda))
        return AF
    
    def weight_multiplication(Self,IP,wt):
    
        add = 0
        for x in range(len(IP)):
        
            prod = IP[x]*wt[x]
            add = add+prod
        return add

    def weight_transpose(self,output, hidden_nodes,inputs):
    
        wt = []
        for x in range(hidden_nodes):
            wt_temp=[]
            for y in range(inputs):
                wt_temp.append(output[y][x])
            wt.append(wt_temp)
        return wt
    

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        #Normalize data
        input_row=ast.literal_eval(input_row)
        input1=(input_row[0]-self.x1min)/(self.x1max-self.x1min)
        input2=(input_row[1]-self.x2min)/(self.x2max-self.x2min)
        input_row=[input1,input2]

        IP_wt_arrange = []
        IP_wt_arrange = self.weight_transpose(self.IP_weight,self.no_of_hidden_nodes,self.no_of_inputs)
        
        ff_1_output = []
        for x in range(self.no_of_hidden_nodes):
            ff_1_output.append(self.activation_fn(self.weight_multiplication(input_row,IP_wt_arrange[x])))
        
            
         #calculation of feedforward values from hidden layer to output layer
        
        OP_wt_arrange = []
        OP_wt_arrange = self.weight_transpose(self.OP_weight,self.no_of_outputs,self.no_of_hidden_nodes)
        
        ff_2_output = []
        for x in range(self.no_of_outputs):
            
            ff_2_output.append(self.activation_fn(self.weight_multiplication(ff_1_output,OP_wt_arrange[x])))

        #Denormalize Data
        ff_2_output[0]=(ff_2_output[0]*(self.y2max-self.y2min)+self.y2min)
        ff_2_output[1]=(ff_2_output[1]*(self.y1max-self.y1min)+self.y1min)
        ff_2_output=[ff_2_output[0],ff_2_output[1]]
        return ff_2_output
