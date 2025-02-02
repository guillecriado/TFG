import networkx as nx
from keras.src.models import Sequential
from keras.src.layers import Dense


class NeuronalNetworkX:
    def __init__(self, numImputs, numOutputs):
        self.nxg = nx.DiGraph()
        self.numInputNeuron=numImputs
        self.numOutputNeuron=numOutputs
        self.nodeID: int =0
        self.model=Sequential()

    def add_node(self, quantity):
        node_id=self.nodeID
        self.nodeID+=1 # Incrementamos el ID para el siguiente nodo
        label=str("ID:"+str(node_id)+"Neurons:"+str(quantity)) # Será la cantidad de neuronas que almacena el nodoy su id
        color=self.color_assignment(quantity) # Dependiendo de las neuronas que almacene será de un color u otro
        self.nxg.add_node(node_id,label=label,color=color)

    def add_edge(self,source,target):
        self.nxg.add_edge(source,target)

    def remove_node(self,node_id):
        if node_id in self.nxg:
            self.nxg.remove_node(node_id)
            return True
        else:
            return False

    def remove_edge(self,source,target):
        if self.nxg.has_edge(source,target):
            self.nxg.remove_edge(source,target)
            return True
        else:
            return False


    def color_assignment(self, quantity):
        color='black'
        if quantity == '':
            color = 'blue'
        elif 1 < int(quantity) < 10:
            color = 'red'
        elif int(quantity) >= 10:
            color = 'green'
        return color

    def isFullyConnectedTopDown(self):
        nxg_aux=self.nxg.copy()
        fullyConnected = True
        index=1
        layer_list = list(nxg_aux.successors(0))
        print("Sucesores del nodo 0: "+str(layer_list))
        size_next_layer=len(layer_list)
        print("Size de la siguiente capa: "+str(size_next_layer))
        while index<self.numInputNeuron and fullyConnected==True:
            print("Index: "+str(index)+ "NumInputNeuron: "+str(self.numInputNeuron))
            aux_list=list(nxg_aux.successors(index))
            print("Sucesores del nodo "+str(index)+ ": " + str(aux_list))
            if set(layer_list)==set(aux_list):
                index+=1
            else:
                fullyConnected=False
        if not fullyConnected:
            return fullyConnected
        else:
            if len(layer_list) > 0:
                return self.isFullyConnectedTopDownAux(nxg_aux, size_next_layer,  layer_list)
            else:
                return fullyConnected

    def isFullyConnectedTopDownAux(self,nxg_aux,size,layer_list):
        fullyConnected=True
        next_layer_list=list(nxg_aux.successors(layer_list[0]))
        print("Sucesores del nodo " + str(layer_list[0]) + ": " + str(next_layer_list))
        i=1
        while i<size and fullyConnected==True:
            print("I: " + str(i) + " Size: " + str(size))
            x=layer_list[i]
            aux_list=list(nxg_aux.successors(x))
            print("Sucesores del nodo " + str(x) + ": " + str(aux_list))
            if set(next_layer_list)==set(aux_list):
                i+=1
            else:
                fullyConnected=False
        if not fullyConnected:
            return fullyConnected
        else:
            if len(next_layer_list)>0:
                return self.isFullyConnectedTopDownAux(nxg_aux, len(next_layer_list), next_layer_list)
            else:
                return fullyConnected

    def isFullyConnectedBottomUp(self): #TODO: Revisar método
        nxg_aux = self.nxg.copy()
        fullyConnected = True
        index = self.numInputNeuron+1
        layer_list = list(nxg_aux.predecessors(0))
        size_next_layer = len(layer_list)
        while index < self.numOutputNeuron and fullyConnected == True:
            aux_list = list(nxg_aux.predecessors(index))
            if set(layer_list) == set(aux_list):
                index += 1
            else:
                fullyConnected = False
        if not fullyConnected:
            return fullyConnected
        else:
            if len(layer_list) > 0:
                return self.isFullyConnectedBottomUpAux(nxg_aux, size_next_layer, layer_list)
            else:
                return fullyConnected

    def isFullyConnectedBottomUpAux(self,nxg_aux,size,layer_list):
        fullyConnected = True
        next_layer_list = list(nxg_aux.predecessors(index=layer_list[0]))
        i = 1
        while i < size and fullyConnected == True:
            x = layer_list[i]
            aux_list = list(nxg_aux.predecessors(x))
            if set(next_layer_list) == set(aux_list):
                i += 1
            else:
                fullyConnected = False
        if not fullyConnected:
            return fullyConnected
        else:
            if len(next_layer_list) > 0:
                return self.isFullyConnectedBottomUpAux(nxg_aux, len(next_layer_list), next_layer_list)
            else:
                return fullyConnected

    def defaultNetwork(self):
        for i in range(self.numInputNeuron): # Capa de entrada
            self.add_node(1)
        for i in range(self.numOutputNeuron): # TODO:  Salida Pendiente de desarrollar para clusterización
            self.add_node(1)

        for i in range(2,6):  # Capa oculta
            self.add_node(1)
            for j in range(self.numInputNeuron):
                self.add_edge(j,i)
            for j in range(self.numOutputNeuron):
                self.add_edge(i,1)

    def parseKeras(self): # TODO: Mejorar método para poder meter gráficos complejos.
        nxg_aux = self.nxg.copy()
        hidden_layer=list(nxg_aux.successors(0))
        self.model.add(Dense(self.numInputNeuron, input_dim=self.numInputNeuron, activation='relu'))
        self.__explore_hidden_layers(hidden_layer,nxg_aux)


    def __explore_hidden_layers(self, layer, nxg_aux):
        size_actual_later=len(layer)
        next_layer=list(nxg_aux.sucessors(layer[0])) # Here we suppose that the neuronal network is fully connected because the verification will be done beforehand
        size_next_layer=len(next_layer)
        if size_next_layer >0 : # There is more than 1 hidden layer
            self.model.add(Dense(size_actual_later,activation='relu'))
            self.__explore_hidden_layers(next_layer, nxg_aux)
        else: # Is going to be the output layer
            self.model.add(Dense(size_actual_later,activation='sigmoid'))

