import networkx as nx
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.api.optimizers import Adam


class NeuronalNetworkX:

    #Constructor
    def __init__(self, numImputs, numOutputs):
        self.nxg = nx.DiGraph()
        self.numInputNeuron=numImputs
        self.inputNeurons=list()
        self.numOutputNeuron=numOutputs
        self.outputNeurons= list()
        self.nodeID: int =0
        self.model=Sequential()
        self.keras_path="my_model.keras"
        self.loss=''

    #Setters and getters

    def set_NumInputs(self,numImputs):
        self.numInputNeuron=numImputs
    def set_NumOutputs(self,numOutputs):
        self.numOutputNeuron=numOutputs

    def __get_neurons_of_layer(self, layer, nxg_aux):
        """
        Método nos sirve para conseguir el número de neuronas que hay en una capa.
        :param layer: Capa de la que queremos sacar el número de neuronas.
        :param nxg_aux: Objeto NetworkX auxiliar para no modificar el grafo principal.
        :return: Número de neuronas que hay en la capa.
        """
        quantity_list = nxg_aux.nodes(data='neurons')
        return sum(q for id_, q in quantity_list if id_ in layer and q is not None)

    def set_loss(self, problem_type):
        if problem_type == 'BINARY CLASSIFICATION':
            self.loss='BinaryCrossentropy'
        elif problem_type == 'REGRESSION':
            self.loss='MeanSquaredError'
        else:
            self.loss='CategoricalCrossentropy'

    #Class methods
    def add_node(self, quantity, neuron_type):
        node_id=self.nodeID
        self.nodeID+=1 # Incrementamos el ID para el siguiente nodo
        label=str("ID:"+str(node_id)+"\nNeurons:"+str(quantity)+"\nNeuron type: "+str(neuron_type)) # Será la cantidad de neuronas que almacena el nodoy su id
        color=self.__color_assignment(quantity) # Dependiendo de las neuronas que almacene será de un color u otro
        self.nxg.add_node(node_id,label=label,color=color, neurons=quantity)

    def add_edge(self,source,target):
        self.nxg.add_edge(source,target,color="36494E")

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

    def __color_assignment(self, quantity):
        color='B7CECE'
        if quantity == 1:
            color = '#B7CECE'
        elif 1 < int(quantity) < 5:
            color = '#9BAEBC'
        elif 5 < int(quantity) < 10:
            color = '#597081'
        elif int(quantity) >= 10:
            color = '#384955'
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
                return self.__isFullyConnectedTopDownAux(nxg_aux, size_next_layer,  layer_list)
            else:
                return fullyConnected

    def __isFullyConnectedTopDownAux(self,nxg_aux,size,layer_list):
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
                return self.__isFullyConnectedTopDownAux(nxg_aux, len(next_layer_list), next_layer_list)
            else:
                return fullyConnected

    def isFullyConnectedBottomUp(self):
        nxg_aux = self.nxg.copy()
        fullyConnected = True
        index = self.numInputNeuron+1
        layer_list = list(nxg_aux.predecessors(self.numInputNeuron))
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
                return self.__isFullyConnectedBottomUpAux(nxg_aux, size_next_layer, layer_list)
            else:
                return fullyConnected

    def __isFullyConnectedBottomUpAux(self,nxg_aux,size,layer_list):
        fullyConnected = True
        next_layer_list = list(nxg_aux.predecessors(layer_list[0]))
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
                return self.__isFullyConnectedBottomUpAux(nxg_aux, len(next_layer_list), next_layer_list)
            else:
                return fullyConnected

    def defaultNetwork(self, hidden_layers, hidden_neurons):
        print("Default Network Inputs: "+str(self.numInputNeuron)+" Outputs: "+str(self.numOutputNeuron))
        count_inputs=0
        previous_layer=list()
        for i in range(self.numInputNeuron):  # Capa de entrada
            count_inputs+=1
            print("Input: "+str(count_inputs))
            self.inputNeurons.append(self.nodeID)
            self.add_node(1,'Input Neuron')
        for j in range(self.numOutputNeuron):  # Capa de salida
            self.outputNeurons.append(self.nodeID)
            self.add_node(1,'Target Neuron')
        previous_layer = self.inputNeurons
        if hidden_layers>0 and hidden_neurons>0:
            for k in range(hidden_layers): # Capas oculta
                actual_layer=list()
                first_hidden_neuron=self.nodeID
                last_hidden_neuron=first_hidden_neuron+hidden_neurons
                for l in range(first_hidden_neuron, last_hidden_neuron):
                    actual_layer.append(self.nodeID)
                    self.add_node(1,('Hidden Neuron Layer: '+str(k+1)))
                    for m in previous_layer:
                        self.add_edge(m,l)
                previous_layer=actual_layer

            for n in previous_layer:
                for p in self.outputNeurons:
                    self.add_edge(n,p)

    def simpleNetwork(self):
        for i in range(self.numInputNeuron): # Capa de entrada
            self.inputNeurons.append(self.nodeID)
            self.add_node(1,"Input Neuron")
        for j in range(self.numOutputNeuron): # Capa de salida
            self.outputNeurons.append(self.nodeID)
            self.add_node(1,'Target Neuron')

    def parseKeras(self):
        """
        Método para parsear el objeto NetworkX a Keras.
        :return:
        """
        nxg_aux = self.nxg.copy()
        hidden_layer=list(nxg_aux.successors(0))
        self.model.add(Dense(self.numInputNeuron, input_dim=self.numInputNeuron, activation='relu'))
        self.__explore_hidden_layers(hidden_layer,nxg_aux)

    def __explore_hidden_layers(self, layer, nxg_aux):
        """
        Método para explorar las capas ocultas de la red neuronal para su parseo a Keras.
        :param layer: Capa a explorar.
        :param nxg_aux: Objeto de NetworkX auxiliar para no modificar el grafo principal.
        :return: void.
        """
        size_actual_later=self.__get_neurons_of_layer(layer, nxg_aux)
        next_layer=list(nxg_aux.sucessors(layer[0])) # Here we suppose that the neuronal network is fully connected because the verification will be done beforehand
        size_next_layer=len(next_layer)
        if size_next_layer >0 : # There is more than 1 hidden layer
            self.model.add(Dense(size_actual_later,activation='relu'))
            self.__explore_hidden_layers(next_layer, nxg_aux)
        else: # Is going to be the output layer
            self.model.add(Dense(size_actual_later,activation='sigmoid'))

    def save_model(self):
        """
        Este método guardará el modelo de Keras en un archivo .keras .
        :return:
        """
        self.model.save(self.keras_path)

    def train_model(self, train_input, train_target, num_epochs):
        """
        Este método será para entrenar la red neuronal.
        :param train_input: Es la parte del DataFrame que nos sirve de entrada para entrenar.
        :param train_target: Es la parte del DataFrame que nos sirve de salida para entrenar.
        :param num_epochs: Es el número de Epochs para entrenar a la red neuronal.
        :return: No se devuelve nada.
        """
        self.model.compile(loss=self.loss, optimizer=Adam())
        self.model.fit(train_input, train_target, epochs=num_epochs, verbose=0)

    def test_model(self, test_input, test_target):
        """
        Este método será para testear la red neuronal.
        :param test_input: Es la parte del DataFrame que nos sirve de entrada para testear.
        :param test_target: Es la parte del DataFrame que nos sirve de salida para testear.
        :return: los resultados de la evaluación de la red neuronal.
        """
        results = self.model.evaluate(test_input, test_target)
        return results

    def predict(self, input_data):
        """
        Este método nos sirve para hacer las predicciones de la red neuronal.
        :param input_data: Son los datos sobre los que se quiere hacer una predicción.
        :return: La predicción.
        """
        return self.model.predict(input_data)

    def clear_graph(self):
        self.nxg = nx.DiGraph()
        self.inputNeurons = list()  # Reiniciar la lista de neuronas de entrada
        self.outputNeurons = list()  # Reiniciar la lista de neuronas de salida
        self.nodeID = 0 # Reiniciar nodeID si quieres que los IDs empiecen desde 0 cada vez