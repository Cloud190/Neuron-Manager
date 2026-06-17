import numpy
import time 

#THESE ARRAYS ARE EXAMPLES FOR HOW TO MAKE THE ARRAYS FOR THE NEURAL NETWORK!!!!!!!
#Connections = numpy.zeros((3,99,99))             # A body , B body , Weight
#Connections.fill(-999)                          # -999 is the number that we miss when masking out empty indices
#Bodies = numpy.zeros((100),dtype= int)       # max 100 bodies total
#LayerLengths = [1,2,2,1,0]                       #first layer doesnt get the bias because its the input


def Make_Bodies(LayerLengths):
    Counter = 0
    for LayerLengthHash in range(len(LayerLengths)):
        
        Bodies[Counter:Counter + LayerLengths[LayerLengthHash]] = LayerLengthHash+1
        Counter += LayerLengths[LayerLengthHash]

    return Bodies


def Make_Connections(Bodies,LayerLengths,Connections):
    for NeuronCounter in range(sum(LayerLengths)):

        NextLayerLength = LayerLengths[Bodies[NeuronCounter]] # finds the length of the layer after this one
        CurrentLayer = Bodies[NeuronCounter]

        NextLayerStart = sum(LayerLengths[0:CurrentLayer])
        NextLayerEnd = sum(LayerLengths[0:CurrentLayer+1])

        
        Connections[0,NeuronCounter,0:NextLayerLength] = NeuronCounter #Adds the origin of the connection the length of the next layer times because we need to make that many connections
        Connections[1,NeuronCounter,0:NextLayerLength] = list(range(NextLayerStart , NextLayerEnd)) # lists the next layer's neurons so it can add them to the current neuron's B proprety
        Connections[2,NeuronCounter,0:NextLayerLength] = numpy.random.randint(-10,10,NextLayerLength)/10 # makes a random weight for each connection
        #print(Connections[0,NeuronCounter,0:NextLayerLength],Connections[1,NeuronCounter,0:NextLayerLength])

    return Connections


def Repurpose_Bodies(Bodies,LayerLengths): # repuposes bodies to store signals rather than layer levels since we dont need those anymore
    LayerBodies = Bodies
    Bodies = numpy.zeros((100),dtype= float)
    Bodies.fill(-999)
    Bodies[0:LayerLengths[0]] = 0
    
    return Bodies,LayerBodies


def Find_Connected_Neurons_And_Weights(Connections,NeuronID):
    
    Neurons = Connections[1,NeuronID,:][Connections[1,NeuronID,:] != -999].astype(int) # this looks wierd but its really just: array[array != 0]
    Weights = Connections[2,NeuronID,:][Connections[2,NeuronID,:] != -999].astype(float) # this also works because I didnt randomize all the weights but only the ones that actually have a neuron attached
    
    return Neurons.tolist(),Weights
    
    
def Add_Signals_Forward_Accordingly(Connections,LayerLengths,Bodies,LayerBodies,Bias):

    Bodies[LayerLengths[0]:sum(LayerLengths)] = 0 #reset all neurons to 0
    
    for NeuronCounter in range(sum(LayerLengths)):
        ConnectedNeurons,ConnectionWeights = Find_Connected_Neurons_And_Weights(Connections,NeuronCounter)

        print(ConnectedNeurons,ConnectionWeights)
        
        if len(ConnectionWeights) != 0:
            MultipliedValues = Bodies[NeuronCounter]*ConnectionWeights[:]
            Bodies[ConnectedNeurons[:]] +=  MultipliedValues[:]# all of the connected neurons get added the current signal inside of the A body times the Weight of the connection goint to them (The bias gets added last) :D (what a long sentece)
            
            if LayerBodies[NeuronCounter] != LayerBodies[NeuronCounter+1]: #if the next layer is different from the current
                Bodies[ConnectedNeurons[:]] += Bias #then we add the bias to the whole next layer (this is crutial that we only do this once or else the biases can pile up and ruin the whole thing)

    return Bodies[sum(LayerLengths)-1] 


def Randomly_Tweak_Weights(Connections,LayerLenghts,LayerBodies):
    for NeuronCounter in range(sum(LayerLenghts)):

        CurrentLayer = LayerBodies[NeuronCounter]-1
        Connections[2,NeuronCounter,0:LayerLenghts[CurrentLayer+1]]  += numpy.random.randint(-1,1,LayerLenghts[CurrentLayer+1])/10

    return Connections

