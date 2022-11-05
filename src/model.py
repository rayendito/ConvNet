import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.lr = None
        self.momentum = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def forwardProp(self, inputs):
        self.verbose = False
        if self.verbose:
            print("----------")
        self.inputs = inputs
        x = np.array(self.inputs)

        for layer in self.layers:
            # if Model.checkShape(x, layer):
            x = layer.calculate(x)
            if self.verbose:
                print()
                print(x)
        
        if self.verbose:
            print()
            print(np.round(x))

        return x

    def predict(self, inputs):
        self.outputs = self.forwardProp(inputs)
        if self.verbose:
            print(self.outputs)
        self.preds = np.round(self.outputs)

        if (len(self.preds[0]) == 1):
            self.preds = self.preds.flatten()
        
        return self.preds

    @staticmethod
    def checkShape(x, layer):
        if ("input_shape" in dir(layer)):
            return x.shape == layer.input_shape
        elif ("batch_size" in dir(layer)):
            return len(x) == layer.batch_size
        else:
            return True

    def compile_model(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

    def fit(self, inputs, labels, batch_size=4, epoch=5):
        for i in range(epoch):
            for j in range(0, len(inputs), batch_size):
                begin = j
                end = min(j+batch_size, len(inputs))
                self.forwardProp(inputs[begin:end])

                for i in range(len(self.layers)-1, -1, -1):
                    preced_err_term = None
                    preced_weights = None
                    if(i < len(self.layers)-1):
                        preced_err_term = self.layers[i+1].error_term
                        if(self.layers[i].layer_type == 'Dense'):
                            preced_weights = self.layers[i+1].get_weights()
                        else:
                            preced_weights = self.layers[i+1].get_all_weights()
                    self.layers[i].update_weights(lr=self.lr,
                                                momentum = self.momentum,
                                                actual=labels[begin:end],
                                                preceding_error_term=preced_err_term,
                                                preceding_weights=preced_weights,
                                                preceding_layer_type=self.layers[min(i+1, len(self.layers)-1)].layer_type
                                                )

    def summary(self):
        print("Model Summary:")
        n_params = 0
        currShape = None
        for i,layer in enumerate(self.layers):
            print("layer {} : {}".format(i, layer.layer_type))
            if("outputShape" in dir(layer)):
                currShape = layer.outputShape
            else:
                if(layer.layer_type == 'Flatten'):
                    currShape = (None, np.prod(currShape[1:]))
                elif(layer.layer_type == 'Pooling'):
                    currShape = (None, int(currShape[1]/layer.size), int(currShape[2]/layer.size), currShape[3])
            print("output shape : {}".format(currShape))
            if("param" in dir(layer)):
                n_params += layer.param
                print("param : {}".format(layer.param))
            print("-----------------------")
        print("Total params : {}".format(n_params))
        print("\n")

