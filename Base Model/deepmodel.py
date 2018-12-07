from audio_utils import *

class deeperModel:

    def __init__(self, usingKickStart, content_tf, style_tf, N_BINS):

        self.usingKickStart = usingKickStart  # True if saved model is available else train and save model
        self.content_tf = content_tf
        self.style_tf = style_tf
        self.N_BINS = N_BINS
        self.N_FILTERS = [128, 256, 512]
        self.styleWeight = 1 #Hyper parameter for total cost
        self.contentWeight = 0.1 #Hyper parameter for total cost
        self.learning_rate = 1e-3
        self.saver = None
        self.contentInitialization = False


    def initializeWeights(self):
        # Initialize parameters for three hidden layers for the model (W1, W2, W3)

        N_FILTERS1 = self.N_FILTERS[0]
        N_FILTERS2 = self.N_FILTERS[1]
        N_FILTERS3 = self.N_FILTERS[2]
        filterLength = 11

        std1 = np.sqrt(2) * np.sqrt(2.0 / ((self.N_BINS + N_FILTERS1) * filterLength))
        W1_init = np.random.randn(1, filterLength, self.N_BINS, N_FILTERS1) * std1
        W1 = tf.constant(W1_init, name='W1', dtype=tf.float32)

        std2 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS1 + N_FILTERS2) * filterLength))
        W2_init = np.random.randn(1, filterLength, N_FILTERS1, N_FILTERS2) * std2
        W2 = tf.constant(W2_init, name='W2', dtype=tf.float32)

        std3 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS2 + N_FILTERS3) * filterLength))
        W3_init = np.random.randn(1, filterLength, N_FILTERS2, N_FILTERS3) * std3
        W3 = tf.constant(W3_init, name='W3', dtype=tf.float32)

        return W1, W2, W3


    # this function gets the style loss for model which is l2 loss between gram mtrices of style and output audios
    def getStyleLoss(self, activationsList):

        styleLoss = 0

        for A_OUT, A_style in activationsList:
            squeezeStyle = tf.squeeze(A_style)
            squeezeOut = tf.squeeze(A_OUT)
            s2 = tf.transpose(squeezeStyle)
            o2 = tf.transpose(squeezeOut)
            Gs = tf.matmul(s2, tf.transpose(s2))
            Go = tf.matmul(o2, tf.transpose(o2))
            vectorStyle = tf.reshape(squeezeStyle, [-1, 1])
            vectorOut = tf.reshape(squeezeOut, [-1, 1])
            styleGramMatrix = tf.matmul(tf.transpose(vectorStyle), vectorStyle)
            outGramMatrix = tf.matmul(tf.transpose(vectorOut), vectorOut)
            styleGramMatrix = Gs
            outGramMatrix = Go
            styleLoss += 2 * tf.nn.l2_loss(styleGramMatrix - outGramMatrix)

            return styleLoss / float(len(activationsList))

    # this function gets the cost tensor for model 1. Total cost  = styleWeight * styleLoss + contentWeight * contentLoss
    def getCostTensor(self, Weights, OUT):
        W1, W2, W3 = Weights

        Z1_OUT = tf.nn.conv2d(OUT, W1, strides=[1, 1, 1, 1], padding='SAME')
        # note: input tensor is shape [batch, in_height, in_width, in_channels]
        # kernel is shape [filter_height, filter_width, in_channels, out_channels]
        # stride according to dimensions of input

        A1_OUT = tf.nn.relu(Z1_OUT)

        Z2_OUT = tf.nn.conv2d(A1_OUT, W2, strides=[1, 1, 1, 1], padding='SAME')
        A2_OUT = tf.nn.relu(Z2_OUT)

        Z3_OUT = tf.nn.conv2d(A2_OUT, W3, strides=[1, 1, 1, 1], padding='SAME')
        A3_OUT = tf.nn.relu(Z3_OUT)

        Z1_content = tf.nn.conv2d(self.content_tf, W1, strides=[1, 1, 1, 1], padding='SAME')
        A1_content = tf.nn.relu(Z1_content)
        Z1_style = tf.nn.conv2d(self.style_tf, W1, strides=[1, 1, 1, 1], padding='SAME')
        A1_style = tf.nn.relu(Z1_style)

        Z2_content = tf.nn.conv2d(A1_content, W2, strides=[1, 1, 1, 1], padding='SAME')
        A2_content = tf.nn.relu(Z2_content)
        Z2_style = tf.nn.conv2d(A1_style, W2, strides=[1, 1, 1, 1], padding='SAME')
        A2_style = tf.nn.relu(Z2_style)

        Z3_content = tf.nn.conv2d(A2_content, W3, strides=[1, 1, 1, 1], padding='SAME')
        A3_content = tf.nn.relu(Z3_content)
        Z3_style = tf.nn.conv2d(A2_style, W3, strides=[1, 1, 1, 1], padding='SAME')
        A3_style = tf.nn.relu(Z3_style)

        # Style Loss is calculated using gram matrix which measures the covariance between activations of different channels and l2 loss of two gram matrices
        styleLoss = self.getStyleLoss([(A1_OUT, A1_style), (A2_OUT, A2_style), (A3_OUT, A3_style)])

        contentLoss = 2 * tf.nn.l2_loss(A3_OUT - A3_content)

        cost = self.styleWeight * styleLoss + self.contentWeight * contentLoss

        return cost

    #Initilaize random noise
    def initializeOutput(self):
        OUT = None

        if self.contentInitialization:
            print("using content as initialization of output")
            OUT = tf.get_variable("OUT", initializer=self.content_tf, dtype=tf.float32)
        else:
            print("using random initialization of output")
            shape = self.content_tf.get_shape().as_list()
            print("shape of content_tf = ", shape)
            init = np.random.randn(shape[0], shape[1], shape[2], shape[3]) * 1e-3
            OUT = tf.get_variable("OUT", initializer=tf.constant(init, dtype=tf.float32), dtype=tf.float32)

        return OUT

    # this function implements model and writes the output to out.wav
    def getWeightsAndOutput(self):

        evalW, evalOut = None, None
        if self.usingKickStart:

            meta_file = 'savedModels/deeperModel/model.meta'
            checkpoint_directory = 'savedModels/deeperModel/'

            with tf.Session() as session:
                saver = tf.train.import_meta_graph(meta_file)
                saver.restore(session, tf.train.latest_checkpoint(checkpoint_directory))
                graph = tf.get_default_graph()
                savedOUT = graph.get_tensor_by_name("OUT:0")
                savedW = graph.get_tensor_by_name("W:0")
                evalW = session.run(savedW)
                evalOut = session.run(savedOUT)
                OUT = tf.get_variable("OUT", initializer=tf.constant(evalOut, dtype=tf.float32))
                return OUT
        else:
            OUT = self.initializeOutput()
            return OUT


    def synthesize(self):

        OUT = self.getWeightsAndOutput()
        print("Shape of OUT is {} ".format(OUT.shape))
        Weights = self.initializeWeights()
        print("Shape of Weight 1 is {} ".format(Weights[0].shape))
        print("Shape of Weight 2 is {} ".format(Weights[1].shape))
        print("Shape of Weight 3 is {} ".format(Weights[2].shape))

        cost = self.getCostTensor(Weights, OUT)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        costs = []
        shortCosts = []
        result = None

        self.saver = tf.train.Saver(max_to_keep=4)

        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)

            self.saver.save(session, "savedModels/deeperModel/model")

            for iteration in range(5000):
                print("Iteration : {}".format(iteration))
                _, currCost = session.run([optimizer, cost], feed_dict={})

                print("Current Cost : {}".format(currCost))
                costs.append(currCost)

                if currCost < 1e-6:
                    break

                if (iteration % 500) == 0:
                    print("saved model")
                    self.saver.save(session, 'savedModels/deeperModel/model', write_meta_graph=False)

                    shortCosts.append(currCost)
                    writeListToFile(shortCosts, "shortCosts.txt")
                    writeListToFile(costs, "costs.txt")

                    result = session.run(OUT)

        return result

