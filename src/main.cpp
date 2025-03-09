class DenseLayer : public BaseLayer {
    public:
        Matrix weights, biases, input, output;
        
        DenseLayer(int input_size, int output_size) 
            : weights(input_size, output_size), biases(1, output_size) {
            weights.randomize();
            biases.randomize();
        }
    
        void forward(Matrix &input) override {
            this->input = input;
            output = (input * weights) + biases;
            output.applyFunction(sigmoid);
        }
    
        void backward(Matrix &error, double learning_rate) override {
            Matrix d_output = error * output.applyFunction(sigmoid_derivative);
            Matrix d_weights = input.transpose() * d_output;
            weights -= d_weights * learning_rate;
            biases -= d_output * learning_rate;
        }
    };

    class NeuralNetwork {
        public:
            vector<BaseLayer*> layers;
        
            void addLayer(BaseLayer* layer) {
                layers.push_back(layer);
            }
        
            Matrix forward(Matrix input) {
                for (auto layer : layers) {
                    layer->forward(input);
                    input = layer->output; // Pass output as next layer's input
                }
                return input;
            }
        
            void train(Matrix &input, Matrix &target, int epochs, double learning_rate) {
                for (int i = 0; i < epochs; i++) {
                    Matrix output = forward(input);
                    Matrix error = target - output;
                    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                        (*it)->backward(error, learning_rate);
                    }
                }
            }
        };
        


// NeuralNetwork nn;
// nn.addLayer(new DenseLayer(784, 128));  // Input to Hidden
// nn.addLayer(new DenseLayer(128, 10));   // Hidden to Output
