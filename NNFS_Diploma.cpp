#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <limits>

using std::cout;
using std::runtime_error;
using std::size_t;
using std::mt19937;
using std::normal_distribution;
using std::sin;
using std::cos;
using std::exp;
using std::log;

using VecD = std::vector<double>;
using VecI = std::vector<int>;

class Matrix
{
public:
    size_t rows;
    size_t cols;
    VecD data;

    Matrix() : rows(0), cols(0), data() {}

    Matrix(size_t r, size_t c, double value = 0.0)
        : rows(r), cols(c), data(r * c, value) {}

    void assign(size_t r, size_t c, double value = 0.0)
    {
        rows = r;
        cols = c;
        data.assign(r * c, value);
    }

    bool is_empty() const
    {
        return rows == 0 || cols == 0;
    }

    double& operator()(size_t r, size_t c)
    {
        return data[r * cols + c];
    }

    double operator()(size_t r, size_t c) const
    {
        return data[r * cols + c];
    }
};

using MatD = Matrix;

// global RNG
mt19937 g_rng(0);

double random_normal()
{
    static normal_distribution<double> dist(0.0, 1.0);
    return dist(g_rng);
}

// math
MatD matmul(const MatD& a, const MatD& b)
{
    if (a.is_empty() || b.is_empty()) {
        throw runtime_error("matmul: matrices must not be empty");
    }

    if (a.cols != b.rows) {
        throw runtime_error("matmul: incompatible shapes");
    }

    MatD result(a.rows, b.cols, 0.0);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            double aik = a(i, k);
            for (size_t j = 0; j < b.cols; ++j) {
                result(i, j) += aik * b(k, j);
            }
        }
    }

    return result;
}

MatD transpose(const MatD& m)
{
    if (m.is_empty()) {
        return MatD();
    }

    MatD result(m.cols, m.rows);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result(j, i) = m(i, j);
        }
    }

    return result;
}

MatD clip_matrix(const MatD& m, double min_value, double max_value)
{
    if (min_value > max_value) {
        throw runtime_error("clip_matrix: min_value must not exceed max_value");
    }

    MatD result(m.rows, m.cols);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            double v = m(i, j);
            if (v < min_value) {
                v = min_value;
            } else if (v > max_value) {
                v = max_value;
            }
            result(i, j) = v;
        }
    }

    return result;
}

double mean(const VecD& values)
{
    if (values.empty()) {
        throw runtime_error("mean: cannot compute mean of empty vector");
    }

    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }

    return sum / static_cast<double>(values.size());
}

// training data
void generate_spiral_data(int samples_per_class, int classes, MatD& X_out, VecI& y_out)
{
    if (samples_per_class <= 1 || classes <= 0) {
        throw runtime_error("generate_spiral_data: invalid arguments");
    }

    int total_samples = samples_per_class * classes;
    X_out.assign(static_cast<size_t>(total_samples), 2);
    y_out.assign(static_cast<size_t>(total_samples), 0);

    for (int class_idx = 0; class_idx < classes; ++class_idx) {
        int class_offset = class_idx * samples_per_class;
        for (int i = 0; i < samples_per_class; ++i) {
            double r = static_cast<double>(i) / (samples_per_class - 1);
            double theta = static_cast<double>(class_idx) * 4.0 + r * 4.0;
            theta += random_normal() * 0.2;

            double x = r * sin(theta);
            double y = r * cos(theta);

            int idx = class_offset + i;
            size_t s_idx = static_cast<size_t>(idx);
            X_out(s_idx, 0) = x;
            X_out(s_idx, 1) = y;
            y_out[s_idx] = class_idx;
        }
    }
}

// neurons
MatD layer_forward_batch(const MatD& inputs, const MatD& weights, const VecD& biases)
{
    if (weights.rows != biases.size()) {
        throw runtime_error("layer_forward_batch: weights.rows must match biases.size()");
    }

    MatD weights_T = transpose(weights);
    MatD outputs = matmul(inputs, weights_T);

    for (size_t i = 0; i < outputs.rows; ++i) {
        for (size_t j = 0; j < outputs.cols; ++j) {
            outputs(i, j) += biases[j];
        }
    }

    return outputs;
}

// Dense layer with weights/biases and cached inputs/output
class LayerDense
{
public:
    MatD weights;
    VecD biases;
    MatD output;
    MatD inputs;

    LayerDense(size_t n_inputs, size_t n_neurons)
        : weights(n_neurons, n_inputs),
          biases(n_neurons, 0.0),
          output(),
          inputs()
    {
        for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
            for (size_t input = 0; input < n_inputs; ++input) {
                weights(neuron, input) = 0.01 * random_normal();
            }
        }
    }

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output = layer_forward_batch(inputs, weights, biases);
    }
};

// activations
// ReLU activation (forward only)
class ActivationReLU
{
public:
    MatD inputs;
    MatD output;

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);
        for (size_t i = 0; i < inputs.rows; ++i) {
            for (size_t j = 0; j < inputs.cols; ++j) {
                double v = inputs(i, j);
                if (v > 0.0) {
                    output(i, j) = v;
                } else {
                    output(i, j) = 0.0;
                }
            }
        }
    }
};

// Softmax activation (forward only)
class ActivationSoftmax
{
public:
    MatD inputs;
    MatD output;

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output.assign(inputs.rows, inputs.cols);

        for (size_t i = 0; i < inputs.rows; ++i) {
            double max_val = inputs(i, 0);
            for (size_t j = 1; j < inputs.cols; ++j) {
                double v = inputs(i, j);
                if (v > max_val) {
                    max_val = v;
                }
            }

            double sum = 0.0;
            for (size_t j = 0; j < inputs.cols; ++j) {
                double e = exp(inputs(i, j) - max_val);
                output(i, j) = e;
                sum += e;
            }

            if (!std::isfinite(sum) || sum <= 0.0) {
                throw runtime_error("ActivationSoftmax: invalid sum of exponentials");
            }

            for (size_t j = 0; j < inputs.cols; ++j) {
                output(i, j) /= sum;
            }
        }
    }
};

// loss
class Loss
{
public:
    virtual ~Loss() = default;

    double calculate(const MatD& output, const VecI& y_true) const
    {
        VecD sample_losses = forward(output, y_true);
        return mean(sample_losses);
    }

    double calculate(const MatD& output, const MatD& y_true) const
    {
        VecD sample_losses = forward(output, y_true);
        return mean(sample_losses);
    }

protected: 
    virtual VecD forward(const MatD& output, const VecI& y_true) const = 0;
    virtual VecD forward(const MatD& output, const MatD& y_true) const = 0;
};

class LossCategoricalCrossEntropy : public Loss
{
public:
    // sparse labels
    VecD forward(const MatD& y_pred, const VecI& y_true) const override
    {
        if (y_pred.rows != y_true.size()) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred.rows must match y_true.size()");
        }

        MatD clipped = clip_matrix(y_pred, 1e-7, 1.0 - 1e-7);
        VecD losses(y_pred.rows, 0.0);

        for (size_t i = 0; i < y_pred.rows; ++i) {
            int class_idx = y_true[i];
            if (class_idx < 0 || static_cast<size_t>(class_idx) >= y_pred.cols) {
                throw runtime_error("LossCategoricalCrossEntropy: class index out of range");
            }

            double confidence = clipped(i, static_cast<size_t>(class_idx));
            losses[i] = -log(confidence);
        }

        return losses;
    }

    // one-hot labels path
    VecD forward(const MatD& y_pred, const MatD& y_true) const override
    {
        if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
            throw runtime_error("LossCategoricalCrossEntropy: y_pred and y_true must have the same shape");
        }

        MatD clipped = clip_matrix(y_pred, 1e-7, 1.0 - 1e-7);
        VecD losses(y_pred.rows, 0.0);

        for (size_t i = 0; i < y_pred.rows; ++i) {
            double confidence = 0.0;
            for (size_t j = 0; j < y_pred.cols; ++j) {
                confidence += clipped(i, j) * y_true(i, j);
            }

            losses[i] = -log(confidence);
        }

        return losses;
    }
};

// accuracy
// for sparse integer labels
double classification_accuracy(const MatD& y_pred, const VecI& y_true)
{
    if (y_pred.rows != y_true.size()) {
        throw runtime_error("classification_accuracy: y_pred.rows must match y_true.size()");
    }
    if (y_pred.rows == 0 || y_pred.cols == 0) {
        throw runtime_error("classification_accuracy: y_pred must be non-empty");
    }

    size_t correct = 0;
    for (size_t i = 0; i < y_pred.rows; ++i) {
        size_t pred_class = 0;
        double max_pred = y_pred(i, 0);
        for (size_t j = 1; j < y_pred.cols; ++j) {
            double v = y_pred(i, j);
            if (v > max_pred) {
                max_pred = v;
                pred_class = j;
            }
        }

        if (pred_class == static_cast<size_t>(y_true[i])) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(y_pred.rows);
}

// for one-hot labels (argmax on both)
double classification_accuracy(const MatD& y_pred, const MatD& y_true)
{
    if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
        throw runtime_error("classification_accuracy: y_pred and y_true must have the same shape");
    }
    if (y_pred.rows == 0 || y_pred.cols == 0) {
        throw runtime_error("classification_accuracy: y_pred must be non-empty");
    }

    size_t correct = 0;
    for (size_t i = 0; i < y_pred.rows; ++i) {
        size_t pred_class = 0;
        double max_pred = y_pred(i, 0);
        for (size_t j = 1; j < y_pred.cols; ++j) {
            double v = y_pred(i, j);
            if (v > max_pred) {
                max_pred = v;
                pred_class = j;
            }
        }

        size_t true_class = 0;
        double max_true = y_true(i, 0);
        for (size_t j = 1; j < y_true.cols; ++j) {
            double v = y_true(i, j);
            if (v > max_true) {
                max_true = v;
                true_class = j;
                if (max_true == 1.0) {
                    break;
                }
            }
        }

        if (pred_class == true_class) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(y_pred.rows);
}

#ifndef NNFS_NO_MAIN
int main()
{
    MatD X;
    VecI y;
    generate_spiral_data(100, 3, X, y);

    LayerDense dense1(2, 3);
    ActivationReLU activation1;
    LayerDense dense2(3, 3);
    ActivationSoftmax activation2;

    dense1.forward(X);
    activation1.forward(dense1.output);

    dense2.forward(activation1.output);
    activation2.forward(dense2.output);

    LossCategoricalCrossEntropy loss_function;
    double loss = loss_function.calculate(activation2.output, y);
    double acc = classification_accuracy(activation2.output, y);

    cout << "Softmax output, first 5 samples:\n";
    for (size_t i = 0; i < 5 && i < activation2.output.rows; ++i) {
        for (size_t j = 0; j < activation2.output.cols; ++j) {
            cout << activation2.output(i, j) << ' ';
        }
        cout << '\n';
    }

    cout << "Loss: " << loss << '\n';
    cout << "Accuracy: " << acc << '\n';

    return 0;
}
#endif
