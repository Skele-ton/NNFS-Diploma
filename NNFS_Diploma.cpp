#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

using std::cout;
using std::runtime_error;
using std::size_t;
using std::mt19937;
using std::normal_distribution;
using std::sin;
using std::cos;
using std::exp;

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

    if (outputs.cols != biases.size()) {
        throw runtime_error("layer_forward_batch: outputs.cols must match biases.size()");
    }

    for (size_t i = 0; i < outputs.rows; ++i) {
        for (size_t j = 0; j < outputs.cols; ++j) {
            outputs(i, j) += biases[j];
        }
    }

    return outputs;
}

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

            if (sum == 0.0) {
                throw runtime_error("ActivationSoftmax: sum of exponentials is zero");
            }

            for (size_t j = 0; j < inputs.cols; ++j) {
                output(i, j) /= sum;
            }
        }
    }
};

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

    cout << "Softmax output, first 5 samples:\n";
    for (size_t i = 0; i < 5 && i < activation2.output.rows; ++i) {
        for (size_t j = 0; j < activation2.output.cols; ++j) {
            cout << activation2.output(i, j) << ' ';
        }
        cout << '\n';
    }

    return 0;
}
#endif
