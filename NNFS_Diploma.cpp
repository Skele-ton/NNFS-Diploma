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

    bool empty() const
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
    if (a.empty() || b.empty()) {
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
    if (m.empty()) {
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

    for (int class_ix = 0; class_ix < classes; ++class_ix) {
        int class_offset = class_ix * samples_per_class;
        for (int i = 0; i < samples_per_class; ++i) {
            double r = static_cast<double>(i) / (samples_per_class - 1);
            double theta = static_cast<double>(class_ix) * 4.0 + r * 4.0;
            theta += random_normal() * 0.2;

            double x = r * sin(theta);
            double y = r * cos(theta);

            int ix = class_offset + i;
            size_t s_ix = static_cast<size_t>(ix);
            X_out(s_ix, 0) = x;
            X_out(s_ix, 1) = y;
            y_out[s_ix] = class_ix;
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

#ifndef NNFS_NO_MAIN
int main()
{
    MatD X;
    VecI y;
    generate_spiral_data(100, 3, X, y);

    LayerDense dense1(2, 3);
    dense1.forward(X);

    cout << "Dense layer output, first 5 samples:\n";
    for (size_t i = 0; i < 5 && i < dense1.output.rows; ++i) {
        for (size_t j = 0; j < dense1.output.cols; ++j) {
            cout << dense1.output(i, j) << ' ';
        }
        cout << '\n';
    }

    return 0;
}
#endif
