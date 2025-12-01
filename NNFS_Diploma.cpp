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
using MatD = std::vector<std::vector<double>>;
using VecI = std::vector<int>;

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

    size_t a_rows = a.size();
    size_t a_cols = a[0].size();
    size_t b_rows = b.size();
    size_t b_cols = b[0].size();

    for (const VecD& row : a) {
        if (row.size() != a_cols) {
            throw runtime_error("matmul: left matrix is not rectangular");
        }
    }

    for (const VecD& row : b) {
        if (row.size() != b_cols) {
            throw runtime_error("matmul: right matrix is not rectangular");
        }
    }

    if (a_cols != b_rows) {
        throw runtime_error("matmul: incompatible shapes");
    }

    MatD result(a_rows, VecD(b_cols, 0.0));

    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t k = 0; k < a_cols; ++k) {
            double aik = a[i][k];
            for (size_t j = 0; j < b_cols; ++j) {
                result[i][j] += aik * b[k][j];
            }
        }
    }

    return result;
}

MatD transpose(const MatD& m)
{
    if (m.empty()) {
        return MatD{};
    }

    size_t rows = m.size();
    size_t cols = m[0].size();

    for (const VecD& row : m) {
        if (row.size() != cols) {
            throw runtime_error("transpose: matrix is not rectangular");
        }
    }

    MatD result(cols, VecD(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = m[i][j];
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
    X_out.assign(static_cast<size_t>(total_samples), VecD(2));
    y_out.assign(static_cast<size_t>(total_samples), 0);

    for (int class_ix = 0; class_ix < classes; ++class_ix) {
        int class_offset = class_ix * samples_per_class;
        for (int i = 0; i < samples_per_class; ++i) {
            double r = static_cast<double>(i) / (samples_per_class - 1); // 0..1
            double theta = static_cast<double>(class_ix) * 4.0 + r * 4.0;
            theta += random_normal() * 0.2;

            double x = r * sin(theta);
            double y = r * cos(theta);

            int ix = class_offset + i;
            X_out[static_cast<size_t>(ix)][0] = x;
            X_out[static_cast<size_t>(ix)][1] = y;
            y_out[static_cast<size_t>(ix)] = class_ix;
        }
    }
}

// neurons
MatD layer_forward_batch(const MatD& inputs, const MatD& weights, const VecD& biases)
{
    if (weights.size() != biases.size()) {
        throw runtime_error("layer_forward_batch: weights.size() must match biases.size()");
    }

    MatD weights_T = transpose(weights);
    MatD outputs = matmul(inputs, weights_T);

    for (VecD& row : outputs) {
        if (row.size() != biases.size()) {
            throw runtime_error("layer_forward_batch: outputs columns must match biases.size()");
        }
        for (size_t j = 0; j < row.size(); ++j) {
            row[j] += biases[j];
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
        : weights(n_neurons, VecD(n_inputs)),
          biases(n_neurons, 0.0)
    {
        for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
            for (size_t input = 0; input < n_inputs; ++input) {
                weights[neuron][input] = 0.01 * random_normal();
            }
        }
    }

    void forward(const MatD& inputs_batch)
    {
        inputs = inputs_batch;
        output = layer_forward_batch(inputs, weights, biases);
    }
};

int main()
{
    MatD X;
    VecI y;
    generate_spiral_data(100, 3, X, y);

    LayerDense dense1(2, 3);
    dense1.forward(X);

    cout << "Dense layer output, first 5 samples:\n"; // 5 rows x 3 values
    for (size_t i = 0; i < 5 && i < dense1.output.size(); ++i) {
        const VecD& row = dense1.output[i];
        for (double v : row) {
            cout << v << ' ';
        }
        cout << '\n';
    }

    return 0;
}
