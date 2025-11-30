#include <iostream>
#include <vector>
#include <stdexcept>

using std::cout;
using std::runtime_error;
using std::size_t;

using VecD = std::vector<double>;
using MatD = std::vector<std::vector<double>>;

double dot(const VecD& a, const VecD& b)
{
    if (a.size() != b.size()) {
        throw runtime_error("dot: vectors must have the same size");
    }

    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

VecD dot(const MatD& m, const VecD& v)
{
    VecD result;
    result.reserve(m.size());

    for (const VecD& row : m) {
        result.push_back(dot(row, v));
    }

    return result;
}

double single_neuron_forward(const VecD& inputs,
                             const VecD& weights,
                             double bias)
{
    return dot(weights, inputs) + bias;
}

VecD layer_forward(const VecD& inputs,
                   const MatD& weights,
                   const VecD& biases)
{
    if (weights.size() != biases.size()) {
        throw runtime_error("weights.size() must match biases.size()");
    }

    VecD outputs = dot(weights, inputs);

    if (outputs.size() != biases.size()) {
        throw runtime_error("outputs.size() must match biases.size()");
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] += biases[i];
    }

    return outputs;
}

int main() {
    VecD inputs = {1.0, 2.0, 3.0, 2.5};

    MatD weights = {
        {0.2,   0.8,  -0.5,  1.0},
        {0.5,  -0.91,  0.26, -0.5},
        {-0.26, -0.27, 0.17,  0.87}
    };

    VecD biases = {
        2.0,
        3.0,
        0.5
    };

    VecD outputs = layer_forward(inputs, weights, biases);

    cout << "Layer output: [ ";
    for (double o : outputs) {
        cout << o << ' ';
    }
    cout << "]\n";

    return 0;
}
