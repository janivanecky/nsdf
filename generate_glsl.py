import argparse
import torch

import models


def _vec4(v):
    return f"vec4({v[0]}, {v[1]}, {v[2]}, {v[3]})"


def _mat3x4(m):
    return f"mat3x4({_vec4(m[0])}, {_vec4(m[1])}, {_vec4(m[2])})"


def _mat4x4(m):
    return f"mat4x4({_vec4(m[0])}, {_vec4(m[1])}, {_vec4(m[2])}, {_vec4(m[3])})"


def _float(v):
    return f"{v}"


def _get_fourier_projection_statement(name, matrix, input_var, alpha):
    return f"vec4 {name} = {matrix} * {input_var} * {alpha};"


def _get_vec_projection_statement(name, matrix, input_var):
    return f"vec4 {name} = {matrix} * {input_var};"


def _get_dot_projection_statement(name, matrix, input_var):
    return f"float {name} = dot({matrix}, {input_var});"


def _get_activation_statement(name, input_var, activation_type="relu"):
    if activation_type == "relu":
        return f"vec4 {name} = max({input_var}, 0.0);"
    elif activation_type == "sigmoid":
        return f"vec4 {name} = 1.0f / (1.0f + exp(-({input_var})));"
    else:
        raise ValueError("Uncreognized activation type.")


def serialize_fourier_features(mapping, alpha, input_var, idx):
    # Return objects initialization.
    sin_vars, cos_vars = [], []
    statements = []

    output_submatrix_count = mapping.shape[1] // 4
    for o in range(output_submatrix_count):
        # Set up output variable names.
        sin_var_name = f"x_{idx}_{o}"
        cos_var_name = f"x_{idx}_{o + output_submatrix_count}"

        # Create matmul statement.
        m = mapping[:, o * 4 : (o + 1) * 4]  # 3x4
        projection = _get_fourier_projection_statement(
            f"x_e_{o}",
            _mat3x4(m),
            input_var,
            alpha * 2.0 * torch.pi,
        )
        sin_output = f"vec4 {sin_var_name} = sin(x_e_{o});"
        cos_output = f"vec4 {cos_var_name} = cos(x_e_{o});"

        # Update output vars.
        statements += [projection, sin_output, cos_output]
        sin_vars += [sin_var_name]
        cos_vars += [cos_var_name]

    # Concatenate sine and cosine output vars.
    output_vars = sin_vars + cos_vars
    return output_vars, statements


def serialize_4x4_layer(weight, bias, input_vars, idx, activation_type="relu"):
    # Return objects initialization.
    output_vars, statements = [], []

    output_submatrices_count = weight.shape[0] // 4
    for o in range(output_submatrices_count):
        submatrix_outputs = []
        for i, input_var in enumerate(input_vars):
            # Set up variable names.
            submatrix_output_name = f"x_{idx}_{o}_{i}"

            # Create matmul statement.
            m = weight[o * 4 : (o + 1) * 4, i * 4 : (i + 1) * 4]
            projection_statement = _get_vec_projection_statement(
                submatrix_output_name, _mat4x4(m.transpose(1, 0)), input_var
            )

            # Store outputs.
            statements += [projection_statement]
            submatrix_outputs += [submatrix_output_name]

        # Create activation statement.
        b = bias[o * 4 : (o + 1) * 4]
        pre_activation_sum = "+".join(submatrix_outputs + [_vec4(b)])
        output_name = f"x_{idx}_{o}"
        activation_statement = _get_activation_statement(
            output_name, pre_activation_sum, activation_type
        )

        # Store outputs.
        statements += [activation_statement]
        output_vars += [output_name]

    return output_vars, statements


def serialize_4x1_layer(weight, bias, input_vars, idx):
    # Return objects initialization.
    statements = []

    submatrix_outputs = []
    for i, input_var in enumerate(input_vars):
        # Set up variable names.
        submatrix_output_name = f"x_{idx}_0_{i}"

        # Create matmul statement.
        m = weight[0, i * 4 : (i + 1) * 4]
        projection = _get_dot_projection_statement(
            submatrix_output_name, _vec4(m), input_var
        )

        # Store outputs.
        statements += [projection]
        submatrix_outputs += [submatrix_output_name]

    # Create output sum statement.
    b = bias[0]
    output_sum = "+".join(submatrix_outputs + [_float(b)])
    output_name = f"x_{idx}"
    output_statement = f"float {output_name} = {output_sum};"

    statements += [output_statement]
    return [output_name], statements


def get_glsl_string(model):
    # Get lists of weights and biases.
    weights = [v.detach().numpy() for k, v in model.named_parameters() if "weight" in k]
    biases = [v.detach().numpy() for k, v in model.named_parameters() if "bias" in k]
    assert len(weights) == len(
        biases
    ), "Invalid model used. Number of weights and biases is not the same."

    # Set activation type from model.
    activation_type = None
    first_activation_layer = model.layers[2]
    if isinstance(first_activation_layer, torch.nn.ReLU):
        activation_type = "relu"
    elif isinstance(first_activation_layer, torch.nn.Sigmoid):
        activation_type = "sigmoid"
    else:
        raise ValueError("Invalid model used. Activation layer is not recognized.")

    # The input layer has to be FourierFeatures layer.
    fourier_layer = model.layers[0]
    assert isinstance(fourier_layer, models.FourierFeatures), "Invalid model used."
    fourier_mapping = fourier_layer.fourier_mappings
    fourier_alpha = fourier_layer.alpha

    statements = []
    module_idx = 0

    # Generate code for FourierFeatures layer.
    output_vars, stmts = serialize_fourier_features(
        fourier_mapping, fourier_alpha, "x", module_idx
    )
    module_idx += 1
    statements += stmts

    # Generate code for hidden layers.
    for w, b in zip(weights[:-1], biases[:-1]):
        output_vars, stmts = serialize_4x4_layer(
            w, b, output_vars, module_idx, activation_type
        )
        module_idx += 1
        statements += stmts

    # Generate code for the output layer.
    output_vars, stmts = serialize_4x1_layer(
        weights[-1], biases[-1], output_vars, module_idx
    )
    statements += stmts

    # Return statement.
    return_statement = f"return {output_vars[0]};"
    statements += [return_statement]

    return statements


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert an SDF model to GLSL code.."
    )
    parser.add_argument("model", help="Model to convert an SDF model to GLSL.")
    parser.add_argument(
        "--separate_lines",
        action="store_true",
        help="If specified, each statement will be put on a separate line.",
    )
    args = parser.parse_args()

    model = torch.load(args.model)
    statements = get_glsl_string(model)

    separator = "\n    " if args.separate_lines else ""

    print("float nsdf(vec3 x) {")
    print("    " + separator.join(statements))
    print("}")


if __name__ == "__main__":
    main()
