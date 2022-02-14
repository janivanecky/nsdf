import argparse
import models
import mesh_to_sdf as mts
import numpy as np
import torch
import trimesh


def get_training_data(mesh_path: str, point_count: int):
    uniform_point_count = point_count // 2
    surface_point_count = point_count // 2

    # SDF computation parameters.
    surface_point_method = "scan"
    scan_count = 100
    scan_resolution = 400
    sample_point_count = 10000000
    sign_method = "depth"
    normal_sample_count = 11
    min_size = 0.0
    return_gradients = False

    # Load the mesh and scale it so it fits into unit sphere.
    mesh = trimesh.load(mesh_path)
    mesh = mts.scale_to_unit_sphere(mesh)

    # Get SDF for points near the mesh surface and uniform points in the unit cube.
    surface_point_cloud = mts.get_surface_point_cloud(
        mesh,
        surface_point_method,
        1,
        scan_count,
        scan_resolution,
        sample_point_count,
        sign_method == "normal",
    )
    points_surface, sdf_surface = surface_point_cloud.sample_sdf_near_surface(
        surface_point_count,
        surface_point_method == "scan",
        sign_method,
        normal_sample_count,
        min_size,
        return_gradients,
    )
    points_uniform = np.random.uniform(-1, 1, size=(uniform_point_count, 3))
    sdf_uniform = surface_point_cloud.get_sdf_in_batches(
        points_uniform,
        use_depth_buffer=sign_method == "depth",
        sample_count=normal_sample_count,
        return_gradients=return_gradients,
    )

    # Merge uniform and near-surface points and sdfs.
    points = np.concatenate([points_surface, points_uniform], axis=0)
    sdf = np.concatenate([sdf_surface, sdf_uniform], axis=0)

    return points, sdf


def shuffle_data(points, sdf):
    indices = np.arange(len(points))
    np.random.shuffle(indices)
    points = points[indices]
    sdf = sdf[indices]
    return points, sdf


def get_current_batch(batch_idx, batch_size, points, sdf):
    batch_size_aligned_length = len(points) // batch_size * batch_size
    batch_start = (batch_idx * batch_size) % (batch_size_aligned_length)
    batch_end = batch_start + batch_size
    return points[batch_start:batch_end], sdf[batch_start:batch_end]


def get_model_parameters(args):
    # Set parameters from templates.
    model_parameters = {
        "small": (16, 32, 1),
        "normal": (64, 64, 1),
        "bigly": (96, 96, 0),
    }
    activation_type_from_style = {
        "smooth": "sigmoid",
        "sharp": "relu",
    }
    size, embedding_size, hidden_layers = model_parameters[args.model_size]
    activation_type = activation_type_from_style[args.style]

    # Override specific values.
    if args.size:
        size = args.size
    if args.embedding_size:
        embedding_size = args.embedding_size
    if args.hidden_layers:
        hidden_layers = args.hidden_layers
    if args.activation_type:
        activation_type = args.activation_type

    return size, embedding_size, hidden_layers, activation_type


def main(args=None):
    parser = argparse.ArgumentParser(description="Script to train an SDF model.")
    parser.add_argument("mesh", help="Mesh to capture.")
    parser.add_argument("--output", help="Path where the model will be saved.")
    parser.add_argument(
        "--steps", type=int, default=600000, help="Number of training steps."
    )

    # Simple options.
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "normal", "bigly"],
        default="normal",
        help="Model size to use. Normal is a sensible default.",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["smooth", "sharp"],
        default="sharp",
        help="Model style.",
    )

    # Options for power users.
    parser.add_argument("--size", type=int, help="Size of hidden layers.")
    parser.add_argument("--embedding_size", type=int, help="Size of fourier embeddings")
    parser.add_argument("--hidden_layers", type=int, help="Number of hidden layers.")
    parser.add_argument(
        "--activation_type",
        type=str,
        choices=["relu", "sigmoid"],
        help="Type of activation fucntion.",
    )

    args = parser.parse_args(args)
    size, embedding_size, hidden_layers, activation_type = get_model_parameters(args)

    # Swap this to `cuda:0` if you want to use GPU.
    device = torch.device("cpu:0")

    # Set up model.
    model = models.SDFDNN(
        size,
        embedding_size,
        hidden_layers=hidden_layers,
        activation_type=activation_type,
    ).to(device)

    # Set up optimizer.
    batch_size = 512
    loss_fn = torch.nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Set up training data.
    point_count = 2000000
    points, sdf = get_training_data(args.mesh, point_count)
    points, sdf = shuffle_data(points, sdf)

    for i in range(1, args.steps + 1):
        # Get current batch of training data.
        inputs, targets = get_current_batch(i + 1, batch_size, points, sdf)
        inputs = torch.Tensor(inputs).to(device)
        targets = torch.Tensor(targets).to(device).unsqueeze(1)

        # Compute predictions and loss.
        predictions = model(inputs)
        loss = torch.mean(loss_fn(predictions, targets))

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update LR scheduler.
        if i % (args.steps // 3) == 0:
            scheduler.step()

        # Print progress.
        progress = 100 * i // args.steps
        print_frequency = args.steps // 100
        if i % print_frequency == 0:
            print(f"{progress}% - loss: {loss.item():.6f}")

    # Save the model.
    torch.save(model, args.output)


if __name__ == "__main__":
    main()
