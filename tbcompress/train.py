# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import os
import traceback


def train_model(
    model,
    train_loader,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    accuracy_threshold=0.95,
):
    """
    Train a tablebase model

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization parameter
        accuracy_threshold: Early stopping threshold (0.0-1.0)

    Returns:
        Dict containing training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler - reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    print(f"Learning rate scheduler configured with patience={5}")

    # For tracking best performance
    best_train_acc = 0.0

    # Training history
    history = {"train_loss": [], "train_acc": [], "learning_rates": []}

    # Main training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Record statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": train_loss / total, "acc": 100.0 * correct / total}
            )

        # Calculate epoch statistics
        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Update learning rate
        scheduler.step(epoch_train_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Store in history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["learning_rates"].append(current_lr)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            + f"Train Loss: {epoch_train_loss:.4f}, "
            + f"Train Acc: {epoch_train_acc:.4f}, "
            + f"LR: {current_lr:.6f}"
        )

        # Save best model
        if epoch_train_acc > best_train_acc:
            best_train_acc = epoch_train_acc

        # Early stopping
        if epoch_train_acc >= accuracy_threshold:
            print(
                f"Reached accuracy threshold of {accuracy_threshold*100:.1f}%. Early stopping."
            )
            break

    return history


def train_with_streaming_dataset(
    model,
    train_dataset,
    batch_size=64,
    num_training_steps=100000,  # Fixed number of training steps instead of epochs
    learning_rate=0.001,
    weight_decay=1e-5,
    accuracy_threshold=0.95,
    eval_interval=1000,  # How often to evaluate accuracy
    verbose=True,
):
    """
    Train a model with a streaming dataset using a fixed number of training steps

    Args:
        model: The model to train
        train_dataset: StreamingTablebaseDataset for training data
        batch_size: Batch size for training
        num_training_steps: Total number of training steps (batches)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        accuracy_threshold: Early stopping threshold (0.0-1.0)
        eval_interval: How often to evaluate accuracy (in training steps)
        verbose: Whether to print progress

    Returns:
        Dict containing training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use a data loader with the streaming dataset
    # We set shuffle=False since the streaming dataset already provides randomized positions
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Important: Use 0 workers with streaming dataset
        drop_last=True,  # Important: Prevent partial batches
    )

    # Ensure we can get at least one batch
    try:
        # Check that we can get at least one batch
        data_iter = iter(train_loader)
        first_batch = next(data_iter)
        if verbose:
            print(
                f"Successfully loaded first batch with {len(first_batch[0])} examples"
            )
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "train_loss": [],
            "train_acc": [],
            "learning_rates": [],
        }

    # Create a new iterator that will cycle through the dataset
    # When we reach the end, the streaming dataset will generate new positions
    data_iter = iter(train_loader)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler - reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    if verbose:
        print(f"Learning rate scheduler configured with patience=5")

    # For tracking best performance
    best_train_acc = 0.0

    # Training history
    history = {"train_loss": [], "train_acc": [], "learning_rates": []}

    # Initialize metrics
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    step = 0

    # Progress tracking
    if verbose:
        pbar = tqdm(total=num_training_steps, desc="Training")

    # Training loop
    model.train()
    while step < num_training_steps:
        try:
            # Get the next batch, cycling through the dataset
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                # Restart the iterator
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(targets).sum().item()
            batch_total = targets.size(0)
            running_correct += batch_correct
            running_total += batch_total

            # Update progress bar
            if verbose:
                batch_acc = batch_correct / batch_total
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "acc": batch_acc * 100,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                pbar.update(1)

            # Periodically evaluate accuracy and save metrics
            if (step + 1) % eval_interval == 0 or step == num_training_steps - 1:
                # Calculate metrics
                period_loss = running_loss / running_total
                period_acc = running_correct / running_total

                # Store in history
                history["train_loss"].append(period_loss)
                history["train_acc"].append(period_acc)
                history["learning_rates"].append(optimizer.param_groups[0]["lr"])

                # Update learning rate
                scheduler.step(period_acc)

                # Print status
                if verbose:
                    print(
                        f"\nStep {step+1}/{num_training_steps}: "
                        + f"Loss: {period_loss:.4f}, "
                        + f"Acc: {period_acc:.4f}, "
                        + f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

                # Check for early stopping
                if period_acc > best_train_acc:
                    best_train_acc = period_acc

                if period_acc >= accuracy_threshold:
                    if verbose:
                        print(
                            f"Reached accuracy threshold of {accuracy_threshold*100:.1f}%. Early stopping."
                        )
                    break

                # Reset running metrics
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            # Increment step
            step += 1

        except KeyboardInterrupt:
            if verbose:
                print("\nTraining interrupted by user.")
            break
        except Exception as e:
            print(f"\nError during training step {step}: {e}")
            traceback.print_exc()
            # Try to continue with the next batch
            continue

    # Close progress bar
    if verbose:
        pbar.close()

    return history


def train_and_save_model(
    model,
    train_dataset,
    output_path,
    model_name,
    batch_size=64,
    num_epochs=50,
    learning_rate=0.001,
    accuracy_threshold=0.95,
):
    """
    Train a model and save it to disk

    Args:
        model: The model to train
        train_dataset: Dataset containing the training data
        output_path: Directory where model will be saved
        model_name: Name for the saved model (without extension)
        batch_size: Batch size for training
        num_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        accuracy_threshold: Early stopping threshold (0.0-1.0)

    Returns:
        Dict with training metrics and model path
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Start time
    start_time = time.time()

    # Train the model
    print(f"Training model for {model_name}...")
    print(f"Estimated dataset size: {len(train_dataset)} positions")
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Check if we're using a streaming dataset
    if hasattr(train_dataset, "cache_size") and hasattr(train_dataset, "_fill_cache"):
        # This is a streaming dataset, use steps-based training
        # For compression, a large number of steps ensures good coverage
        num_training_steps = 500000  # Aim for 500K steps for good compression
        print(f"Using streaming dataset training with {num_training_steps} steps")

        history = train_with_streaming_dataset(
            model=model,
            train_dataset=train_dataset,
            batch_size=batch_size,
            num_training_steps=num_training_steps,
            learning_rate=learning_rate,
            accuracy_threshold=accuracy_threshold,
        )
    else:
        # Regular dataset, use epoch-based training
        print(f"Using epoch-based training with {num_epochs} epochs")

        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = train_model(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            accuracy_threshold=accuracy_threshold,
        )

    # End time
    training_time = time.time() - start_time
    print(
        f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)"
    )

    # Save the model
    model_path = os.path.join(output_path, model_name)
    print(f"Saving model to {model_path}.pth")

    # Save PyTorch model
    torch.save(model.state_dict(), f"{model_path}.pth")

    # # Export model to other formats
    # from tbcompress.model import export_model

    # export_model(model, model_path)

    # Get final accuracy from history
    final_accuracy = (
        history.get("train_acc", [0.0])[-1] if "train_acc" in history else 0.0
    )

    # Return metrics
    return {
        "model_name": model_name,
        "training_time": training_time,
        "final_accuracy": final_accuracy,
        "model_path": model_path,
        "parameters": model.get_num_parameters(),
    }
