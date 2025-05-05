import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tbcompress.model import TablebaseModel
from tbcompress.streaming_dataset import StreamingTablebaseDataset


def find_model_and_rtbw_pairs(models_dir, rtbw_dir):
    """
    Returns a list of (material, model_path, rtbw_path) tuples for evaluation.
    Assumes model files are named <material>.pth and rtbw files are <material>.rtbw
    """
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    rtbw_files = set(f for f in os.listdir(rtbw_dir) if f.endswith(".rtbw"))
    pairs = []
    for model_file in model_files:
        material = model_file[:-4]  # strip .pth
        rtbw_file = f"{material}.rtbw"
        if rtbw_file in rtbw_files:
            pairs.append(
                (
                    material,
                    os.path.join(models_dir, model_file),
                    os.path.join(rtbw_dir, rtbw_file),
                )
            )
    return pairs


def evaluate_model_on_tablebase(
    model_path, rtbw_path, device, batch_size=4096, max_positions=None
):
    # Use same architecture as in training
    model = TablebaseModel(input_size=769, hidden_size=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    dataset = StreamingTablebaseDataset(
        rtbw_file=rtbw_path,
        tablebase_dir=os.path.dirname(rtbw_path),
        max_positions=max_positions,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        running_accuracy = 0.0
        for features, labels in tqdm(
            loader, desc=f"Evaluating {os.path.basename(model_path)}", leave=False
        ):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = batch_correct / batch_total
            running_accuracy += batch_accuracy
            correct += batch_correct
            total += batch_total
            tqdm.write(f"acc: {correct/total:.4f}")
    avg_accuracy = running_accuracy / len(loader) if len(loader) > 0 else 0.0
    return avg_accuracy, total


def main():
    # Directories
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    rtbw_dir = os.path.join(os.path.dirname(__file__), "Syzygy345_WDL")
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = find_model_and_rtbw_pairs(models_dir, rtbw_dir)
    log_lines = []
    for material, model_path, rtbw_path in pairs:
        print(f"Evaluating {material}...")
        accuracy, total = evaluate_model_on_tablebase(model_path, rtbw_path, device)
        line = f"{material}: {accuracy:.6f} ({total} positions)"
        print(line)
        log_lines.append(line)
    # Write log
    log_path = os.path.join(logs_dir, f"eval_full_{torch.__version__}.log")
    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")
    print(f"Evaluation complete. Log written to {log_path}")


if __name__ == "__main__":
    main()
