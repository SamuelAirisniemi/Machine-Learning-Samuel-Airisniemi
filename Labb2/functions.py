import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask


def load_image(img_path):
    return Image.open(img_path).convert("RGB")


def get_prediction(model, img_pil, preprocess):
    batch = preprocess(img_pil).unsqueeze(0)
    output = model(batch).squeeze(0).softmax(0)
    return output.detach()


def topk_predictions(output_tensor: torch.Tensor, class_index_path: str, k: int = 5) -> list:
    with open(class_index_path, "r") as f:
        class_index = json.load(f)

    probs = output_tensor.squeeze()

    if probs.ndim != 1 or probs.shape[0] != 1000:
        raise ValueError(f"Expected a tensor of 1000 values, got shape {tuple(output_tensor.shape)}")

    values, indices = torch.topk(probs, k)

    results = []
    for value, idx in zip(values, indices):
        idx = int(idx)
        synset_id, class_name = class_index[str(idx)]
        results.append({
            "class_index": idx,
            "class_id": synset_id,
            "class_name": class_name,
            "confidence": float(value),
        })

    return results


def generate_cam(model, img_pil, preprocess, target_layer="layer4"):
    input_tensor = preprocess(img_pil)

    with LayerCAM(model, target_layer=target_layer) as cam_extractor:
        out = model(input_tensor.unsqueeze(0))
        class_idx = out.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, out)

    return activation_map, class_idx


def generate_overlay(img_pil, activation_map, alpha=0.5):
    return overlay_mask(img_pil, to_pil_image(activation_map[0], mode="F"), alpha=alpha)


def show_cam_comparison(img_pil, activation_map, overlay, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(activation_map[0].squeeze(0).cpu().numpy(), cmap="jet")
    axes[0].set_title("Heatmap")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Overlay")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def analyze_image(model, preprocess, img_path, class_index_path, label, target_layer="layer4"):
    img_pil = load_image(img_path)

    prediction = get_prediction(model, img_pil, preprocess)
    top5 = topk_predictions(prediction, class_index_path, k=5)

    activation_map, class_idx = generate_cam(
        model, img_pil, preprocess, target_layer=target_layer
    )
    overlay = generate_overlay(img_pil, activation_map)

    print("Top 5 predictions:")
    for item in top5:
        print(item)

    show_cam_comparison(
        img_pil,
        activation_map,
        overlay,
        title=f"{label} ({target_layer})"
    )

    return {
        "top5": top5,
        "activation_map": activation_map,
        "overlay": overlay,
        "class_idx": class_idx
    }