import time
import torch
from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2


def measure_fps(model, device, num_objects):

    future_frames = CONFIG_V2["future_frames"]

    history = torch.randn(1, 5, num_objects, 7).to(device)
    state_t = torch.randn(1, num_objects, 7).to(device)

    masks = [
        torch.ones(1, num_objects).to(device)
        for _ in range(future_frames)
    ]

    model.eval()

    warmup = 50
    iterations = 200

    with torch.no_grad():

        for _ in range(warmup):
            model(history, state_t, masks)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        for _ in range(iterations):
            model(history, state_t, masks)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    fps = iterations / (end - start)

    return fps


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DOCOFv2(CONFIG_V2).to(device)

    model.load_state_dict(
        torch.load("docof_v2_best_mean.pth", map_location=device)
    )

    object_counts = [20, 50, 100, 200]

    print("\n==============================")
    print("DOCOF Object Scalability")
    print("==============================")

    for n in object_counts:

        fps = measure_fps(model, device, n)

        print(f"Objects: {n:3d} | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()