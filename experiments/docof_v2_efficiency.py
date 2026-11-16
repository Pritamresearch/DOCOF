import time
import torch
from thop import profile

from config_v2 import CONFIG_V2
from models.docof_model_v2 import DOCOFv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, history, state_t, masks):

    flops, params = profile(
        model,
        inputs=(history, state_t, masks),
        verbose=False
    )

    return flops, params


def measure_fps(model, history, state_t, masks, device):

    model.eval()

    iterations = 200
    warmup = 50

    with torch.no_grad():

        # warmup
        for _ in range(warmup):
            model(history, state_t, masks)

        torch.cuda.synchronize() if device.type == "cuda" else None

        start = time.time()

        for _ in range(iterations):
            model(history, state_t, masks)

        torch.cuda.synchronize() if device.type == "cuda" else None

        end = time.time()

    fps = iterations / (end - start)

    return fps


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DOCOFv2(CONFIG_V2).to(device)

    model.load_state_dict(
        torch.load("docof_v2_best_mean.pth", map_location=device)
    )

    model.eval()

    # Dummy inputs (same shape as evaluation)
    history = torch.randn(1, 5, 50, 7).to(device)
    state_t = torch.randn(1, 50, 7).to(device)

    masks = [
        torch.ones(1, 50).to(device)
        for _ in range(CONFIG_V2["future_frames"])
    ]

    # Parameters
    params = count_parameters(model)

    # FLOPs
    flops, _ = measure_flops(model, history, state_t, masks)

    # FPS
    fps = measure_fps(model, history, state_t, masks, device)

    print("\n==============================")
    print("DOCOF Efficiency Analysis")
    print("==============================")

    print(f"Parameters : {params/1e6:.2f} M")
    print(f"FLOPs      : {flops/1e9:.2f} G")
    print(f"FPS        : {fps:.2f}")


if __name__ == "__main__":
    main()