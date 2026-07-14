"""CLI training runner — no Streamlit required.

Examples:
    python run_training_cli.py --model DigitCNN --purpose English --dataset "MNIST + Fonts" --epochs 20
    python run_training_cli.py --model MultiTaskCNN --purpose Multi --epochs 30 --lr 0.001
    python run_training_cli.py --model EfficientNetDigit --purpose Persian --aug none --phase1 6 --phase2 14
"""
import argparse
import sys
import threading

import torch
from tqdm.auto import tqdm

from src.train_runner import TrainingConfig, run_training, RunCancelled


def main():
    parser = argparse.ArgumentParser(description="Sudoku digit model training (no Streamlit)")
    parser.add_argument("--model",    default="DigitCNN",
                        choices=["DigitCNN", "LegacyDigitCNN", "MultiTaskCNN",
                                 "UnifiedCNN — MobileNetV3", "UnifiedCNN — ShuffleNetV2",
                                 "EfficientNetDigit"])
    parser.add_argument("--purpose",  default="English",
                        choices=["English", "Persian", "Multi"])
    parser.add_argument("--dataset",  default="MNIST + Fonts",
                        choices=["MNIST Only", "MNIST + Fonts", "all",
                                 "MNIST + Hoda", "hoda"])
    parser.add_argument("--aug",      default="none",
                        choices=["none", "light", "full"])
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=128)
    parser.add_argument("--wd",       type=float, default=1e-4)
    parser.add_argument("--schedule", default="ReduceLROnPlateau",
                        choices=["ReduceLROnPlateau",
                                 "CosineAnnealingWarmRestarts",
                                 "CosineAnnealing",
                                 "OneCycleLR"],
                        help=(
                            "ReduceLROnPlateau: halves LR on stagnation. "
                            "CosineAnnealingWarmRestarts: cosine with restarts (use --cosine-t0, --cosine-mult). "
                            "CosineAnnealing: single cosine decay (uses --cosine-t0 as T_max). "
                            "OneCycleLR: warmup + decay."
                        ))
    parser.add_argument("--data",     default="data")
    parser.add_argument("--phase1",   type=int, default=6,  help="EfficientNet head-only epochs")
    parser.add_argument("--phase2",   type=int, default=14, help="EfficientNet fine-tune epochs")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights for EfficientNet/Unified")
    parser.add_argument("--backbone", default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "shufflenet_v2_x0_5"])
    parser.add_argument("--multi-mode", default="Unified model (20-class)",
                        choices=["Unified model (20-class)", "Separate models (Persian + English)"])
    parser.add_argument("--cosine-t0",   type=int,   default=10,
                        help="CosineAnnealingWarmRestarts: T_0 (first cycle epochs). "
                             "CosineAnnealing: T_max. Recommended: epochs÷2 (e.g. 10 for 20 epochs).")
    parser.add_argument("--cosine-mult", type=int,   default=1,
                        help="CosineAnnealingWarmRestarts T_mult. 1=equal cycles, 2=doubling cycles.")
    parser.add_argument("--eta-min",     type=float, default=1e-6,
                        help="Minimum LR floor for cosine schedulers. Recommended: lr/1000.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = TrainingConfig(
        model_choice=args.model,
        purpose=args.purpose,
        dataset_choice=args.dataset,
        aug_preset=args.aug,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        weight_decay=args.wd,
        lr_schedule=args.schedule,
        data_path=args.data,
        eff_pretrained=not args.no_pretrained,
        eff_phase1_epochs=args.phase1,
        eff_phase2_epochs=args.phase2,
        unified_backbone=args.backbone,
        unified_pretrained=not args.no_pretrained,
        multi_mode=args.multi_mode,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_mult,
        cosine_eta_min=args.eta_min,
    )

    sched_detail = config.lr_schedule
    if config.lr_schedule in ("CosineAnnealingWarmRestarts", "CosineAnnealing"):
        sched_detail += (f"  T_0={config.cosine_t0}"
                         + (f"  T_mult={config.cosine_t_mult}"
                            if config.lr_schedule == "CosineAnnealingWarmRestarts" else "")
                         + f"  eta_min={config.cosine_eta_min:.2e}")

    print("─" * 72)
    print(f"  Device  : {device}")
    print(f"  Model   : {config.model_choice}")
    print(f"  Purpose : {config.purpose}")
    print(f"  Dataset : {config.dataset_choice}  aug={config.aug_preset}")
    print(f"  Epochs  : {config.epochs}  lr={config.learning_rate}  "
          f"batch={config.batch_size}  wd={config.weight_decay}")
    print(f"  Schedule: {sched_detail}")
    print("─" * 72)

    total_epochs = (config.eff_phase1_epochs + config.eff_phase2_epochs
                    if config.model_choice == "EfficientNetDigit"
                    else config.epochs)

    epoch_bar = tqdm(
        total=total_epochs,
        desc="Training",
        unit="ep",
        ncols=90,
        colour="green",
        dynamic_ncols=True,
    )

    def _on_epoch(update: dict) -> None:
        phase = update.get('phase') or ''
        postfix = {
            'tr_loss': f"{update['train_loss']:.4f}",
            'tr_acc':  f"{update['train_acc']:.1f}%",
            'vl_loss': f"{update['val_loss']:.4f}",
            'vl_acc':  f"{update['val_acc']:.1f}%",
        }
        if phase and phase not in ('train',):
            postfix['phase'] = phase
        epoch_bar.set_postfix(postfix, refresh=False)
        epoch_bar.update(1)

    stop_ev = threading.Event()

    # Allow Ctrl+C to stop cleanly
    import signal
    def _sig(sig, frame):
        print("\nCtrl+C — stopping after current batch…")
        stop_ev.set()
    signal.signal(signal.SIGINT, _sig)

    try:
        result = run_training(config, device, on_epoch=_on_epoch, stop_event=stop_ev)
    except RunCancelled:
        epoch_bar.close()
        print("\nTraining cancelled.")
        sys.exit(0)
    except Exception:
        epoch_bar.close()
        import traceback
        traceback.print_exc()
        sys.exit(1)

    epoch_bar.close()
    print("─" * 72)
    print(f"Done. Model saved → {result.get('save_path', '?')}")
    if 'test_acc' in result:
        print(f"Test accuracy : {result['test_acc']:.2f}%")
    if 'test_digit_acc' in result:
        print(f"Test digit acc: {result['test_digit_acc']:.2f}%  "
              f"lang acc: {result.get('test_lang_acc', 0):.2f}%")
    if 'report_path' in result:
        print(f"Report        → {result['report_path']}")


if __name__ == "__main__":
    main()