"""
Enterprise AI-Powered Person Tracking & Crowd Analytics
Module: Model Optimization & ONNX Export Pipeline

Author: Aryan Jha / Antigravity Team
Description: Exports Ultralytics YOLOv8 PyTorch weights (.pt) to Open Neural Network
             Exchange (ONNX) format for hardware-accelerated, high-throughput edge
             and cloud inference. Includes ONNX graph integrity validation and
             ONNXRuntime benchmark verification.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ONNX_Exporter")


def parse_arguments():
    """Parse command line arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export and validate YOLOv8 PyTorch models to ONNX format."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 PyTorch weights file (.pt). Defaults to 'yolov8n.pt'.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640, 640],
        help="Input image resolution [height, width] or single integer. Default: 640 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export. Default: 1",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input/output axes for variable resolution and batch size.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 half-precision quantization (requires compatible GPU/runtime).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
        help="Simplify ONNX graph using ONNX-Slim (if installed).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version. Default: 12 (wide compatibility).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run ONNXRuntime benchmark and shape verification on the exported model.",
    )
    return parser.parse_args()


def export_to_onnx(
    weights_path: str,
    imgsz: list,
    batch_size: int = 1,
    dynamic: bool = False,
    half: bool = False,
    simplify: bool = False,
    opset: int = 12,
) -> Path:
    """
    Exports a YOLO model to ONNX using the official Ultralytics engine.

    Args:
        weights_path: Path to PyTorch model (.pt).
        imgsz: Image resolution as [H, W] or int.
        batch_size: Export batch dimension.
        dynamic: Whether to enable dynamic axes.
        half: Enable FP16 quantization.
        simplify: Simplify ONNX graph.
        opset: ONNX operator set version.

    Returns:
        Path to exported ONNX model file.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error("Ultralytics library is not installed. Run 'pip install ultralytics'.")
        raise e

    logger.info("=" * 60)
    logger.info("Initializing YOLOv8 ONNX Export Pipeline")
    logger.info("=" * 60)
    logger.info(f"Target Weights : {weights_path}")
    logger.info(f"Input Size     : {imgsz}")
    logger.info(f"Batch Size     : {batch_size}")
    logger.info(f"Dynamic Axes   : {dynamic}")
    logger.info(f"FP16 Precision : {half}")
    logger.info(f"Opset Version  : {opset}")
    logger.info("-" * 60)

    # Load PyTorch YOLO model
    logger.info(f"Loading weights from: {weights_path}...")
    model = YOLO(weights_path)

    # Perform ONNX export
    start_time = time.perf_counter()
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch_size,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
        opset=opset,
    )
    export_duration = time.perf_counter() - start_time

    exported_path = Path(exported_path_str)
    if not exported_path.exists():
        raise FileNotFoundError(f"Export completed but output file not found at: {exported_path}")

    file_size_mb = exported_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX Model successfully exported in {export_duration:.2f}s!")
    logger.info(f"Saved Path: {exported_path.resolve()}")
    logger.info(f"Model File Size: {file_size_mb:.2f} MB")
    return exported_path


def verify_onnx_model(onnx_model_path: Path, imgsz: list, dynamic: bool = False):
    """
    Validates structural graph integrity with the 'onnx' library and performs
    synthetic inference benchmarks with 'onnxruntime'.

    Args:
        onnx_model_path: Path to exported .onnx file.
        imgsz: Expected input resolution [H, W].
        dynamic: Whether dynamic axes were enabled.
    """
    logger.info("-" * 60)
    logger.info("Starting ONNX Model Verification & Benchmarking")
    logger.info("-" * 60)

    # 1. Structural graph check
    try:
        import onnx

        logger.info("Verifying ONNX computational graph integrity...")
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX graph integrity check passed with zero errors.")
    except ImportError:
        logger.warning("'onnx' package not found. Skipping static graph validation.")
    except Exception as e:
        logger.error(f"ONNX graph integrity validation failed: {e}")
        raise e

    # 2. Runtime benchmark with ONNXRuntime
    try:
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        preferred_provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available_providers
            else "CPUExecutionProvider"
        )
        logger.info(f"Available Providers : {available_providers}")
        logger.info(f"Selected Provider   : {preferred_provider}")

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=session_opts,
            providers=[preferred_provider],
        )

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        logger.info(f"Input Name   : '{input_meta.name}' | Type: {input_meta.type} | Shape: {input_meta.shape}")
        logger.info(f"Output Name  : '{output_meta.name}' | Type: {output_meta.type} | Shape: {output_meta.shape}")

        # Resolve height and width
        h = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        w = imgsz[1] if isinstance(imgsz, (list, tuple)) and len(imgsz) > 1 else h

        # Generate synthetic input tensor (Batch=1, Channels=3, H, W)
        dummy_input = np.random.rand(1, 3, h, w).astype(np.float32)

        # Warmup iterations
        logger.info("Executing 5 warmup inference cycles...")
        for _ in range(5):
            _ = session.run([output_meta.name], {input_meta.name: dummy_input})

        # Benchmark latency
        num_runs = 20
        logger.info(f"Benchmarking inference latency across {num_runs} iterations...")
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            outputs = session.run([output_meta.name], {input_meta.name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        fps = 1000.0 / avg_latency

        logger.info(f"Average Latency : {avg_latency:.2f} ms")
        logger.info(f"P95 Latency     : {p95_latency:.2f} ms")
        logger.info(f"Throughput      : ~{fps:.1f} FPS")
        logger.info(f"Output Shape    : {outputs[0].shape}")
        logger.info("ONNXRuntime Verification Succeeded!")

    except ImportError:
        logger.warning("'onnxruntime' package not found. Skipping live runtime validation.")
    except Exception as e:
        logger.error(f"Runtime verification failed: {e}")
        raise e


def main():
    args = parse_arguments()
    try:
        exported_file = export_to_onnx(
            weights_path=args.weights,
            imgsz=args.imgsz,
            batch_size=args.batch,
            dynamic=args.dynamic,
            half=args.half,
            simplify=args.simplify,
            opset=args.opset,
        )

        if args.verify:
            verify_onnx_model(
                onnx_model_path=exported_file,
                imgsz=args.imgsz,
                dynamic=args.dynamic,
            )

        logger.info("=" * 60)
        logger.info("Production Export Pipeline Completed Successfully.")
        logger.info("=" * 60)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
