import argparse
import os.path as osp
import traceback  # Added for detailed error reporting
from typing import Dict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config, DictAction
from torch import Tensor

from Net.dataset.track_dataset import TrackDataModule
from Net.utils import (MODELS, generate_save_dir,
                           training_info)

def export_to_onnx(model, data_module, save_path):
    print("\n>>> Starting ONNX Export...")

    # 1. Get core model
    core_model = getattr(model, 'model', getattr(model, 'net', model))
    core_model.eval()
    onnx_file = osp.join(save_path, 'model.onnx')

    try:
        # 2. Get input from val_loader
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))
        
        # Determine sensor key
        sensor_key = getattr(model, 'sensor_based', 'imu')
        
        # Extract first sample/timestep for tracing
        # Batch shapes are [bs, dim, seq_len]
        sensor = batch[sensor_key][0:1, :, 0:1].to(model.device)
        correction = batch['filtered_gps'][0:1, :, 0:1].to(model.device)
        initial_state = batch['initial_state'][0:1].to(model.device)

        # Initialize internal beliefs for tracing
        core_model.init_beliefs(initial_state)

        print(f">>> Exporting using legacy tracing...")
        print(f">>> Sensor shape: {sensor.shape}, Correction shape: {correction.shape}")

        torch.onnx.export(
            core_model,
            (sensor, correction),
            onnx_file,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['sensor', 'correction'],
            output_names=['output'],
            dynamic_axes={
                'sensor': {0: 'batch_size'},
                'correction': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        if osp.exists(onnx_file):
            print(f">>> SUCCESS! File generated: {onnx_file}")
            return onnx_file
        else:
            print(">>> Export finished but file not found.")
            return None

    except Exception as e:
        print("\n>>> EXPORT FAILURE")
        traceback.print_exc()
        print("Tip: Ensure all assertions in forward path are commented out.")
        return None

def export_to_tf(onnx_file, save_path):
    print("\n>>> Starting ONNX to TensorFlow Conversion...")
    tf_path = osp.join(save_path, 'tf_model')
    
    # 1. Simplify ONNX (highly recommended for TF conversion)
    try:
        import onnxsim
        import onnx
        print(">>> Simplifying ONNX model...")
        model_onnx = onnx.load(onnx_file)
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx_file_simp = onnx_file.replace('.onnx', '_simp.onnx')
            onnx.save(model_simp, onnx_file_simp)
            onnx_file = onnx_file_simp
            print(f">>> Simplified ONNX saved to: {onnx_file}")
    except Exception as e:
        print(f">>> ONNX simplification skipped/failed: {e}")

    # 2. Convert to TF
    # Try onnx2tf first (most reliable modern tool)
    try:
        import subprocess
        import os
        onnx2tf_path = os.path.expanduser('~/.local/bin/onnx2tf')
        if not os.path.exists(onnx2tf_path):
            onnx2tf_path = 'onnx2tf' # Fallback to PATH

        print(f">>> Attempting conversion via {onnx2tf_path}...")
        subprocess.run([onnx2tf_path, '-i', onnx_file, '-o', tf_path, '--non_verbose'], check=True)
        print(f">>> SUCCESS! TensorFlow model directory created: {tf_path}")

        # 3. Handle TFLite (onnx2tf often creates them automatically)
        tflite_auto = osp.join(tf_path, onnx_file.split('/')[-1].replace('.onnx', '_float32.tflite'))
        tflite_dest = osp.join(save_path, 'model.tflite')

        if osp.exists(tflite_auto):
            import shutil
            shutil.copy(tflite_auto, tflite_dest)
            print(f">>> SUCCESS! TFLite model (auto-generated) saved to: {tflite_dest}")
        else:
            # Fallback manual conversion if onnx2tf didn't make a TFLite file but made a SavedModel
            try:
                import tensorflow as tf
                print(">>> Attempting manual TFLite conversion from SavedModel...")
                converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
                tflite_model = converter.convert()
                with open(tflite_dest, 'wb') as f:
                    f.write(tflite_model)
                print(f">>> SUCCESS! TFLite model (manual) saved to: {tflite_dest}")
            except Exception as e:
                print(f">>> Manual TFLite conversion skipped/failed: {e}")
        return
    except Exception as e:
        print(f">>> onnx2tf failed or not installed: {e}")

    # Try onnx-tf (legacy)
    try:
        # Patch onnx import for onnx-tf if needed
        import onnx
        if not hasattr(onnx, 'mapping'):
            import onnx.helper
            # Mocking mapping for old onnx-tf compatibility
            class MockMapping:
                NP_TYPE_TO_TENSOR_TYPE = {
                    'float32': 1, 'float64': 11, 'int32': 6, 'int64': 7
                }
            onnx.mapping = MockMapping()

        from onnx_tf.backend import prepare
        print(">>> Attempting conversion via onnx-tf...")
        onnx_model = onnx.load(onnx_file)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_path)
        print(f">>> SUCCESS! TensorFlow model saved to: {tf_path}")
    except Exception as e:
        print(f">>> onnx-tf conversion failed: {e}")
        print("\n--- TF CONVERSION TIPS ---")
        print("1. Install onnx2tf: pip install onnx2tf")
        print("2. Ensure TensorFlow version matches your ONNX opset.")
        print("3. Use a stable environment (e.g., Python 3.10) for conversion if 3.14 fails.")

def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()
    torch.manual_seed(3407)
    save_dir: Dict = generate_save_dir(root='./runs',
                                       project=cfg.logger.project,
                                       name=cfg.logger.name)
    cfg.logger.name = save_dir['new_name']
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))

    data_module = TrackDataModule(
        cfg, use_transform=cfg.data.transforms.use_transform)
    data_module.setup()

    model = MODELS.build(
        dict(type=cfg.trainer.type, cfg=cfg, save_dir=save_dir))

    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_monitor = ModelCheckpoint(
        dirpath=save_dir['weight_dir'],
        filename='{epoch}-{val_loss:.2f}-{val_MSE_dB:.2f}',
        mode='min',
        save_top_k=10,
        monitor='val_MSE_dB')
    callbacks = [lr_monitor, model_monitor]

    wandb_logger = WandbLogger(project=cfg.logger.project,
                               name=cfg.logger.name,
                               offline=cfg.logger.offline)

    trainer = Trainer(
        accelerator='cpu',
        max_epochs=cfg.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        detect_anomaly=cfg.trainer.detect_anomaly,
        callbacks=callbacks,
        devices=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch if cfg.trainer.check_val_every_n_epoch is not None else 1
    )

    # Run training
    trainer.fit(model, datamodule=data_module)

    # Export to ONNX and then TF after training
    if cfg.get('export_onnx', True):
        onnx_file = export_to_onnx(model, data_module, save_dir['weight_dir'])
        if onnx_file and cfg.get('export_tf', True):
            export_to_tf(onnx_file, save_dir['weight_dir'])


def parse_args():
    parser = argparse.ArgumentParser(
        prog='KalmanNet',
        description='Dataset, training and network parameters')
    parser.add_argument('--config',
                        '--cfg',
                        type=str,
                        metavar='config',
                        help='model and seq ')

    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config.')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg)
    main(args, cfg)
