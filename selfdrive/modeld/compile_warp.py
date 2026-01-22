#!/usr/bin/env python3
import time
import pickle
import numpy as np
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.device import Device
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, DM_INPUT_SIZE
from openpilot.common.transformations.camera import _ar_ox_config, _os_config
from openpilot.selfdrive.modeld.constants import ModelConstants


MODELS_DIR = Path(__file__).parent / 'models'

MODEL_WIDTH, MODEL_HEIGHT = MEDMODEL_INPUT_SIZE
DM_WIDTH, DM_HEIGHT = DM_INPUT_SIZE
IMG_BUFFER_SHAPE = (6 * (ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ + 1), MODEL_HEIGHT // 2, MODEL_WIDTH // 2)

CAMERA_CONFIGS = {
  'ar_ox': _ar_ox_config.fcam,  # tici/tizi: 1928x1208
  'os': _os_config.fcam,        # mici: 1344x760
}

UV_SCALE_MATRIX = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float32)
UV_SCALE_MATRIX_INV = np.linalg.inv(UV_SCALE_MATRIX)


def get_warp_pkl_path(cam_width: int, cam_height: int) -> Path:
  return MODELS_DIR / f'warp_{cam_width}x{cam_height}.pkl'


def get_dm_warp_pkl_path(cam_width: int, cam_height: int) -> Path:
  return MODELS_DIR / f'dm_warp_{cam_width}x{cam_height}.pkl'


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, ratio):
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst)
  ones = Tensor.ones_like(x)
  dst_coords = x.reshape(1, -1).cat(y.reshape(1, -1)).cat(ones.reshape(1, -1))

  src_coords = M_inv @ dst_coords
  src_coords = src_coords / src_coords[2:3, :]

  x_nn_clipped = Tensor.round(src_coords[0]).clip(0, w_src - 1).cast('int')
  y_nn_clipped = Tensor.round(src_coords[1]).clip(0, h_src - 1).cast('int')
  idx = y_nn_clipped * w_src + (y_nn_clipped * ratio).cast('int') * stride_pad + x_nn_clipped

  sampled = src_flat[idx]
  return sampled


def frames_to_tensor(frames):
  H = (frames.shape[0]*2)//3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                        frames[1:H:2, 0::2],
                        frames[0:H:2, 1::2],
                        frames[1:H:2, 1::2],
                        frames[H:H+H//4].reshape((H//2,W//2)),
                        frames[H+H//4:H+H//2].reshape((H//2,W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1


def make_frame_prepare_tinygrad(cam_width: int, cam_height: int):
  stride, y_height, _, _ = get_nv12_info(cam_width, cam_height)
  uv_offset = stride * y_height

  def frame_prepare_tinygrad(input_frame, M_inv):
    tg_scale = Tensor(UV_SCALE_MATRIX)
    M_inv_uv = tg_scale @ M_inv @ Tensor(UV_SCALE_MATRIX_INV)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_height*stride],
                                    M_inv, (MODEL_WIDTH, MODEL_HEIGHT),
                                    (cam_height, cam_width), stride - cam_width, 1).realize()
      u = warp_perspective_tinygrad(input_frame[uv_offset:uv_offset + (cam_height//4)*stride],
                                    M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2),
                                    (cam_height//2, cam_width//2), stride - cam_width, 0.5).realize()
      v = warp_perspective_tinygrad(input_frame[uv_offset + (cam_height//4)*stride:uv_offset + (cam_height//2)*stride],
                                    M_inv_uv, (MODEL_WIDTH//2, MODEL_HEIGHT//2),
                                    (cam_height//2, cam_width//2), stride - cam_width, 0.5).realize()
    yuv = y.cat(u).cat(v).reshape((MODEL_HEIGHT*3//2, MODEL_WIDTH))
    tensor = frames_to_tensor(yuv)
    return tensor

  return frame_prepare_tinygrad


def make_update_img_input_tinygrad(frame_prepare_fn):
  def update_img_input_tinygrad(tensor, frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT)
    new_img = frame_prepare_fn(frame, M_inv)
    full_buffer = tensor[6:].cat(new_img, dim=0).contiguous()
    return full_buffer, Tensor.cat(full_buffer[:6], full_buffer[-6:], dim=0).contiguous().reshape(1, 12, MODEL_HEIGHT//2, MODEL_WIDTH//2)

  return update_img_input_tinygrad


def make_update_both_imgs_tinygrad(frame_prepare_fn):
  update_img_fn = make_update_img_input_tinygrad(frame_prepare_fn)

  def update_both_imgs_tinygrad(calib_img_buffer, new_img, M_inv,
                                calib_big_img_buffer, new_big_img, M_inv_big):
    calib_img_buffer, calib_img_pair = update_img_fn(calib_img_buffer, new_img, M_inv)
    calib_big_img_buffer, calib_big_img_pair = update_img_fn(calib_big_img_buffer, new_big_img, M_inv_big)
    return calib_img_buffer, calib_img_pair, calib_big_img_buffer, calib_big_img_pair

  return update_both_imgs_tinygrad


def make_dm_warp_tinygrad(cam_width: int, cam_height: int):
  stride, _, _, _ = get_nv12_info(cam_width, cam_height)

  def warp_dm(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT)
    with Context(SPLIT_REDUCEOP=0):
      result = warp_perspective_tinygrad(input_frame[:cam_height*stride], M_inv, (DM_WIDTH, DM_HEIGHT),
                                         (cam_height, cam_width), stride - cam_width, 1).reshape(-1, DM_HEIGHT * DM_WIDTH)
    return result

  return warp_dm


def compile_warp_for_camera(cam_width: int, cam_height: int):
  from tinygrad.engine.jit import TinyJit

  print(f"\n=== Compiling modeld warp for {cam_width}x{cam_height} ===")

  _, _, _, yuv_size = get_nv12_info(cam_width, cam_height)

  frame_prepare_fn = make_frame_prepare_tinygrad(cam_width, cam_height)
  update_both_imgs_fn = make_update_both_imgs_tinygrad(frame_prepare_fn)
  update_img_jit = TinyJit(update_both_imgs_fn, prune=True)

  full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()
  big_full_buffer = Tensor.zeros(IMG_BUFFER_SHAPE, dtype='uint8').contiguous().realize()

  for i in range(10):
    new_frame_np = (32*np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    img_inputs = [full_buffer,
                  Tensor.from_blob(new_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                  Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    new_big_frame_np = (32*np.random.randn(yuv_size).astype(np.float32) + 128).clip(0, 255).astype(np.uint8)
    big_img_inputs = [big_full_buffer,
                      Tensor.from_blob(new_big_frame_np.ctypes.data, (yuv_size,), dtype='uint8').realize(),
                      Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]
    inputs = img_inputs + big_img_inputs
    Device.default.synchronize()
    st = time.perf_counter()
    out = update_img_jit(*inputs)
    full_buffer = out[0].contiguous().realize().clone()
    big_full_buffer = out[2].contiguous().realize().clone()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  iter {i}: enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  pkl_path = get_warp_pkl_path(cam_width, cam_height)
  with open(pkl_path, "wb") as f:
    pickle.dump(update_img_jit, f)
  print(f"  Saved to {pkl_path}")

  # Verify loaded pickle works
  jit = pickle.load(open(pkl_path, "rb"))
  jit(*inputs)
  print(f"  Verified pickle loads correctly")


def compile_dm_warp_for_camera(cam_width: int, cam_height: int):
  from tinygrad.engine.jit import TinyJit

  print(f"\n=== Compiling DM warp for {cam_width}x{cam_height} ===")

  _, _, _, yuv_size = get_nv12_info(cam_width, cam_height)

  warp_dm_fn = make_dm_warp_tinygrad(cam_width, cam_height)
  warp_dm_jit = TinyJit(warp_dm_fn, prune=True)

  for i in range(10):
    inputs = [Tensor.from_blob((32*Tensor.randn(yuv_size,) + 128).cast(dtype='uint8').realize().numpy().ctypes.data, (yuv_size,), dtype='uint8'),
              Tensor(Tensor.randn(3, 3).mul(8).realize().numpy(), device='NPY')]

    Device.default.synchronize()
    st = time.perf_counter()
    warp_dm_jit(*inputs)
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  iter {i}: enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  pkl_path = get_dm_warp_pkl_path(cam_width, cam_height)
  with open(pkl_path, "wb") as f:
    pickle.dump(warp_dm_jit, f)
  print(f"  Saved to {pkl_path}")


def run_and_save_pickle():
  for name, cam in CAMERA_CONFIGS.items():
    print(f"\nProcessing camera config: {name} ({cam.width}x{cam.height})")
    compile_warp_for_camera(cam.width, cam.height)
    compile_dm_warp_for_camera(cam.width, cam.height)


if __name__ == "__main__":
  run_and_save_pickle()
