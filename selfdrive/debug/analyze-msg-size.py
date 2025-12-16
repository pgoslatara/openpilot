#!/usr/bin/env python3
from tqdm import tqdm
from openpilot.tools.lib.logreader import LogReader
from cereal.services import SERVICE_LIST


if __name__ == "__main__":
  lr = LogReader("98395b7c5b27882e/000000a8--f87e7cd255")

  szs = {}
  for msg in tqdm(lr):
    sz = len(msg.as_builder().to_bytes())
    msg_type = msg.which()
    if msg_type not in szs:
      szs[msg_type] = {'min': sz, 'max': sz, 'sum': sz, 'count': 1}
    else:
      szs[msg_type]['min'] = min(szs[msg_type]['min'], sz)
      szs[msg_type]['max'] = max(szs[msg_type]['max'], sz)
      szs[msg_type]['sum'] += sz
      szs[msg_type]['count'] += 1

  # Print sorted table
  print()
  print(f"{'Service':<36} {'Min (KB)':>12} {'Max (KB)':>12} {'Avg (KB)':>12} {'KB/min':>12} {'KB/sec':>12} {'Minutes in 10MB':>18}")
  print("-" * 114)
  def sort_key(x):
    k, v = x
    avg = v['sum'] / v['count']
    freq = SERVICE_LIST.get(k, None)
    freq_val = freq.frequency if freq else 0.0
    kb_per_min = (avg * freq_val * 60) / 1024 if freq_val > 0 else 0.0
    return kb_per_min
  total_kb_per_min = 0.0
  RINGBUFFER_SIZE_MB = 10  # this is the current MSGQ ringbuffer size
  RINGBUFFER_SIZE_KB = RINGBUFFER_SIZE_MB * 1024
  for k, v in sorted(szs.items(), key=sort_key, reverse=True):
    avg = v['sum'] / v['count']
    freq = SERVICE_LIST.get(k, None)
    freq_val = freq.frequency if freq else 0.0
    kb_per_min = (avg * freq_val * 60) / 1024 if freq_val > 0 else 0.0
    kb_per_sec = kb_per_min / 60
    minutes_in_buffer = RINGBUFFER_SIZE_KB / kb_per_min if kb_per_min > 0 else float('inf')
    total_kb_per_min += kb_per_min
    min_str = f"{minutes_in_buffer:.2f}" if minutes_in_buffer != float('inf') else "inf"
    print(f"{k:<36} {v['min']/1024:>12.2f} {v['max']/1024:>12.2f} {avg/1024:>12.2f} {kb_per_min:>12.2f} {kb_per_sec:>12.2f} {min_str:>18}")

  # Summary section
  print()
  print(f"Total usage: {total_kb_per_min / 1024:.2f} MB/min")
