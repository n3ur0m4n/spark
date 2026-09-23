type LossListener = (info: GPUDeviceLostInfo) => void;
type LossState = {
  listeners: Set<LossListener>;
  lost?: GPUDeviceLostInfo;
};

const devices = new WeakMap<GPUDevice, LossState>();

// Keep the promise callback in its own scope: it must never retain an
// individual renderer's listener after that renderer unsubscribes.
function observeDevice(device: GPUDevice): LossState {
  const state: LossState = { listeners: new Set() };
  void device.lost.then((info) => {
    state.lost = info;
    try {
      for (const listener of state.listeners) listener(info);
    } finally {
      state.listeners.clear();
    }
  });
  return state;
}

/** One observer per host device, with removable per-renderer subscriptions. */
export function subscribeDeviceLost(
  device: GPUDevice,
  listener: LossListener,
): () => void {
  let state = devices.get(device);
  if (!state) {
    state = observeDevice(device);
    devices.set(device, state);
  }
  if (state.lost) {
    listener(state.lost);
  } else {
    state.listeners.add(listener);
  }
  return () => state.listeners.delete(listener);
}
