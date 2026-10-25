type LossListener = (info: GPUDeviceLostInfo) => void;
/** One observer per host device, with removable per-renderer subscriptions. */
export declare function subscribeDeviceLost(device: GPUDevice, listener: LossListener): () => void;
export {};
