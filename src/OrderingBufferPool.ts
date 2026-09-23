import type * as THREE from "three";

/** Worker scratch may be transferred only after its texture owner releases it. */
export class OrderingBufferPool {
  private spare: Uint32Array<ArrayBuffer> | undefined;
  private readonly observed = new WeakSet<THREE.DataTexture>();

  take(length: number): Uint32Array<ArrayBuffer> {
    const buffer = this.spare;
    this.spare = undefined;
    return buffer && buffer.length >= length ? buffer : new Uint32Array(length);
  }

  private release(data: ArrayBufferView | null): void {
    if (
      !(data instanceof Uint32Array) ||
      !(data.buffer instanceof ArrayBuffer)
    ) {
      return;
    }
    if (data.buffer.byteLength / 4 > (this.spare?.length ?? 0)) {
      this.spare = new Uint32Array(data.buffer);
    }
  }

  /** Keep current CPU data attached for initial upload and context restoration. */
  attach(texture: THREE.DataTexture, data: Uint32Array): void {
    const previous = texture.image.data;
    texture.image.data = data;
    if (previous?.buffer !== data.buffer) this.release(previous);
    if (!this.observed.has(texture)) {
      this.observed.add(texture);
      const onDispose = () => {
        texture.removeEventListener("dispose", onDispose);
        this.observed.delete(texture);
        this.release(texture.image.data);
      };
      texture.addEventListener("dispose", onDispose);
    }
  }

  clear(): void {
    this.spare = undefined;
  }
}
