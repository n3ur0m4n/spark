import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { ExtSplats } from "./ExtSplats";
import { PackedSplats } from "./PackedSplats";
import type { SparkHostRenderer } from "./RendererAdapter";
import type { SparkRenderStats } from "./SparkRenderStats";
import type { SparkRenderer } from "./SparkRenderer";
import { SplatAccumulator } from "./SplatAccumulator";
import { SplatEdit } from "./SplatEdit";
import { SplatGenerator } from "./SplatGenerator";
import { SplatMesh } from "./SplatMesh";
import { SplatWorker } from "./SplatWorker";
import { WebGPUComposite } from "./WebGPUComposite";
import type { NativeMeshGeneration } from "./WebGPUGeneration";
import { WebGPUSplatBackend } from "./WebGPUSplatBackend";

/** The native submission path; the WebGL renderer and LoD traversal stay unchanged. */
export class SparkNativeRenderer {
  readonly backend: WebGPUSplatBackend;
  private composite: WebGPUComposite | null = null;
  private queue: Promise<void> = Promise.resolve();
  private disposed = false;
  private renderVisible = true;
  private readonly hostSize = new THREE.Vector2();
  private readonly hostViewport = new THREE.Vector4();
  private sequence = 0;
  private currentSignature = "";
  private updateQueue: Promise<SplatMesh[]> = Promise.resolve([]);
  private worker: SplatWorker | null = null;
  private ordering = new Uint32Array(0);
  private readonly identities = new WeakMap<object, number>();
  private nextIdentity = 1;
  private stats: SparkRenderStats = {
    generation: 0,
    selectedSplats: 0,
    drawnSplats: 0,
    ordering: "readback",
  };

  getRenderStats(): SparkRenderStats {
    return {
      ...this.stats,
      drawnSplats: this.renderVisible ? this.stats.drawnSplats : 0,
    };
  }

  private isVisible(camera: THREE.Camera): boolean {
    for (
      let node: THREE.Object3D | null = this.owner;
      node;
      node = node.parent
    ) {
      if (!node.visible) return false;
    }
    return camera.layers.test(this.owner.layers);
  }

  async readRenderStatsAsync(): Promise<SparkRenderStats> {
    if (this.disposed) throw new Error("SparkRenderer is disposed");
    const snapshot = this.getRenderStats();
    if (snapshot.drawnSplats !== null) return snapshot;
    // The copy is submitted synchronously, before another generation can reuse
    // this slot. A late result describes its original generation only.
    const drawnSplats = await this.backend.readNativeDrawCount();
    const measured = { ...snapshot, drawnSplats };
    if (!this.disposed && this.stats.generation === snapshot.generation) {
      this.stats = measured;
      this.owner.activeSplats = drawnSplats;
    }
    return measured;
  }

  private identity(value?: object): number {
    if (!value) return 0;
    let id = this.identities.get(value);
    if (id === undefined) {
      id = this.nextIdentity++;
      this.identities.set(value, id);
    }
    return id;
  }

  constructor(
    private readonly owner: SparkRenderer<SparkHostRenderer>,
    private readonly renderer: WebGPURenderer,
    device: GPUDevice,
  ) {
    this.backend = new WebGPUSplatBackend(device);
  }

  assertSupported(scene: THREE.Scene, camera: THREE.Camera) {
    const requireWebGL = (feature: string): never => {
      throw new Error(
        `Spark: ${feature} requires THREE.WebGLRenderer; native output was not submitted.`,
      );
    };
    if (this.disposed) throw new Error("SparkRenderer is disposed");
    if (camera instanceof THREE.ArrayCamera) requireWebGL("array cameras");
    if (this.backend.deviceLost)
      throw new Error(
        `Spark: host GPU device lost: ${this.backend.deviceLostReason}`,
      );
    if (this.backend.uncapturedErrorCount)
      throw new Error(
        `Spark: native GPU validation failed: ${this.backend.firstUncapturedError}`,
      );
    if (this.renderer.getRenderTarget())
      requireWebGL("specialized render targets");
    if (this.renderer.logarithmicDepthBuffer) requireWebGL("logarithmic depth");
    if (
      camera.reversedDepth ||
      ("reversedDepthBuffer" in this.renderer &&
        this.renderer.reversedDepthBuffer === true)
    )
      requireWebGL("reversed depth");
    if (this.renderer.getScissorTest()) requireWebGL("scissor test");
    if (
      !this.renderer.autoClear ||
      !this.renderer.autoClearColor ||
      !this.renderer.autoClearDepth
    )
      requireWebGL("manual clearing");
    // Both methods use logical pixels, including on high-DPI canvases.
    this.renderer.getSize(this.hostSize);
    this.renderer.getViewport(this.hostViewport);
    if (
      this.hostViewport.x !== 0 ||
      this.hostViewport.y !== 0 ||
      this.hostViewport.z !== this.hostSize.x ||
      this.hostViewport.w !== this.hostSize.y
    )
      requireWebGL("custom viewports");
    if (this.renderer.xr.isPresenting) requireWebGL("WebXR");
    if (this.renderer.samples > 0) requireWebGL("native MSAA");
    if (scene.overrideMaterial) requireWebGL("scene override materials");
    if (this.owner.covSplats) requireWebGL("covariance splats");
    if (
      !(
        camera instanceof THREE.PerspectiveCamera ||
        camera instanceof THREE.OrthographicCamera
      )
    ) {
      requireWebGL("custom cameras");
    }
    scene.updateMatrixWorld();
    scene.traverseVisible((node) => {
      if (!camera.layers.test(node.layers)) return;
      if (
        node instanceof THREE.Mesh ||
        node instanceof THREE.Line ||
        node instanceof THREE.Points ||
        node instanceof THREE.Sprite
      ) {
        const materials = Array.isArray(node.material)
          ? node.material
          : [node.material];
        if (
          materials.some(
            (material) =>
              material.transparent ||
              ("transmission" in material &&
                typeof material.transmission === "number" &&
                material.transmission > 0),
          )
        ) {
          requireWebGL("transparent or transmissive Three.js objects");
        }
      }
      if (node instanceof SplatEdit) requireWebGL("editing");
      if (!(node instanceof SplatGenerator)) return;
      if (!(node instanceof SplatMesh) || !node.hasNativeSourceGenerator())
        requireWebGL("custom splat generators");
      if (!(node instanceof SplatMesh)) return;
      if (node.paged) requireWebGL("paging");
      if (node.covSplats) requireWebGL("covariance splats");
      if (
        node.objectModifiers?.length ||
        node.worldModifiers?.length ||
        node.covObjectModifiers?.length ||
        node.covWorldModifiers?.length
      )
        requireWebGL("modifiers");
      if (node.skinning) requireWebGL("skinning");
      if (node.edits?.length || node.rgbaDisplaceEdits || node.splatRgba)
        requireWebGL("editing");
      if (
        node.enableViewToObject ||
        node.enableViewToWorld ||
        node.enableWorldToView ||
        node.showLodPage != null
      )
        requireWebGL("custom view-dependent generation");
      const e = node.matrixWorld.elements;
      const x = new THREE.Vector3(e[0], e[1], e[2]);
      const y = new THREE.Vector3(e[4], e[5], e[6]);
      const z = new THREE.Vector3(e[8], e[9], e[10]);
      const scale = x.lengthSq();
      const tolerance = 1e-5 * Math.max(scale, 1e-20);
      if (
        scale === 0 ||
        node.matrixWorld.determinant() <= 0 ||
        Math.abs(y.lengthSq() - scale) > tolerance ||
        Math.abs(z.lengthSq() - scale) > tolerance ||
        Math.abs(x.dot(y)) > tolerance ||
        Math.abs(x.dot(z)) > tolerance ||
        Math.abs(y.dot(z)) > tolerance
      )
        requireWebGL("nonuniform, mirrored or sheared splat transforms");
    });
  }

  /** Each call keeps its camera snapshot and has its own submission promise. */
  renderAsync(scene: THREE.Scene, camera: THREE.Camera): Promise<void> {
    if (this.disposed)
      return Promise.reject(new Error("SparkRenderer is disposed"));
    // ArrayCamera inherits PerspectiveCamera, but needs one generation per view.
    // Reject it before cloning can discard its subcamera list (Three r180).
    if (camera instanceof THREE.ArrayCamera)
      return Promise.reject(
        new Error(
          "Spark: array cameras require THREE.WebGLRenderer; native output was not submitted.",
        ),
      );
    // Camera.copy in Three r180 does not retain the reversed-depth flag.
    if (camera.reversedDepth)
      return Promise.reject(
        new Error(
          "Spark: reversed depth requires THREE.WebGLRenderer; native output was not submitted.",
        ),
      );
    camera.updateWorldMatrix(true, false);
    const snapshot = camera.clone(false);
    snapshot.matrix.copy(camera.matrixWorld);
    snapshot.matrixWorld.copy(camera.matrixWorld);
    snapshot.matrixWorldInverse.copy(camera.matrixWorldInverse);
    snapshot.matrixAutoUpdate = false;
    snapshot.matrixWorldAutoUpdate = false;
    const task = this.queue
      .catch(() => {})
      .then(async () => {
        this.assertSupported(scene, snapshot);
        const visible = this.isVisible(snapshot);
        this.renderer.getDrawingBufferSize(this.owner.renderSize);
        if (visible)
          await this.owner.onBeforeRender(this.renderer, scene, snapshot);
        this.assertSupported(scene, snapshot);
        if (!this.composite) {
          const composite = await WebGPUComposite.create();
          if (this.disposed) {
            composite.dispose();
            throw new Error("SparkRenderer is disposed");
          }
          this.composite = composite;
        }
        // The host can resize while generation/readback or module loading is
        // pending. Projection uniforms must match the attachments drawn below.
        this.renderer.getDrawingBufferSize(this.owner.renderSize);
        this.renderVisible = visible && this.isVisible(snapshot);
        this.updateUniforms(snapshot);
        this.composite.render(
          this.renderer,
          scene,
          snapshot,
          ({ color, depth }) => {
            if (!this.renderVisible) return;
            this.backend.premultipliedAlpha = this.owner.premultipliedAlpha;
            this.backend.ensurePipeline(color.format);
            if (
              !this.backend.setExternalDepthTexture(
                depth,
                color.width,
                color.height,
              )
            ) {
              throw new Error(
                "Spark: incompatible native composite depth attachment",
              );
            }
            this.backend.renderSeparatePass(
              color,
              this.owner.activeSplats,
              true,
            );
          },
        );
      });
    this.queue = task;
    return task;
  }

  private updateUniforms(camera: THREE.Camera) {
    const owner = this.owner;
    const uniforms = owner.uniforms;
    const typed = camera as THREE.PerspectiveCamera | THREE.OrthographicCamera;
    typed.coordinateSystem = THREE.WebGPUCoordinateSystem;
    typed.updateProjectionMatrix();
    uniforms.renderSize.value.copy(owner.renderSize);
    uniforms.near.value = typed.near;
    uniforms.far.value = typed.far;
    const transform = camera.matrixWorld.clone().invert();
    if (!owner.display.extSplats)
      transform.multiply(
        new THREE.Matrix4().makeTranslation(owner.display.viewOrigin),
      );
    transform.decompose(
      uniforms.renderToViewPos.value,
      uniforms.renderToViewQuat.value,
      new THREE.Vector3(),
    );
    uniforms.renderToViewBasis.value.setFromMatrix4(transform);
    for (const key of [
      "maxStdDev",
      "minPixelRadius",
      "maxPixelRadius",
      "minAlpha",
      "preBlurAmount",
      "blurAmount",
      "focalDistance",
      "apertureAngle",
      "falloff",
      "clipXY",
      "focalAdjustment",
    ] as const) {
      uniforms[key].value = owner[key];
    }
    uniforms.enable2DGS.value = owner.enable2DGS;
    uniforms.lodInflate.value = owner.lodInflate;
    uniforms.encodeLinear.value = true;
    uniforms.enableExtSplats.value = owner.display.extSplats;
    uniforms.enableCovSplats.value = false;
    uniforms.time.value = owner.display.time;
    uniforms.deltaTime.value = owner.display.deltaTime;
    this.backend.updateUniforms(
      uniforms,
      typed.projectionMatrix,
      typed instanceof THREE.OrthographicCamera,
    );
  }

  /** Runs ordinary mesh callbacks, then snapshots and commits one complete generation. */
  update(scene: THREE.Scene, camera: THREE.Camera): Promise<SplatMesh[]> {
    const task = this.updateQueue
      .catch(() => [])
      .then(() => this.updateFrame(scene, camera));
    this.updateQueue = task;
    return task;
  }

  private async updateFrame(
    scene: THREE.Scene,
    camera: THREE.Camera,
  ): Promise<SplatMesh[]> {
    this.assertSupported(scene, camera);
    camera.updateWorldMatrix(true, false);
    const owner = this.owner;
    const sortRadial = owner.sortRadial;
    const sortMode = owner.webgpuSort;
    const next = new SplatAccumulator({
      extSplats: owner.accumExtSplats,
      covSplats: false,
    });
    next.time = owner.timer.getElapsed();
    next.deltaTime = owner.timer.getDelta();
    next.viewToWorld.copy(camera.matrixWorld);
    camera.getWorldPosition(next.viewOrigin);
    camera.getWorldDirection(next.viewDirection);
    const all: SplatMesh[] = [];
    scene.traverse((node) => {
      if (node instanceof SplatMesh && camera.layers.test(node.layers))
        all.push(node);
    });
    for (const node of all) {
      node.update({
        object: node,
        time: next.time,
        deltaTime: next.deltaTime,
        viewToWorld: next.viewToWorld,
        camera,
        renderSize: owner.renderSize,
        globalEdits: [],
        lodIndices: owner.enableLod ? owner.lodInstances.get(node) : undefined,
      });
    }
    this.assertSupported(scene, camera);
    const visible: SplatMesh[] = [];
    scene.traverseVisible((node) => {
      if (node instanceof SplatMesh && camera.layers.test(node.layers))
        visible.push(node);
    });
    const counts = visible.map((node) => {
      const pending =
        owner.enableLod &&
        owner.enableDriveLod &&
        node.enableLod !== false &&
        (node.packedSplats?.lodSplats || node.extSplats?.lodSplats) &&
        !owner.lodInstances.has(node);
      return pending ? 0 : node.numSplats;
    });
    const layout = next.generateMapping(counts);
    const meshes: NativeMeshGeneration[] = [];
    for (let i = 0; i < visible.length; i++) {
      const node = visible[i];
      const { base, count } = layout.mapping[i];
      if (!count) continue;
      const source = node.context.splats;
      if (!(source instanceof PackedSplats || source instanceof ExtSplats))
        throw new Error(
          "Spark: native source requires PackedSplats or ExtSplats",
        );
      if (source instanceof PackedSplats && source.target) {
        throw new Error(
          "Spark: GPU-generated source textures require THREE.WebGLRenderer",
        );
      }
      const extended = source instanceof ExtSplats;
      const primary = extended ? source.extArrays[0] : source.packedArray;
      const secondary = extended ? source.extArrays[1] : undefined;
      const indices =
        owner.enableLod && node.context.enableLod.value
          ? owner.lodInstances.get(node)?.indices
          : undefined;
      if (
        !primary ||
        primary.length < source.numSplats * 4 ||
        (extended && (!secondary || secondary.length < source.numSplats * 4)) ||
        (indices && indices.length < count)
      )
        throw new Error("Spark: incomplete native source or selection");
      const enc = extended ? undefined : source.splatEncoding;
      const sh = [
        source.extra.sh1,
        source.extra.sh2,
        extended ? source.extra.sh3a : source.extra.sh3,
        extended ? source.extra.sh3b : undefined,
      ].map((value) => (value instanceof Uint32Array ? value : undefined));
      const numSh = Math.min(node.maxSh, source.maxSh, source.getNumSh());
      for (let band = 0; band < (numSh === 3 && extended ? 4 : numSh); band++) {
        const words = source.numSplats * (!extended && band === 0 ? 2 : 4);
        const coefficients = sh[band];
        if (!coefficients || coefficients.length < words) {
          throw new Error("Spark: incomplete native spherical-harmonic source");
        }
      }
      const transform = node.context.transform;
      meshes.push({
        source,
        revision: source.dataVersion,
        primary,
        secondary,
        sh,
        sourceCount: source.numSplats,
        sourceExt: extended,
        base,
        count,
        indices,
        rotation: transform.rotate.value.toArray(),
        translationScale: [
          ...transform.translate.value.toArray(),
          transform.scale.value,
        ],
        localViewOrigin: node.context.viewToObject.translate.value.toArray(),
        encoding: [
          enc?.rgbMin ?? 0,
          enc?.rgbMax ?? 1,
          enc?.lnScaleMin ?? -12,
          enc?.lnScaleMax ?? 9,
        ],
        shMaxOpacity: [
          enc?.sh1Max ?? 1,
          enc?.sh2Max ?? 1,
          enc?.sh3Max ?? 1,
          enc?.lodOpacity ? 2 : 1,
        ],
        numSh,
        recolor: node.context.recolor.value.toArray(),
      });
      next.mapping.push({
        node,
        base,
        count,
        version: node.version,
        mappingVersion: node.mappingVersion,
        generator: node.generator,
      });
      next.numSplats = Math.max(next.numSplats, base + count);
    }
    const signature = JSON.stringify([
      sortMode,
      sortRadial,
      owner.accumExtSplats,
      next.viewToWorld.toArray(),
      next.viewOrigin.toArray(),
      next.viewDirection.toArray(),
      visible.map((node, i) => [node.uuid, node.version, counts[i]]),
      meshes.map((mesh) => [
        this.identity(mesh.source),
        this.identity(mesh.primary),
        this.identity(mesh.secondary),
        ...mesh.sh.map((array) => this.identity(array)),
        this.identity(mesh.indices),
        mesh.base,
        mesh.count,
        mesh.revision,
        mesh.sourceCount,
        mesh.numSh,
        mesh.rotation,
        mesh.translationScale,
        mesh.recolor,
        mesh.encoding,
        mesh.shMaxOpacity,
      ]),
    ]);
    if (signature === this.currentSignature) {
      next.dispose();
      return visible;
    }
    const sequence = ++this.sequence;
    next.version = sequence;
    next.mappingVersion = sequence;
    owner.sorting = true;
    this.backend.beginRenderSlotUpdate("native-generation");
    const generation = this.backend.getWorkingSlotGeneration();
    try {
      let activeSplats: number;
      let indirect = false;
      this.backend.generateNative({
        meshes,
        numSplats: next.numSplats,
        outputExt: next.extSplats,
        viewOrigin: next.viewOrigin.toArray(),
        viewDirection: next.viewDirection.toArray(),
        sortRadial,
      });
      const rows = Math.max(1, Math.ceil(next.numSplats / 16384));
      if (
        sortMode === "radix" &&
        next.numSplats > 0 &&
        (await this.backend.gpuDepthSortAndPack({
          numSplats: next.numSplats,
          viewOrigin: next.viewOrigin,
          viewDir: next.viewDirection,
          sortRadial,
          enableExtSplats: next.extSplats,
          orderingRows: rows,
          orderingTarget: "working",
        }))
      ) {
        indirect = true;
        activeSplats = meshes.reduce((sum, mesh) => sum + mesh.count, 0);
      } else {
        const readback = await this.backend.readNativeDepths(next.numSplats);
        if (this.disposed || sequence !== this.sequence)
          throw new Error(
            "SparkRenderer is disposed or native generation was superseded",
          );
        if (this.ordering.length < rows * 16384)
          this.ordering = new Uint32Array(rows * 16384);
        this.worker ??= new SplatWorker();
        const result = await this.worker.call("sortSplats32", {
          numSplats: next.numSplats,
          readback,
          ordering: this.ordering,
        });
        this.ordering = result.ordering;
        if (this.disposed || sequence !== this.sequence)
          throw new Error(
            "SparkRenderer is disposed or native generation was superseded",
          );
        activeSplats = result.activeSplats;
        this.backend.uploadOrdering(result.ordering, 4096, rows, "working");
        this.backend.useIndirectDraw = false;
      }
      if (
        this.disposed ||
        sequence !== this.sequence ||
        generation !== this.backend.getWorkingSlotGeneration()
      )
        throw new Error("Spark: stale native generation");
      this.backend.commitRenderSlot();
      owner.activeSplats = activeSplats;
      this.stats = {
        generation: sequence,
        selectedSplats: meshes.reduce((sum, mesh) => sum + mesh.count, 0),
        drawnSplats: indirect ? null : activeSplats,
        ordering: indirect ? "radix" : "readback",
      };
      const old = owner.display;
      owner.current = next;
      owner.display = next;
      old.dispose();
      owner.sortedCenter.copy(next.viewOrigin);
      owner.sortedDir.copy(next.viewDirection);
      owner.setDirty();
      this.currentSignature = signature;
      return visible;
    } catch (error) {
      this.backend.cancelRenderSlotUpdate(generation);
      next.dispose();
      throw error;
    } finally {
      owner.sorting = false;
    }
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.sequence++;
    this.worker?.dispose();
    this.worker = null;
    this.composite?.dispose();
    this.composite = null;
    this.backend.dispose();
  }
}
