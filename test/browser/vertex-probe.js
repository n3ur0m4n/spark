export async function probeVertices(spark, renderer, count, math = false) {
  const stride = math ? 23 : 11;
  if (spark.isWebGPU) {
    const shaderRoot = "/src/shaders/";
    const [{ default: defines }, { default: vertex }] = await Promise.all([
      import(/* @vite-ignore */ `${shaderRoot}splatDefines.wgsl?raw`),
      import(/* @vite-ignore */ `${shaderRoot}splatVertex.wgsl?raw`),
    ]);
    let code = vertex.replace("@vertex\n", "");
    if (math) {
      code = code.replace(
        "struct VertexOutput {",
        "struct VertexOutput {\n @location(5) diagnosticCov:vec4f,\n @location(6) diagnosticEigen:vec4f,\n @location(7) diagnosticAxis:vec4f,",
      );
      code = code.replace(
        "    // Compute the NDC coordinates for the ellipsoid",
        `    output.diagnosticCov=vec4f(a,d,b,det);
    output.diagnosticEigen=vec4f(eigenAvg,eigenDelta,eigen1,eigen2);
    output.diagnosticAxis=vec4f(eigenVec1,scale1,scale2);
    // Compute the NDC coordinates for the ellipsoid`,
      );
    }
    code = `${defines}\n${code}\n@group(0) @binding(11) var<storage,read_write> outputs:array<f32>;
   @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=${count * 4}u){return;}
    let corners=array<vec2f,4>(vec2f(-1,-1),vec2f(1,-1),vec2f(1,1),vec2f(-1,1));let result=vs_main(VertexInput(corners[i%4u],i/4u));
    let values=array<f32,${stride}>(result.position.x,result.position.y,result.position.z,result.position.w,result.vSplatUv.x,result.vSplatUv.y,result.vRgba.x,result.vRgba.y,result.vRgba.z,result.vRgba.w,result.adjustedStdDev${math ? ",result.diagnosticCov.x,result.diagnosticCov.y,result.diagnosticCov.z,result.diagnosticCov.w,result.diagnosticEigen.x,result.diagnosticEigen.y,result.diagnosticEigen.z,result.diagnosticEigen.w,result.diagnosticAxis.x,result.diagnosticAxis.y,result.diagnosticAxis.z,result.diagnosticAxis.w" : ""});
    for(var k=0u;k<${stride}u;k++){outputs[i*${stride}u+k]=values[k];}}`;
    const b = spark.webgpuBackend;
    const d = renderer.backend.device;
    const slot = b.renderSlots[b.activeSlotIndex];
    const bytes = count * 4 * stride * 4;
    const out = d.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const map = d.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pipeline = d.createComputePipeline({
      layout: "auto",
      compute: {
        module: d.createShaderModule({ code }),
        entryPoint: "probe",
        constants: {
          HALVED_ALPHA: b.halvedAlpha ? 1 : 0,
          PREMULTIPLIED_ALPHA: 1,
        },
      },
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: b.sparkUniformBuffer } },
          { binding: 1, resource: { buffer: b.projMatrixBuffer } },
          { binding: 2, resource: slot.orderingTexture.createView() },
          { binding: 3, resource: slot.splatTextureView },
          {
            binding: 4,
            resource: slot.splatTextureView2 || slot.splatTextureView,
          },
          { binding: 11, resource: { buffer: out } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil((count * 4) / 64));
    pass.end();
    enc.copyBufferToBuffer(out, 0, map, 0, bytes);
    d.queue.submit([enc.finish()]);
    await map.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(map.getMappedRange().slice(0)));
    map.unmap();
    map.destroy();
    out.destroy();
    return values;
  }
  const gl = renderer.getContext();
  const source = renderer.info.programs.find((p) =>
    gl.getShaderSource(p.vertexShader)?.includes("orderingCoord"),
  );
  if (!source) throw Error("Splat shader not found");
  let shader = source.vertexShader;
  if (math) {
    let code = gl
      .getShaderSource(shader)
      .replace(
        "void main() {",
        "out vec4 diagnosticCov;\nout vec4 diagnosticEigen;\nout vec4 diagnosticAxis;\nvoid main() {",
      );
    code = code.replace(
      "vec2 pixelOffset =",
      `    diagnosticCov=vec4(a,d,b,det);
    diagnosticEigen=vec4(eigenAvg,eigenDelta,eigen1,eigen2);
    diagnosticAxis=vec4(eigenVec1,scale1,scale2);
    vec2 pixelOffset =`,
    );
    shader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(shader, code);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
      throw Error(gl.getShaderInfoLog(shader));
  }
  const program = gl.createProgram();
  gl.attachShader(program, shader);
  gl.attachShader(program, source.fragmentShader);
  gl.transformFeedbackVaryings(
    program,
    [
      "gl_Position",
      "vSplatUv",
      "vRgba",
      "adjustedStdDev",
      ...(math ? ["diagnosticCov", "diagnosticEigen", "diagnosticAxis"] : []),
    ],
    gl.INTERLEAVED_ATTRIBS,
  );
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  for (
    let i = 0;
    i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    i++
  ) {
    const info = gl.getActiveUniform(program, i);
    const location = gl.getUniformLocation(program, info.name);
    const value = gl.getUniform(
      source.program,
      gl.getUniformLocation(source.program, info.name),
    );
    const simple = new Map([
      [gl.FLOAT, "uniform1f"],
      [gl.BOOL, "uniform1i"],
      [gl.INT, "uniform1i"],
      [gl.UNSIGNED_INT, "uniform1ui"],
      [gl.FLOAT_VEC2, "uniform2fv"],
      [gl.FLOAT_VEC3, "uniform3fv"],
      [gl.FLOAT_VEC4, "uniform4fv"],
    ]);
    if (simple.has(info.type)) gl[simple.get(info.type)](location, value);
    else if (info.type === gl.FLOAT_MAT4 || info.type === gl.FLOAT_MAT3)
      gl[info.type === gl.FLOAT_MAT4 ? "uniformMatrix4fv" : "uniformMatrix3fv"](
        location,
        false,
        value,
      );
    else {
      gl.uniform1i(location, value);
      const texture = spark.uniforms[info.name]?.value;
      if (!texture) throw Error(`Missing sampler ${info.name}`);
      const target =
        info.type === gl.UNSIGNED_INT_SAMPLER_2D_ARRAY
          ? gl.TEXTURE_2D_ARRAY
          : gl.TEXTURE_2D;
      gl.activeTexture(gl.TEXTURE0 + value);
      gl.bindTexture(target, renderer.properties.get(texture).__webglTexture);
    }
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vertices = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]),
    gl.STATIC_DRAW,
  );
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
  const buffer = gl.createBuffer();
  const feedback = gl.createTransformFeedback();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedback);
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, buffer);
  gl.bufferData(
    gl.TRANSFORM_FEEDBACK_BUFFER,
    count * 4 * stride * 4,
    gl.STREAM_READ,
  );
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffer);
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArraysInstanced(gl.POINTS, 0, 4, count);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);
  const result = new Float32Array(count * 4 * stride);
  gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, result);
  const error = gl.getError();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
  gl.bindVertexArray(null);
  gl.deleteTransformFeedback(feedback);
  gl.deleteBuffer(buffer);
  gl.deleteBuffer(vertices);
  gl.deleteVertexArray(vao);
  gl.deleteProgram(program);
  if (math) gl.deleteShader(shader);
  renderer.resetState();
  if (error) throw Error(`Transform feedback GL error ${error}`);
  return Array.from(result);
}
