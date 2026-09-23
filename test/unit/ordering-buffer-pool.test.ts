import assert from "node:assert/strict";
import { once } from "node:events";
import { MessageChannel } from "node:worker_threads";
import * as THREE from "three";
import { test } from "vitest";
import { OrderingBufferPool } from "../../src/OrderingBufferPool.js";

async function transfer(data: Uint32Array): Promise<Uint32Array> {
  const { port1, port2 } = new MessageChannel();
  const received = once(port2, "message");
  port1.postMessage(data, [data.buffer as ArrayBuffer]);
  const [result] = await received;
  port1.close();
  port2.close();
  assert.equal(data.byteLength, 0);
  return result;
}

test("ordering storage preserves ownership across transfers and retirement", async () => {
  const pool = new OrderingBufferPool();
  const texture = new THREE.DataTexture(null, 4096, 1);
  let first = pool.take(16384);
  first[0] = 7;
  first = await transfer(first);
  pool.attach(texture, first);
  let second = pool.take(16384);
  assert.notEqual(second.buffer, first.buffer);
  second[0] = 9;
  second = await transfer(second);
  assert.equal(
    texture.image.data.byteLength,
    65536,
    "context restoration retains attached data during transfer",
  );
  assert.equal(texture.image.data[0], 7);
  pool.attach(texture, second);
  assert.equal(
    texture.image.data[0],
    9,
    "restoration sees latest committed order",
  );
  assert.equal(
    pool.take(16384).buffer,
    first.buffer,
    "released CPU storage is reused",
  );

  // A retained previous texture keeps owning its data until it is disposed.
  const previous = new THREE.DataTexture(null, 4096, 1);
  pool.attach(previous, first);
  assert.notEqual(pool.take(16384).buffer, first.buffer);
  previous.dispose();
  assert.equal(
    pool.take(8192).buffer,
    first.buffer,
    "smaller cuts reuse capacity after retirement",
  );

  const growth = pool.take(32768);
  assert.equal(growth.length, 32768);
  pool.attach(texture, growth);
  assert.notEqual(
    pool.take(32768).buffer,
    second.buffer,
    "undersized spare is replaced",
  );
  texture.dispose();
  pool.clear();
  assert.notEqual(
    pool.take(32768).buffer,
    growth.buffer,
    "renderer disposal releases the pool",
  );
  console.log("Ordering buffer ownership and actual transfer checks passed.");

  // Active-row views must not lose the capacity of the dedicated worker buffer.
  const large = new Uint32Array(65536);
  const trimmed = new THREE.DataTexture(large.subarray(0, 16384), 4096, 1);
  pool.attach(trimmed, trimmed.image.data as Uint32Array);
  trimmed.dispose();
  assert.equal(pool.take(65536).buffer, large.buffer);

  // Repeated dispose events cannot reclaim a buffer now owned by another texture.
  const retired = new THREE.DataTexture(new Uint32Array(65536), 4096, 4);
  pool.attach(retired, retired.image.data as Uint32Array);
  retired.dispose();
  const reused = pool.take(65536);
  const current = new THREE.DataTexture(reused, 4096, 4);
  pool.attach(current, reused);
  retired.dispose();
  assert.notEqual(pool.take(65536).buffer, reused.buffer);
  current.dispose();
  pool.clear();

  const aliasBuffer = new Uint32Array(32768);
  const aliasTexture = new THREE.DataTexture(aliasBuffer, 4096, 2);
  pool.attach(aliasTexture, aliasBuffer);
  pool.attach(aliasTexture, aliasBuffer.subarray(0, 16384));
  assert.notEqual(pool.take(32768).buffer, aliasBuffer.buffer);
  aliasTexture.dispose();
  assert.equal(pool.take(32768).buffer, aliasBuffer.buffer);
});
