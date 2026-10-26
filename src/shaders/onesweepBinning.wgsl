// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © Thomas Smith (b0nes164/GPUSorting)
// SPDX-License-Identifier: MIT
//
// OneSweep DigitBinningPass: fused rank + block scan + decoupled lookback +
// coalesced scatter for one 8-bit radix pass. Adapted from playcanvas/engine
// 209ce5752527ab5050ec9632ca6dc1272542b70d (onesweep-binning.js), see that
// file for the phase-by-phase description. The key count is always read from
// b_count[0]; the dispatch must be exactly ceil(count / PART_SIZE) 1D
// workgroups. The lookback spins, so it needs forward-progress guarantees
// (NVIDIA/AMD/Intel desktop); other GPUs use the multipass sort.

enable subgroups;

@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(2) var<storage, read> inputValues: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
@group(0) @binding(4) var<storage, read_write> b_passHist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> b_index: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> b_count: array<u32>;

// FLAGS bit 0: first pass (values = original index), bit 1: skip key write.
override PASS: u32 = 0u;
override FLAGS: u32 = 0u;

const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const D_DIM: u32 = 256u;
// Host-patched: 15 keys/thread need 16420 B of workgroup memory, above the
// 16384 B default limit; 14 fits.
const KEYS_PER_THREAD: u32 = 14u;
const PART_SIZE: u32 = D_DIM * KEYS_PER_THREAD;
const MAX_SUBGROUPS: u32 = 8u;
const WAVE_HISTS_SIZE: u32 = MAX_SUBGROUPS * RADIX;
const G_D_SIZE: u32 = max(PART_SIZE, WAVE_HISTS_SIZE);

const FLAG_REDUCTION: u32 = 1u;
const FLAG_INCLUSIVE: u32 = 2u;
const FLAG_MASK: u32 = 3u;

var<workgroup> g_d: array<atomic<u32>, G_D_SIZE>;
var<workgroup> digit_base: array<u32, RADIX>;
var<workgroup> sg_totals: array<u32, MAX_SUBGROUPS>;
var<workgroup> wg_partIndex: u32;

fn passHistOffset(tb: u32, partitionIdx: u32) -> u32 {
    return PASS * tb * RADIX + partitionIdx * RADIX;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_index) TID: u32,
    @builtin(subgroup_invocation_id) sgInvId: u32,
    @builtin(subgroup_size) sgSize: u32,
) {
    let waveIndex = TID / sgSize;
    let ltMask = (1u << sgInvId) - 1u;
    let activeMask = select(0xFFFFFFFFu, (1u << sgSize) - 1u, sgSize < 32u);
    let currentBit = PASS << 3u;
    let numKeys = b_count[0];
    let threadBlocks = (numKeys + PART_SIZE - 1u) / PART_SIZE;
    let isFirstPass = (FLAGS & 1u) != 0u;
    let isLastPass = (FLAGS & 2u) != 0u;

    if (TID == 0u) {
        wg_partIndex = atomicAdd(&b_index[PASS], 1u);
    }
    let partitionIndex = workgroupUniformLoad(&wg_partIndex);

    for (var i = TID; i < WAVE_HISTS_SIZE; i = i + D_DIM) {
        atomicStore(&g_d[i], 0u);
    }
    workgroupBarrier();

    let tileStart = partitionIndex * PART_SIZE;
    let validInBlock = select(0u, min(PART_SIZE, numKeys - tileStart), tileStart < numKeys);

    let subPartSize = sgSize * KEYS_PER_THREAD;
    let waveBase = tileStart + waveIndex * subPartSize;

    var keys: array<u32, KEYS_PER_THREAD>;
    var values: array<u32, KEYS_PER_THREAD>;
    var validMask: u32 = 0u;
    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        let gid = waveBase + sgInvId + i * sgSize;
        let is_valid = gid < numKeys;
        keys[i] = select(0xFFFFFFFFu, inputKeys[gid], is_valid);
        values[i] = select(select(0u, inputValues[gid], is_valid), gid, isFirstPass);
        if (is_valid) {
            validMask = validMask | (1u << i);
        }
    }

    var offsets: array<u32, KEYS_PER_THREAD>;
    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        let k = keys[i];
        let isValid = ((validMask >> i) & 1u) == 1u;
        let digit = (k >> currentBit) & RADIX_MASK;

        var waveFlag: u32 = activeMask;
        for (var b = 0u; b < 8u; b = b + 1u) {
            let t = ((digit >> b) & 1u) == 1u;
            let ballot = subgroupBallot(t).x;
            waveFlag = waveFlag & select(~ballot, ballot, t);
        }
        waveFlag = waveFlag & subgroupBallot(isValid).x;

        let peerBits = countOneBits(waveFlag & ltMask);
        let totalBits = countOneBits(waveFlag);
        let lowestRankPeer = firstTrailingBit(waveFlag);

        var preIncrementVal: u32 = 0u;
        if (isValid && peerBits == 0u) {
            preIncrementVal = atomicAdd(&g_d[waveIndex * RADIX + digit], totalBits);
        }
        offsets[i] = subgroupShuffle(preIncrementVal, lowestRankPeer) + peerBits;
        // Reconverge lanes (Independent Thread Scheduling) before next round.
        workgroupBarrier();
    }

    var myHistRed: u32 = 0u;
    {
        var histReduction = atomicLoad(&g_d[TID]);
        for (var w = 1u; w < MAX_SUBGROUPS; w = w + 1u) {
            let idx = TID + w * RADIX;
            let cnt = atomicLoad(&g_d[idx]);
            histReduction = histReduction + cnt;
            atomicStore(&g_d[idx], histReduction - cnt);
        }
        myHistRed = histReduction;
    }

    if (partitionIndex + 1u < threadBlocks) {
        atomicAdd(&b_passHist[passHistOffset(threadBlocks, partitionIndex + 1u) + TID],
            FLAG_REDUCTION | (myHistRed << 2u));
    }

    let warpExcl = subgroupExclusiveAdd(myHistRed);
    let warpTotal = subgroupAdd(myHistRed);
    if (sgInvId == 0u) {
        sg_totals[waveIndex] = warpTotal;
    }
    workgroupBarrier();
    if (TID == 0u) {
        var acc: u32 = 0u;
        for (var w = 0u; w < MAX_SUBGROUPS; w = w + 1u) {
            let t = sg_totals[w];
            sg_totals[w] = acc;
            acc = acc + t;
        }
    }
    workgroupBarrier();
    let myDigitBase = warpExcl + sg_totals[waveIndex];

    var scatterPos: array<u32, KEYS_PER_THREAD>;
    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        let digit = (keys[i] >> currentBit) & RADIX_MASK;
        let warpBase = select(0u, atomicLoad(&g_d[waveIndex * RADIX + digit]), waveIndex > 0u);
        scatterPos[i] = offsets[i] + warpBase;
    }
    digit_base[TID] = myDigitBase;
    workgroupBarrier();
    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        let digit = (keys[i] >> currentBit) & RADIX_MASK;
        scatterPos[i] = scatterPos[i] + digit_base[digit];
    }
    workgroupBarrier();

    // Decoupled lookback (spins on FLAG_NOT_READY).
    {
        var lookbackReduction: u32 = 0u;
        var k: u32 = partitionIndex;
        var done: bool = false;
        loop {
            if (done) { break; }
            let flagPayload = atomicLoad(&b_passHist[passHistOffset(threadBlocks, k) + TID]);
            let flag = flagPayload & FLAG_MASK;
            if (flag == FLAG_INCLUSIVE) {
                lookbackReduction = lookbackReduction + (flagPayload >> 2u);
                if (partitionIndex + 1u < threadBlocks) {
                    atomicAdd(&b_passHist[passHistOffset(threadBlocks, partitionIndex + 1u) + TID],
                        1u | (lookbackReduction << 2u));
                }
                digit_base[TID] = lookbackReduction - myDigitBase;
                done = true;
            } else if (flag == FLAG_REDUCTION) {
                lookbackReduction = lookbackReduction + (flagPayload >> 2u);
                if (k == 0u) { done = true; } else { k = k - 1u; }
            }
        }
    }

    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        if (((validMask >> i) & 1u) == 1u) {
            atomicStore(&g_d[scatterPos[i]], keys[i]);
        }
    }
    workgroupBarrier();

    var linearDigits: array<u32, KEYS_PER_THREAD>;
    for (var r = 0u; r < KEYS_PER_THREAD; r = r + 1u) {
        let linearIdx = TID + r * D_DIM;
        if (linearIdx < validInBlock) {
            let k = atomicLoad(&g_d[linearIdx]);
            let digit = (k >> currentBit) & RADIX_MASK;
            linearDigits[r] = digit;
            if (!isLastPass) {
                outputKeys[digit_base[digit] + linearIdx] = k;
            }
        }
    }
    workgroupBarrier();

    for (var i = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
        if (((validMask >> i) & 1u) == 1u) {
            atomicStore(&g_d[scatterPos[i]], values[i]);
        }
    }
    workgroupBarrier();

    for (var r = 0u; r < KEYS_PER_THREAD; r = r + 1u) {
        let linearIdx = TID + r * D_DIM;
        if (linearIdx < validInBlock) {
            outputValues[digit_base[linearDigits[r]] + linearIdx] = atomicLoad(&g_d[linearIdx]);
        }
    }
}
