// SPDX-License-Identifier: MPL-2.0
// Canonical little-endian, gzip/base64-encoded typed arrays for bounded JSON checkpoints.
import { gzipSync, gunzipSync } from 'node:zlib'
type Integers = Uint8Array | Uint16Array | Int32Array

export function packIntegers(a: Integers): string {
  const width = a.BYTES_PER_ELEMENT
  const bytes = Buffer.allocUnsafe(a.length * width)
  for (let i = 0; i < a.length; i++) {
    if (width === 1) bytes.writeUInt8(a[i], i)
    else if (width === 2) bytes.writeUInt16LE(a[i], i * 2)
    else bytes.writeInt32LE(a[i], i * 4)
  }
  return gzipSync(bytes).toString('base64')
}
export function unpackIntegers(encoded: string, out: Integers): void {
  const expected = out.length * out.BYTES_PER_ELEMENT
  if (typeof encoded !== 'string' || encoded.length > (expected * 2 + 1024) ||
      encoded.length % 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(encoded))
    throw Error('invalid CM array encoding')
  let bytes: Buffer
  try { bytes = gunzipSync(Buffer.from(encoded, 'base64'), { maxOutputLength: Math.max(1, expected + 1) }) }
  catch { throw Error('invalid compressed CM array') }
  if (bytes.length !== expected) throw Error('invalid CM array length')
  for (let i = 0; i < out.length; i++) {
    if (out.BYTES_PER_ELEMENT === 1) out[i] = bytes.readUInt8(i)
    else if (out.BYTES_PER_ELEMENT === 2) out[i] = bytes.readUInt16LE(i * 2)
    else out[i] = bytes.readInt32LE(i * 4)
  }
}
