# Image-To-Equation (.i2e) Format Specification (v1.0)

This document specifies the binary bitstream format for the `.i2e` Image-to-Equation Codec.

The `.i2e` format represents raster images either via a generative AST (Abstract Syntax Tree) mathematical equation, or via a compressed DCT (Discrete Cosine Transform) frequency block.

## 1. File Structure

All values are written in Little-Endian format.

| Offset | Size (Bytes) | Type | Name | Description |
|--------|--------------|------|------|-------------|
| 0x00 | 4 | `char[4]` | `Magic` | Must be exactly `I2E\x01` (ASCII "I2E" followed by byte `0x01`). |
| 0x04 | 2 | `uint16` | `Width` | Unsigned 16-bit integer representing the image width. |
| 0x06 | 2 | `uint16` | `Height` | Unsigned 16-bit integer representing the image height. |
| 0x08 | 1 | `uint8` | `Mode` | `0x01` (Symbolic AST) or `0x02` (DCT Fallback). |
| 0x09 | 4 | `uint32` | `PayloadLength`| Unsigned 32-bit integer indicating the size of the compressed payload in bytes. |
| 0x0D | N | `byte[]` | `Payload` | A Zlib-compressed (`DEFLATE`) binary payload. |

## 2. Modes and Payload Structures

The payload must be decompressed using standard Zlib (RFC 1950) before interpretation.

### Mode `0x01`: Symbolic AST
The decompressed payload is interpreted as a raw UTF-8 string containing the source equation.

**Constraints:**
- The string must evaluate to a valid mathematical expression outputting an RGB array.
- The string should not exceed 8,192 characters in most reference implementations.

### Mode `0x02`: DCT Fallback
If an image contains too much high-frequency unstructured data (e.g., photographs) for the neural-symbolic router to compress cleanly, the encoder falls back to a frequency representation.

The decompressed payload follows this layout:

For each channel in order `[R, G, B]`:
1. **Term Count (4 bytes)**: Unsigned 32-bit int (`uint32`). The number of non-zero DCT coefficients for this channel.
2. **Coefficients (8 bytes per term)**:
   - `delta_u` (2 bytes, `int16`): The difference between this term's `u` frequency and the previous term's `u` frequency.
   - `delta_v` (2 bytes, `int16`): The difference between this term's `v` frequency and the previous term's `v` frequency.
   - `amplitude` (4 bytes, `float32`): The IEEE-754 single-precision float amplitude.

*Note: `prev_u` and `prev_v` reset to `0` at the start of each new channel.*
