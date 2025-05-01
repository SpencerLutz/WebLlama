// Functions like createBuffer, copyToBuffer, calculateBufferSize

export const bufferUsageDict = {
    "copy_src": GPUBufferUsage.COPY_SRC,
    "copy_dst": GPUBufferUsage.COPY_DST,
    "storage": GPUBufferUsage.STORAGE,
    "uniform": GPUBufferUsage.UNIFORM,
    "map_read": GPUBufferUsage.MAP_READ,
    "indirect": GPUBufferUsage.INDIRECT,
};

export function calcBufferSize(x_size: number, y_size: number = 1) {
    return x_size * y_size * Float32Array.BYTES_PER_ELEMENT;
}