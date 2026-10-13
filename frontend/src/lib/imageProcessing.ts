export const MAX_UPLOAD_DIM = 512;

export interface Dimensions {
    width: number;
    height: number;
}

/**
 * Calculates resized dimensions preserving original aspect ratio.
 * Enforces maximum dimension without accidental width/height cross-assignment.
 */
export function calculateTargetDimensions(
    width: number,
    height: number,
    maxDim: number = MAX_UPLOAD_DIM
): Dimensions {
    if (width <= 0 || height <= 0 || !Number.isFinite(width) || !Number.isFinite(height)) {
        return { width: Math.max(1, maxDim), height: Math.max(1, maxDim) };
    }
    if (width <= maxDim && height <= maxDim) {
        return { width: Math.round(width), height: Math.round(height) };
    }
    let newW = width;
    let newH = height;
    if (width > height) {
        // Landscape: width is dominant
        newW = maxDim;
        newH = Math.max(1, Math.round((height / width) * maxDim));
    } else {
        // Portrait or Square: height is dominant
        newH = maxDim;
        newW = Math.max(1, Math.round((width / height) * maxDim));
    }
    return { width: newW, height: newH };
}

/**
 * Computes SHA-256 hex digest from an ArrayBuffer.
 */
export async function calculateBufferSha256(buffer: ArrayBuffer): Promise<string> {
    try {
        const digest = await crypto.subtle.digest("SHA-256", buffer);
        const array = Array.from(new Uint8Array(digest));
        return array.map((b) => b.toString(16).padStart(2, "0")).join("");
    } catch {
        return "Unavailable";
    }
}
