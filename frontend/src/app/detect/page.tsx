"use client";

import { useState, useEffect, useCallback } from "react";
import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { ReportCard } from "@/components/shared/ReportCard";
import { HeatmapViewer } from "@/components/shared/HeatmapViewer";
import { motion } from "framer-motion";
import { Loader2, AlertCircle } from "lucide-react";
import Image from "next/image";
import { detectionApi, DetectionResult } from "@/lib/api";

const MAX_UPLOAD_DIM = 1024;

/**
 * Resizes an image on the client side before uploading.
 * Caps the image at MAX_UPLOAD_DIM px while preserving the original format
 * to avoid destroying compression artifacts that the deepfake model relies on.
 */
function resizeImageForUpload(file: File): Promise<File> {
    return new Promise((resolve, reject) => {
        const img = new window.Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            const { width, height } = img;

            // If already within bounds, send original unchanged
            if (width <= MAX_UPLOAD_DIM && height <= MAX_UPLOAD_DIM) {
                resolve(file);
                return;
            }

            // Calculate new dimensions maintaining aspect ratio
            let newW = width;
            let newH = height;
            if (width > height) {
                newW = MAX_UPLOAD_DIM;
                newH = Math.round((height / width) * MAX_UPLOAD_DIM);
            } else {
                newH = MAX_UPLOAD_DIM;
                newW = Math.round((width / height) * MAX_UPLOAD_DIM);
            }

            const canvas = document.createElement("canvas");
            canvas.width = newW;
            canvas.height = newH;
            const ctx = canvas.getContext("2d");
            if (!ctx) { resolve(file); return; }
            ctx.drawImage(img, 0, 0, newW, newH);

            // Preserve original format — avoid re-encoding to JPEG which
            // destroys compression artifacts the deepfake detector relies on
            const mimeType = file.type || "image/png";
            const quality = mimeType === "image/jpeg" ? 0.95 : undefined;

            canvas.toBlob(
                (blob) => {
                    if (!blob) { resolve(file); return; }
                    const resized = new File([blob], file.name, { type: mimeType });
                    resolve(resized);
                },
                mimeType,
                quality
            );
        };
        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Failed to load image for resizing"));
        };
        img.src = url;
    });
}

export default function DetectPage() {
    const [file, setFile] = useState<File | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [resultReady, setResultReady] = useState(false);
    const [viewMode, setViewMode] = useState<"Original" | "ELA" | "Heatmap">("Original");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const [imageUrl, setImageUrl] = useState<string>("");
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);

    const handleUpload = useCallback(async (uploadedFile: File) => {
        setFile(uploadedFile);
        setImageUrl(URL.createObjectURL(uploadedFile));
        setAnalyzing(true);
        setResultReady(false);
        setDetectionResult(null);
        setErrorMessage(null);

        try {
            // Resize the image client-side before uploading to reduce payload
            const optimizedFile = await resizeImageForUpload(uploadedFile);
            const result = await detectionApi.detectImage(optimizedFile);
            setDetectionResult(result);
            setResultReady(true);
        } catch (error: unknown) {
            console.error("Detection failed:", error);
            const msg = error instanceof Error ? error.message : "Unknown error occurred";
            setErrorMessage(`ML model analysis failed: ${msg}. Make sure the ML service is running.`);
            setFile(null);
        } finally {
            setAnalyzing(false);
        }
    }, []);

    useEffect(() => {
        return () => {
            if (imageUrl) URL.revokeObjectURL(imageUrl);
        };
    }, [imageUrl]);

    // Build the real ELA image URL from base64 data
    const elaImageUrl = detectionResult?.details?.ela_image_base64
        ? `data:image/png;base64,${detectionResult.details.ela_image_base64}`
        : null;

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 md:p-8 max-w-7xl mx-auto space-y-8">
            {!file && !analyzing && !resultReady && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col items-center justify-center min-h-[60vh]"
                >
                    <div className="max-w-xl w-full text-center space-y-6">
                        <h1 className="text-4xl font-bold tracking-tight">Detect Deepfakes</h1>
                        <p className="text-muted-foreground text-balance">
                            Upload an image to perform forensic analysis. Supported formats: JPG, PNG, WEBP.
                        </p>

                        <p className="text-xs text-muted-foreground">
                            Powered by SigLIP Vision Transformer with ELA forensic analysis
                        </p>

                        <div className="bg-card p-4 rounded-[2rem] shadow-sm border">
                            <UploadDropzone onUpload={handleUpload} isUploading={false} />
                        </div>

                        {errorMessage && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="p-4 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-sm font-medium"
                            >
                                <div className="flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                    <span>{errorMessage}</span>
                                </div>
                            </motion.div>
                        )}
                    </div>
                </motion.div>
            )}

            {analyzing && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex flex-col items-center justify-center min-h-[60vh] space-y-6"
                >
                    <div className="relative">
                        <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full" />
                        <Loader2 className="w-16 h-16 text-primary animate-spin relative" />
                    </div>
                    <h2 className="text-2xl font-semibold tracking-tight animate-pulse">
                        Analyzing image using forensic AI...
                    </h2>
                    <p className="text-muted-foreground">
                        Running ResNet classifier and ELA CNN analysis
                    </p>
                </motion.div>
            )}

            {resultReady && file && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="grid lg:grid-cols-2 gap-8"
                >
                    <div className="space-y-6">
                        <div className="bg-card p-6 rounded-3xl border shadow-sm flex flex-col items-center">
                            <div className="flex bg-muted/50 p-1 rounded-full mb-6 relative">
                                {["Original", "ELA", "Heatmap"].map((mode) => (
                                    <button
                                        key={mode}
                                        onClick={() => setViewMode(mode as "Original" | "ELA" | "Heatmap")}
                                        className={`px-6 py-2 rounded-full text-sm font-medium transition-all duration-300 ${viewMode === mode
                                            ? "bg-background shadow-md text-foreground"
                                            : "text-muted-foreground hover:text-foreground"
                                            }`}
                                    >
                                        {mode}
                                    </button>
                                ))}
                            </div>
                            <div className="w-full aspect-square relative rounded-2xl overflow-hidden glass-panel flex items-center justify-center bg-muted/10 group">
                                {viewMode === "Original" && (
                                    <Image src={imageUrl} alt="Original uploaded image" className="object-contain" fill />
                                )}
                                {viewMode === "ELA" && (
                                    <div className="absolute inset-0 bg-black flex items-center justify-center">
                                        {elaImageUrl ? (
                                            /* eslint-disable-next-line @next/next/no-img-element */
                                            <img
                                                src={elaImageUrl}
                                                alt="Error Level Analysis"
                                                className="object-contain w-full h-full"
                                            />
                                        ) : (
                                            <p className="text-muted-foreground text-sm">ELA data not available</p>
                                        )}
                                    </div>
                                )}
                                {viewMode === "Heatmap" && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                                        <Image src={imageUrl} alt="Heatmap base" className="object-contain" fill />
                                        <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/40 via-yellow-500/40 to-red-500/60 mix-blend-overlay mix-blend-hard-light filter blur-2xl p-10 mask-image"></div>
                                    </div>
                                )}
                            </div>
                            <p className="text-sm text-muted-foreground mt-4 text-center">
                                {viewMode === "ELA"
                                    ? "Server-generated Error Level Analysis (JPEG recompression difference)"
                                    : viewMode === "Heatmap"
                                        ? "Attention visualization (approximate)"
                                        : "Original uploaded image"}
                            </p>
                        </div>
                    </div>

                    <div className="space-y-8 flex flex-col lg:pt-0">
                        {detectionResult && (
                            <>
                                <ReportCard
                                    result={detectionResult.prediction === "Fake" ? "AI Generated" : "Real"}
                                    confidence={detectionResult.confidence ?? 0}
                                    elaScore={detectionResult.details?.ela_mean !== undefined
                                        ? `Mean: ${Number(detectionResult.details.ela_mean).toFixed(4)}, Std: ${Number(detectionResult.details.ela_std ?? 0).toFixed(4)}`
                                        : "N/A"}
                                    modelName={detectionResult.model_status ?? "Unknown"}
                                    timestamp={new Date().toLocaleString()}
                                />

                                {/* Raw scores breakdown */}
                                {detectionResult.details?.raw_output && (
                                    <div className="bg-card p-6 rounded-3xl border shadow-sm">
                                        <h3 className="text-xl font-bold tracking-tight mb-4">Classification Scores</h3>
                                        <div className="space-y-3">
                                            {detectionResult.details.raw_output.map((item) => (
                                                <div key={item.label} className="flex items-center justify-between bg-muted/30 p-3 rounded-xl border border-border/50">
                                                    <span className="text-sm font-medium text-muted-foreground">{item.label}</span>
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                                                            <div
                                                                className={`h-full rounded-full transition-all duration-1000 ${item.label === "Fake" ? "bg-red-500" : "bg-green-500"}`}
                                                                style={{ width: `${(item.score * 100).toFixed(1)}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-sm font-bold text-foreground w-16 text-right">
                                                            {(item.score * 100).toFixed(2)}%
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-3">
                                            Processing time: {detectionResult.processing_time_ms}ms
                                        </p>
                                    </div>
                                )}

                                <div className="bg-card p-6 rounded-3xl border shadow-sm">
                                    <h3 className="text-xl font-bold tracking-tight mb-2">Explainability</h3>
                                    <p className="text-sm text-muted-foreground mb-6">
                                        Comparing original vs. ELA analysis for forensic insight.
                                    </p>
                                    <HeatmapViewer
                                        originalImage={imageUrl}
                                        heatmapImage={elaImageUrl || imageUrl}
                                    />
                                </div>
                            </>
                        )}

                        <div className="flex gap-4">
                            <button
                                onClick={() => { setFile(null); setResultReady(false); setDetectionResult(null); }}
                                className="flex-1 px-6 py-3 bg-card border hover:bg-muted text-foreground transition-all rounded-xl font-semibold"
                            >
                                Scan Another
                            </button>
                            <button className="flex-1 px-6 py-3 bg-primary text-primary-foreground hover:bg-primary/90 transition-all rounded-xl font-semibold">
                                Generate Official Report
                            </button>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
