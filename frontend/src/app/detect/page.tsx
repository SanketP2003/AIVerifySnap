"use client";

import { useState, useEffect, useCallback } from "react";
import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { ReportCard } from "@/components/shared/ReportCard";
import { HeatmapViewer } from "@/components/shared/HeatmapViewer";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, AlertCircle, Flag, CheckCircle2, Upload, Layers, X, FileImage } from "lucide-react";
import Image from "next/image";
import { detectionApi, DetectionResult } from "@/lib/api";
import { useSearchParams, useRouter } from "next/navigation";

// No client-side resizing — sending the original file preserves compression
// artifacts and pixel-level details that are critical for deepfake detection.
// The ML service handles its own resizing via torchvision transforms.

interface BulkItem {
    file: File;
    url: string;
    status: "pending" | "analyzing" | "done" | "error";
    result?: DetectionResult;
    error?: string;
}

function FeedbackModal({ onClose, onSubmit }: { onClose: () => void; onSubmit: (reason: string) => void }) {
    const [reason, setReason] = useState("");
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = () => {
        onSubmit(reason);
        setSubmitted(true);
        setTimeout(onClose, 1500);
    };

    return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
            <motion.div initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }} className="bg-card border border-border rounded-2xl p-6 max-w-md w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
                {submitted ? (
                    <div className="flex flex-col items-center py-8 space-y-3">
                        <CheckCircle2 className="w-12 h-12 text-green-500" />
                        <h3 className="text-lg font-bold">Thank you for your feedback!</h3>
                        <p className="text-sm text-muted-foreground text-center">Your report helps us improve our detection model.</p>
                    </div>
                ) : (
                    <>
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-bold tracking-tight flex items-center gap-2"><Flag className="w-5 h-5 text-orange-500" /> Report Inaccuracy</h3>
                            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-muted transition-colors"><X className="w-4 h-4" /></button>
                        </div>
                        <p className="text-sm text-muted-foreground mb-4">If you believe this result is incorrect, let us know. This helps retrain our model.</p>
                        <div className="space-y-3 mb-4">
                            {["This image is real but marked as AI-generated", "This image is AI-generated but marked as real", "The confidence score seems wrong", "Other"].map((opt) => (
                                <label key={opt} className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all ${reason === opt ? "border-primary bg-primary/5" : "border-border/50 hover:border-border"}`}>
                                    <input type="radio" name="reason" value={opt} checked={reason === opt} onChange={() => setReason(opt)} className="accent-primary" />
                                    <span className="text-sm font-medium">{opt}</span>
                                </label>
                            ))}
                        </div>
                        <button onClick={handleSubmit} disabled={!reason} className="w-full py-3 bg-foreground text-background font-semibold rounded-xl hover:opacity-90 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                            Submit Feedback
                        </button>
                    </>
                )}
            </motion.div>
        </motion.div>
    );
}

export default function DetectPage() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const isBulkMode = searchParams.get("mode") === "bulk";

    const [mode, setMode] = useState<"single" | "bulk">(isBulkMode ? "bulk" : "single");
    const [file, setFile] = useState<File | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [resultReady, setResultReady] = useState(false);
    const [viewMode, setViewMode] = useState<"Original" | "ELA" | "Heatmap">("Original");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [imageUrl, setImageUrl] = useState<string>("");
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
    const [showFeedback, setShowFeedback] = useState(false);

    // Bulk state
    const [bulkItems, setBulkItems] = useState<BulkItem[]>([]);
    const [bulkProcessing, setBulkProcessing] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    const handleUpload = useCallback(async (uploadedFile: File) => {
        setFile(uploadedFile);
        setImageUrl(URL.createObjectURL(uploadedFile));
        setAnalyzing(true);
        setResultReady(false);
        setDetectionResult(null);
        setErrorMessage(null);
        try {
            const result = await detectionApi.detectImage(uploadedFile);
            setDetectionResult(result);
            setResultReady(true);
        } catch (error: unknown) {
            const msg = error instanceof Error ? error.message : "Unknown error";
            setErrorMessage(`ML model analysis failed: ${msg}. Make sure the ML service is running.`);
            setFile(null);
        } finally {
            setAnalyzing(false);
        }
    }, []);

    const handleBulkFiles = useCallback((files: FileList | File[]) => {
        const items: BulkItem[] = Array.from(files).slice(0, 50).filter(f => f.type.startsWith("image/")).map(f => ({
            file: f, url: URL.createObjectURL(f), status: "pending" as const,
        }));
        setBulkItems(items);
    }, []);

    const processBulk = useCallback(async () => {
        setBulkProcessing(true);
        for (let i = 0; i < bulkItems.length; i++) {
            setBulkItems(prev => prev.map((item, idx) => idx === i ? { ...item, status: "analyzing" } : item));
            try {
                const result = await detectionApi.detectImage(bulkItems[i].file);
                setBulkItems(prev => prev.map((item, idx) => idx === i ? { ...item, status: "done", result } : item));
            } catch (err) {
                const msg = err instanceof Error ? err.message : "Error";
                setBulkItems(prev => prev.map((item, idx) => idx === i ? { ...item, status: "error", error: msg } : item));
            }
        }
        setBulkProcessing(false);
    }, [bulkItems]);

    useEffect(() => { return () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }; }, [imageUrl]);

    const elaImageUrl = detectionResult?.details?.ela_image_base64
        ? `data:image/png;base64,${detectionResult.details.ela_image_base64}` : null;

    const heatmapImageUrl = detectionResult?.details?.ela_heatmap_base64
        ? `data:image/png;base64,${detectionResult.details.ela_heatmap_base64}` : null;

    if (!mounted) {
        return (
            <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 flex flex-col items-center justify-center space-y-3">
                <Loader2 className="w-6 h-6 text-accent animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 space-y-6">
            {/* Mode Toggle */}
            {!file && !analyzing && !resultReady && (
                <div className="flex justify-center pt-4">
                    <div className="flex p-1 rounded-lg border border-border/50" style={{ background: 'hsl(var(--muted) / 0.4)' }}>
                        <button onClick={() => setMode("single")} className={`flex items-center gap-2 px-5 py-2 rounded-md text-[13px] font-medium transition-all ${mode === "single" ? "bg-card shadow-soft text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                            <Upload className="w-3.5 h-3.5" strokeWidth={1.8} /> Single Scan
                        </button>
                        <button onClick={() => setMode("bulk")} className={`flex items-center gap-2 px-5 py-2 rounded-md text-[13px] font-medium transition-all ${mode === "bulk" ? "bg-card shadow-soft text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                            <Layers className="w-3.5 h-3.5" strokeWidth={1.8} /> Bulk Scan
                        </button>
                    </div>
                </div>
            )}

            {/* SINGLE MODE - Upload */}
            {mode === "single" && !file && !analyzing && !resultReady && (
                <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16,1,0.3,1] }} className="flex flex-col items-center justify-center min-h-[55vh]">
                    <div className="max-w-lg w-full text-center space-y-5">
                        <h1 className="text-heading-lg font-serif text-foreground">Detect Deepfakes</h1>
                        <p className="text-body text-muted-foreground text-balance">Upload an image to perform forensic analysis. Supported formats: JPG, PNG, WEBP.</p>
                        <p className="text-caption text-muted-foreground">Dual-stream forensic analysis with Error Level Analysis</p>
                        <div className="card-elevated rounded-xl p-4">
                            <UploadDropzone onUpload={handleUpload} isUploading={false} />
                        </div>
                        {errorMessage && (
                            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="p-4 rounded-2xl bg-destructive/10 border border-destructive/20 text-destructive text-sm font-medium">
                                <div className="flex items-center gap-2"><AlertCircle className="w-4 h-4 flex-shrink-0" /><span>{errorMessage}</span></div>
                            </motion.div>
                        )}
                    </div>
                </motion.div>
            )}

            {/* BULK MODE */}
            {mode === "bulk" && !file && !analyzing && !resultReady && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col items-center justify-center min-h-[55vh]">
                    <div className="max-w-3xl w-full text-center space-y-6">
                        <h1 className="text-4xl font-bold tracking-tight">Bulk Scan</h1>
                        <p className="text-muted-foreground text-balance">Upload up to 50 images at once. Each will be analyzed individually.</p>

                        {bulkItems.length === 0 ? (
                            <div className="bg-card p-4 rounded-[2rem] shadow-sm border">
                                <label className="relative flex flex-col items-center justify-center w-full h-60 rounded-[1.5rem] border-2 border-dashed border-border hover:border-primary/50 hover:bg-muted/50 transition-all cursor-pointer">
                                    <input type="file" multiple accept="image/*" className="hidden" onChange={(e) => e.target.files && handleBulkFiles(e.target.files)} />
                                    <Layers className="w-10 h-10 text-muted-foreground mb-3" />
                                    <p className="text-lg font-semibold">Click to select multiple images</p>
                                    <p className="text-sm text-muted-foreground mt-1">JPG, PNG, WEBP — up to 50 files</p>
                                </label>
                            </div>
                        ) : (
                            <div className="bg-card p-6 rounded-[2rem] shadow-sm border space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium">{bulkItems.length} images selected</span>
                                    <div className="flex gap-2">
                                        <button onClick={() => setBulkItems([])} className="px-4 py-2 text-sm font-medium rounded-lg bg-muted hover:bg-muted/80 transition-colors">Clear</button>
                                        <button onClick={processBulk} disabled={bulkProcessing} className="px-6 py-2 text-sm font-semibold rounded-lg bg-foreground text-background hover:opacity-90 transition-all disabled:opacity-50">
                                            {bulkProcessing ? "Processing..." : "Analyze All"}
                                        </button>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[400px] overflow-y-auto">
                                    {bulkItems.map((item, i) => (
                                        <div key={i} className="relative rounded-xl overflow-hidden border border-border/50 aspect-square bg-muted/20">
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img src={item.url} alt={item.file.name} className="w-full h-full object-cover" />
                                            <div className={`absolute inset-0 flex items-center justify-center ${
                                                item.status === "analyzing" ? "bg-black/50" :
                                                item.status === "done" ? "bg-black/30" :
                                                item.status === "error" ? "bg-red-500/30" : ""
                                            }`}>
                                                {item.status === "analyzing" && <Loader2 className="w-6 h-6 text-white animate-spin" />}
                                                {item.status === "done" && (
                                                    <span className={`px-2 py-1 rounded-lg text-xs font-bold ${item.result?.prediction === "Fake" ? "bg-red-500 text-white" : "bg-green-500 text-white"}`}>
                                                        {item.result?.prediction} {item.result?.confidence?.toFixed(0)}%
                                                    </span>
                                                )}
                                                {item.status === "error" && <AlertCircle className="w-6 h-6 text-white" />}
                                            </div>
                                            <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-2 py-1">
                                                <p className="text-[10px] text-white truncate">{item.file.name}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                {bulkItems.some(i => i.status === "done") && (
                                    <div className="pt-3 border-t border-border/40 text-sm text-muted-foreground">
                                        <span className="font-semibold text-foreground">{bulkItems.filter(i => i.status === "done").length}</span> analyzed •{" "}
                                        <span className="text-red-500 font-semibold">{bulkItems.filter(i => i.result?.prediction === "Fake").length}</span> fake •{" "}
                                        <span className="text-green-500 font-semibold">{bulkItems.filter(i => i.result?.prediction === "Real").length}</span> real
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </motion.div>
            )}

            {/* Analyzing spinner */}
            {analyzing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center min-h-[60vh] space-y-5">
                    <div className="relative">
                        <div className="absolute inset-0 bg-accent/15 blur-xl rounded-full" />
                        <Loader2 className="w-12 h-12 text-accent animate-spin relative" />
                    </div>
                    <h2 className="text-heading font-semibold text-foreground animate-pulse">Analyzing image using forensic AI...</h2>
                    <p className="text-caption text-muted-foreground">Running forensic analysis with ELA compression detection</p>
                </motion.div>
            )}

            {/* Results */}
            {resultReady && file && (
                <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16,1,0.3,1] }} className="grid lg:grid-cols-2 gap-6">
                    <div className="space-y-5">
                        <div className="card-elevated rounded-2xl p-5 flex flex-col items-center">
                            <div className="flex p-1 rounded-lg mb-5" style={{ background: 'hsl(var(--muted) / 0.4)' }}>
                                {["Original", "ELA", "Heatmap"].map((m) => (
                                    <button key={m} onClick={() => setViewMode(m as "Original" | "ELA" | "Heatmap")}
                                        className={`px-5 py-1.5 rounded-md text-[13px] font-medium transition-all duration-200 ${viewMode === m ? "bg-card shadow-soft text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                                        {m}
                                    </button>
                                ))}
                            </div>
                            <div className="w-full aspect-square relative rounded-2xl overflow-hidden glass-panel flex items-center justify-center bg-muted/10">
                                {viewMode === "Original" && <Image src={imageUrl} alt="Original" className="object-contain" fill />}
                                {viewMode === "ELA" && (
                                    <div className="absolute inset-0 bg-black flex items-center justify-center">
                                        {elaImageUrl ? (
                                            /* eslint-disable-next-line @next/next/no-img-element */
                                            <img src={elaImageUrl} alt="ELA" className="object-contain w-full h-full" />
                                        ) : <p className="text-muted-foreground text-sm">ELA data not available</p>}
                                    </div>
                                )}
                                {viewMode === "Heatmap" && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                                        {heatmapImageUrl ? (
                                            /* eslint-disable-next-line @next/next/no-img-element */
                                            <img src={heatmapImageUrl} alt="Forensic Heatmap" className="object-contain w-full h-full" />
                                        ) : <p className="text-muted-foreground text-sm">Heatmap data not available</p>}
                                    </div>
                                )}
                            </div>
                            <p className="text-caption text-muted-foreground mt-3.5 text-center">
                                {viewMode === "ELA" ? "Server-generated Error Level Analysis" : viewMode === "Heatmap" ? "JET-colorized ELA forensic heatmap" : "Original uploaded image"}
                            </p>
                        </div>
                    </div>

                    <div className="space-y-8 flex flex-col">
                        {detectionResult && (
                            <>
                                <ReportCard
                                    result={detectionResult.prediction === "Fake" ? "AI Generated" : "Real"}
                                    confidence={detectionResult.confidence ?? 0}
                                    elaScore={detectionResult.details?.ela_mean !== undefined
                                        ? `Mean: ${Number(detectionResult.details.ela_mean).toFixed(4)}, Std: ${Number(detectionResult.details.ela_std ?? 0).toFixed(4)}` : "N/A"}
                                    modelName={detectionResult.model_status ?? "Unknown"}
                                    timestamp={new Date().toLocaleString()}
                                />

                                {detectionResult.details?.raw_output && (
                                    <div className="card-elevated rounded-2xl p-5">
                                        <h3 className="text-heading font-semibold mb-4">Classification Scores</h3>
                                        <div className="space-y-3">
                                            {detectionResult.details.raw_output.map((item) => (
                                                <div key={item.label} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'hsl(var(--muted) / 0.3)' }}>
                                                    <span className="text-sm font-medium text-muted-foreground">{item.label}</span>
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                                                            <div className={`h-full rounded-full transition-all duration-1000 ${item.label === "Fake" ? "bg-red-500" : "bg-green-500"}`} style={{ width: `${(item.score * 100).toFixed(1)}%` }} />
                                                        </div>
                                                        <span className="text-sm font-bold text-foreground w-16 text-right">{(item.score * 100).toFixed(2)}%</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted-foreground mt-3">Processing time: {detectionResult.processing_time_ms}ms</p>
                                    </div>
                                )}

                                <div className="card-elevated rounded-2xl p-5">
                                    <h3 className="text-heading font-semibold mb-1.5">Explainability</h3>
                                    <p className="text-caption text-muted-foreground mb-5">Comparing original vs. ELA analysis for forensic insight.</p>
                                    <HeatmapViewer originalImage={imageUrl} heatmapImage={heatmapImageUrl || elaImageUrl || imageUrl} />
                                </div>
                            </>
                        )}

                        <div className="flex gap-2.5">
                            <button onClick={() => { setFile(null); setResultReady(false); setDetectionResult(null); }} className="flex-1 py-2.5 card-elevated rounded-lg text-[14px] font-semibold text-foreground hover:bg-muted/50 transition-all">
                                Scan Another
                            </button>
                            <button onClick={() => setShowFeedback(true)} className="flex items-center justify-center gap-1.5 px-4 py-2.5 bg-orange-500/8 text-orange-600 dark:text-orange-400 border border-orange-500/15 hover:bg-orange-500/15 transition-all rounded-lg text-[13px] font-semibold">
                                <Flag className="w-3.5 h-3.5" /> Report
                            </button>
                            <button
                                onClick={() => {
                                    if (detectionResult?.scan_id) {
                                        router.push(`/report/${detectionResult.scan_id}`);
                                    }
                                }}
                                disabled={!detectionResult?.scan_id}
                                className="flex-1 btn-primary py-2.5 text-[14px] rounded-lg disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                Generate Report
                            </button>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Feedback Modal */}
            <AnimatePresence>
                {showFeedback && <FeedbackModal onClose={() => setShowFeedback(false)} onSubmit={(reason) => console.log("Feedback:", reason)} />}
            </AnimatePresence>
        </div>
    );
}
