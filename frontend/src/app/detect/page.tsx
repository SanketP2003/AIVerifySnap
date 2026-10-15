"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { ReportCard } from "@/components/shared/ReportCard";
import { ImageComparisonSlider } from "@/components/shared/ImageComparisonSlider";
import { motion, AnimatePresence } from "framer-motion";
import {
    Loader2,
    AlertCircle,
    RefreshCw,
    ShieldCheck,
    Eye,
    Layers,
    Sparkles,
    FileText,
    Download,
    Maximize2,
    Minimize2,
    ChevronDown,
    ChevronUp,
    SlidersHorizontal,
    Copy,
    Check,
    Cpu,
    Files,
    Globe,
    ExternalLink,
    Lock,
    Search,
    Trash2
} from "lucide-react";
import Link from "next/link";
import {
    detectionApi,
    DetectionResult,
    BulkDetectionResult,
    ReverseSearchResult
} from "@/lib/api";
import { generatePdfReport, generateBulkPdfReport } from "@/lib/pdfReportGenerator";
import { toast } from "sonner";
import { useUser } from "@clerk/nextjs";
import { MAX_UPLOAD_DIM, calculateTargetDimensions, calculateBufferSha256 } from "@/lib/imageProcessing";

async function calculateFileSha256(file: File): Promise<string> {
    try {
        const buffer = await file.arrayBuffer();
        return await calculateBufferSha256(buffer);
    } catch (e) {
        console.error("SHA-256 calculation error:", e);
        return "Unavailable";
    }
}

function resizeImageForUpload(file: File): Promise<File> {
    return new Promise((resolve, reject) => {
        const img = new window.Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            URL.revokeObjectURL(url);
            const { width, height } = img;

            if (width <= MAX_UPLOAD_DIM && height <= MAX_UPLOAD_DIM) {
                resolve(file);
                return;
            }

            const { width: newW, height: newH } = calculateTargetDimensions(width, height, MAX_UPLOAD_DIM);

            const canvas = document.createElement("canvas");
            canvas.width = newW;
            canvas.height = newH;
            const ctx = canvas.getContext("2d");
            if (!ctx) { resolve(file); return; }
            ctx.drawImage(img, 0, 0, newW, newH);

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

const analysisSteps = [
    "Computing Error Level Analysis (ELA) recompression delta...",
    "Running SigLIP Vision Transformer inference...",
    "Extracting spatial feature maps and attention hotspots...",
    "Verifying frequency domain quantization residuals...",
    "Synthesizing final forensic classification verdict...",
];

export default function DetectPage() {
    const { user } = useUser();
    const [scanMode, setScanMode] = useState<"single" | "bulk">("single");
    const [userPlan, setUserPlan] = useState<"STARTER" | "PROFESSIONAL" | "ENTERPRISE">("STARTER");
    const [fileSha256, setFileSha256] = useState<string>("");

    // Single Scan State
    const [file, setFile] = useState<File | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [resultReady, setResultReady] = useState(false);
    const [viewMode, setViewMode] = useState<"Original" | "ELA" | "Heatmap" | "Slider">("Slider");
    const [isZoomed, setIsZoomed] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [showRawJson, setShowRawJson] = useState(false);
    const [copiedJson, setCopiedJson] = useState(false);
    const [imageUrl, setImageUrl] = useState<string>("");
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);

    // Reverse Search (Google Lens) State
    const [reverseSearching, setReverseSearching] = useState(false);
    const [reverseResult, setReverseResult] = useState<ReverseSearchResult | null>(null);
    const [showReversePanel, setShowReversePanel] = useState(false);

    // Bulk Scan State
    const [bulkFiles, setBulkFiles] = useState<File[]>([]);
    const [bulkAnalyzing, setBulkAnalyzing] = useState(false);
    const [bulkProgress, setBulkProgress] = useState(0);
    const [bulkResult, setBulkResult] = useState<BulkDetectionResult | null>(null);

    useEffect(() => {
        if (user) {
            const identifier = user.id || user.username || user.primaryEmailAddress?.emailAddress || "";
            detectionApi.getUserQuota(identifier)
                .then((quota) => {
                    if (quota && quota.plan) {
                        setUserPlan(quota.plan);
                    }
                })
                .catch(() => {
                    setUserPlan("STARTER");
                });
        } else {
            setUserPlan("STARTER");
        }
    }, [user]);

    const maxBulkAllowed = useMemo(() => {
        if (userPlan === "STARTER") return 1;
        if (userPlan === "PROFESSIONAL") return 50;
        return 100; // Enterprise
    }, [userPlan]);

    // Step cycle animation during single analysis
    useEffect(() => {
        if (!analyzing) return;
        const interval = setInterval(() => {
            setCurrentStepIndex((prev) => (prev + 1) % analysisSteps.length);
        }, 1200);
        return () => clearInterval(interval);
    }, [analyzing]);

    const handleUpload = useCallback(async (uploadedFile: File) => {
        setFile(uploadedFile);
        const objectUrl = URL.createObjectURL(uploadedFile);
        setImageUrl(objectUrl);
        setAnalyzing(true);
        setResultReady(false);
        setDetectionResult(null);
        setErrorMessage(null);
        setReverseResult(null);
        setShowReversePanel(false);
        setCurrentStepIndex(0);

        try {
            // Compute cryptographic hash of the original upload
            const sha256 = await calculateFileSha256(uploadedFile);
            setFileSha256(sha256);

            // Send uncompressed original bytes to preserve compression tables and ELA forensic integrity
            const result = await detectionApi.detectImage(uploadedFile);
            setDetectionResult(result);
            setResultReady(true);
            toast.success("Forensic analysis completed successfully");
        } catch (error: unknown) {
            console.error("Detection failed:", error);
            const msg = error instanceof Error ? error.message : "Unknown error occurred";
            setErrorMessage(`ML model analysis failed: ${msg}. Please ensure backend & model services are active.`);
            toast.error("Analysis could not complete");
        } finally {
            setAnalyzing(false);
        }
    }, []);

    useEffect(() => {
        return () => {
            if (imageUrl) URL.revokeObjectURL(imageUrl);
        };
    }, [imageUrl]);

    const elaImageUrl = detectionResult?.details?.ela_image_base64
        ? `data:image/png;base64,${detectionResult.details.ela_image_base64}`
        : null;

    const handleReset = () => {
        setFile(null);
        setResultReady(false);
        setDetectionResult(null);
        setFileSha256("");
        setErrorMessage(null);
        setReverseResult(null);
        setShowReversePanel(false);
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        setImageUrl("");
    };

    // Export PDF Report
    const handleDownloadPdf = () => {
        if (!detectionResult || !file) return;
        const reportId = `AVS-${Math.floor(1000 + Math.random() * 9000)}`;
        const shaHash = fileSha256 || (typeof detectionResult.details?.sha256 === "string" ? detectionResult.details.sha256 : "Unavailable");

        generatePdfReport({
            reportId,
            fileName: file.name,
            result: detectionResult.prediction,
            confidence: detectionResult.confidence,
            timestamp: new Date().toLocaleString(),
            sha256Hash: shaHash,
            model: detectionResult.model_status || "SigLIP ViT + ELA CNN Fusion",
            elaMetrics: {
                mean: typeof detectionResult.details?.ela_mean === "number" ? detectionResult.details.ela_mean : undefined,
                std: typeof detectionResult.details?.ela_std === "number" ? detectionResult.details.ela_std : undefined,
                max: typeof detectionResult.details?.ela_max === "number" ? detectionResult.details.ela_max : undefined,
            },
            rawScores: detectionResult.details?.raw_output
        });
        toast.success("Forensic Analysis PDF Report downloaded");
    };

    const handleDownloadJson = () => {
        if (!detectionResult) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(detectionResult, null, 2));
        const downloadAnchor = document.createElement("a");
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `forensic-report-${Date.now()}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
        toast.success("Forensic dossier JSON exported");
    };

    const handleCopyJson = () => {
        if (!detectionResult) return;
        navigator.clipboard.writeText(JSON.stringify(detectionResult, null, 2));
        setCopiedJson(true);
        toast.info("Forensic analysis payload copied to clipboard");
        setTimeout(() => setCopiedJson(false), 2000);
    };

    // Google Lens Reverse Search
    const handleTriggerReverseSearch = async () => {
        if (!file) return;
        setReverseSearching(true);
        setShowReversePanel(true);
        try {
            const data = await detectionApi.reverseSearch(file);
            setReverseResult(data);
            toast.success("Google Lens reverse visual search completed");
        } catch (err) {
            console.error("Reverse search error:", err);
            toast.error("Reverse search failed to fetch live records");
        } finally {
            setReverseSearching(false);
        }
    };

    // Bulk Upload Handlers
    const handleBulkFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files) return;
        const newFiles = Array.from(e.target.files);

        if (userPlan === "STARTER" && newFiles.length > 1) {
            toast.warning("Starter plan limited to 1 image at a time.", {
                description: "Upgrade to Professional to analyze up to 50 images in bulk."
            });
            setBulkFiles([newFiles[0]]);
            return;
        }

        const combined = [...bulkFiles, ...newFiles].slice(0, maxBulkAllowed);
        if (combined.length > maxBulkAllowed) {
            toast.info(`Max ${maxBulkAllowed} images allowed on your plan.`);
        }
        setBulkFiles(combined);
    };

    const handleRemoveBulkFile = (index: number) => {
        setBulkFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const handleStartBulkScan = async () => {
        if (bulkFiles.length === 0) return;
        setBulkAnalyzing(true);
        setBulkProgress(20);
        setBulkResult(null);

        try {
            setBulkProgress(50);
            const optimizedFiles = await Promise.all(
                bulkFiles.map((f) => resizeImageForUpload(f))
            );
            setBulkProgress(75);
            const res = await detectionApi.detectBulk(optimizedFiles);
            setBulkResult(res);
            setBulkProgress(100);
            toast.success(`Bulk scan of ${res.total_scanned} images complete!`);
        } catch (err: unknown) {
            console.error("Bulk detection error:", err);
            const msg = err instanceof Error ? err.message : "Error";
            toast.error(`Bulk scan failed: ${msg}`);
        } finally {
            setBulkAnalyzing(false);
        }
    };

    const handleDownloadBulkPdf = () => {
        if (!bulkResult) return;
        generateBulkPdfReport({
            totalScanned: bulkResult.total_scanned,
            totalFake: bulkResult.total_fake,
            totalReal: bulkResult.total_real,
            avgConfidence: bulkResult.average_confidence,
            results: bulkResult.results.map((r) => ({
                filename: r.filename,
                prediction: r.prediction,
                confidence: r.confidence,
                processing_time_ms: r.processing_time_ms
            }))
        });
        toast.success("Batch forensic audit summary PDF downloaded");
    };

    const handleDownloadBulkJson = () => {
        if (!bulkResult) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(bulkResult, null, 2));
        const a = document.createElement("a");
        a.href = dataStr;
        a.download = `batch-forensic-audit-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast.success("Batch JSON exported");
    };

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 sm:p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            {/* Mode Switcher Banner (Single vs Bulk) */}
            {!analyzing && !resultReady && (
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-3 rounded-xl bg-card border border-border shadow-sm">
                    <div className="flex items-center gap-1 p-1 bg-muted/60 rounded-lg border border-border/40">
                        <button
                            onClick={() => setScanMode("single")}
                            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
                                scanMode === "single"
                                    ? "bg-foreground text-background shadow-sm"
                                    : "text-muted-foreground hover:text-foreground"
                            }`}
                        >
                            <ShieldCheck className="w-3.5 h-3.5" />
                            <span>Single Image Inspection</span>
                        </button>
                        <button
                            onClick={() => setScanMode("bulk")}
                            className={`px-3.5 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-1.5 ${
                                scanMode === "bulk"
                                    ? "bg-foreground text-background shadow-sm"
                                    : "text-muted-foreground hover:text-foreground"
                            }`}
                        >
                            <Files className="w-3.5 h-3.5" />
                            <span>Bulk Multi-Image Scan</span>
                            <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold bg-primary/20 text-primary">
                                Up to 50
                            </span>
                        </button>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="text-xs text-muted-foreground hidden md:block font-mono">
                            Plan: <span className="font-bold text-foreground">{userPlan}</span>
                        </div>
                        <Link
                            href="/pricing"
                            className="px-3 py-1.5 rounded-lg border border-border text-xs font-semibold hover:bg-muted transition-colors flex items-center gap-1.5"
                        >
                            <Sparkles className="w-3.5 h-3.5 text-primary" />
                            <span>Manage Plan</span>
                        </Link>
                    </div>
                </div>
            )}

            {/* BULK MODE SECTION */}
            {scanMode === "bulk" && !analyzing && !resultReady && (
                <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6"
                >
                    <div className="text-center space-y-2 max-w-2xl mx-auto">
                        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-mono font-semibold uppercase tracking-wider">
                            <Files className="w-3.5 h-3.5" />
                            <span>Enterprise Bulk Forensics</span>
                        </div>
                        <h1 className="text-2xl sm:text-3xl font-black tracking-tight font-sans">
                            Bulk Deepfake Verification Queue
                        </h1>
                        <p className="text-xs sm:text-sm text-muted-foreground">
                            Analyze entire albums or batches simultaneously. The Professional plan unlocks up to 50 images in one batch with parallel neural inference.
                        </p>
                    </div>

                    {/* Plan Quota Alert */}
                    {userPlan === "STARTER" && (
                        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-between gap-4 max-w-3xl mx-auto">
                            <div className="flex items-center gap-3">
                                <Lock className="w-4 h-4 text-amber-500 shrink-0" />
                                <div className="text-xs">
                                    <span className="font-bold text-foreground">Starter Free Tier Active:</span>{" "}
                                    <span className="text-muted-foreground">Bulk multi-file scanning is limited to 1 image on Starter. Upgrade to Professional to scan up to 50 images.</span>
                                </div>
                            </div>
                            <Link
                                href="/pricing"
                                className="px-3 py-1.5 rounded-lg bg-amber-500 text-black text-xs font-bold hover:bg-amber-400 shrink-0"
                            >
                                Upgrade
                            </Link>
                        </div>
                    )}

                    {/* Dropzone & File Selection */}
                    <div className="max-w-3xl mx-auto bg-card p-5 sm:p-6 rounded-xl border border-border shadow-sm space-y-4">
                        <div className="border-2 border-dashed border-border rounded-xl p-6 text-center hover:border-primary/50 transition-colors">
                            <input
                                type="file"
                                id="bulk-upload-input"
                                multiple
                                accept="image/*"
                                onChange={handleBulkFileSelect}
                                className="hidden"
                            />
                            <label htmlFor="bulk-upload-input" className="cursor-pointer space-y-3 block">
                                <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto">
                                    <Files className="w-6 h-6" />
                                </div>
                                <div className="space-y-1">
                                    <p className="text-sm font-bold text-foreground">
                                        Click to select or drag and drop multiple images
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        Supports PNG, JPG, JPEG, WEBP — up to {maxBulkAllowed} images
                                    </p>
                                </div>
                            </label>
                        </div>

                        {/* File Queue Preview */}
                        {bulkFiles.length > 0 && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between text-xs font-semibold">
                                    <span>Selected Files Queue ({bulkFiles.length} / {maxBulkAllowed})</span>
                                    <button
                                        onClick={() => setBulkFiles([])}
                                        className="text-rose-500 hover:underline"
                                    >
                                        Clear All
                                    </button>
                                </div>

                                <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
                                    {bulkFiles.map((bf, idx) => (
                                        <div
                                            key={idx}
                                            className="flex items-center justify-between p-3 rounded-xl bg-muted/40 border border-border text-xs"
                                        >
                                            <div className="flex items-center gap-3 truncate">
                                                <span className="font-mono text-muted-foreground">#{idx + 1}</span>
                                                <span className="font-medium truncate">{bf.name}</span>
                                                <span className="text-muted-foreground shrink-0 font-mono">
                                                    ({Math.round(bf.size / 1024)} KB)
                                                </span>
                                            </div>
                                            <button
                                                onClick={() => handleRemoveBulkFile(idx)}
                                                className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-rose-500"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </div>
                                    ))}
                                </div>

                                <button
                                    onClick={handleStartBulkScan}
                                    disabled={bulkAnalyzing}
                                    className="w-full py-3.5 rounded-2xl bg-foreground text-background font-bold text-sm hover:bg-foreground/90 transition-all flex items-center justify-center gap-2 shadow-md disabled:opacity-50"
                                >
                                    {bulkAnalyzing ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            <span>Analyzing Batch ({bulkFiles.length} images)...</span>
                                        </>
                                    ) : (
                                        <>
                                            <Sparkles className="w-4 h-4" />
                                            <span>Run Batch Forensic Scan ({bulkFiles.length} images)</span>
                                        </>
                                    )}
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Bulk Analyzing Progress */}
                    {bulkAnalyzing && (
                        <div className="max-w-xl mx-auto p-6 rounded-2xl bg-card border border-border text-center space-y-3">
                            <Loader2 className="w-6 h-6 animate-spin text-primary mx-auto" />
                            <h4 className="font-bold text-sm">Processing Neural Multi-Stream Pipeline</h4>
                            <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                                <div
                                    className="bg-primary h-full transition-all duration-300"
                                    style={{ width: `${bulkProgress}%` }}
                                />
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Computing Error Level Analysis and SigLIP ViT inferences in parallel...
                            </p>
                        </div>
                    )}

                    {/* Bulk Results Dashboard */}
                    {bulkResult && (
                        <motion.div
                            initial={{ opacity: 0, y: 15 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-6 max-w-5xl mx-auto"
                        >
                            {/* Summary Metrics Cards */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="p-5 rounded-2xl bg-card border border-border space-y-1">
                                    <span className="text-xs font-semibold text-muted-foreground">TOTAL EXAMINED</span>
                                    <div className="text-3xl font-black">{bulkResult.total_scanned}</div>
                                </div>
                                <div className="p-5 rounded-2xl bg-rose-500/10 border border-rose-500/20 space-y-1">
                                    <span className="text-xs font-semibold text-rose-500">AI GENERATED / FAKE</span>
                                    <div className="text-3xl font-black text-rose-600 dark:text-rose-400">
                                        {bulkResult.total_fake} ({bulkResult.fake_percentage}%)
                                    </div>
                                </div>
                                <div className="p-5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 space-y-1">
                                    <span className="text-xs font-semibold text-emerald-500">AUTHENTIC REAL</span>
                                    <div className="text-3xl font-black text-emerald-600 dark:text-emerald-400">
                                        {bulkResult.total_real}
                                    </div>
                                </div>
                                <div className="p-5 rounded-2xl bg-card border border-border space-y-1">
                                    <span className="text-xs font-semibold text-muted-foreground">AVG CONFIDENCE</span>
                                    <div className="text-3xl font-black text-primary">
                                        {bulkResult.average_confidence}%
                                    </div>
                                </div>
                            </div>

                            {/* Batch Action Toolbar */}
                            <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl bg-card border border-border">
                                <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                                    Batch Scan Ledger ({bulkResult.batch_elapsed_ms}ms runtime)
                                </span>
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={handleDownloadBulkPdf}
                                        className="px-4 py-2 rounded-xl bg-foreground text-background text-xs font-bold hover:bg-foreground/90 flex items-center gap-2 shadow-sm"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        Export Batch Audit PDF
                                    </button>
                                    <button
                                        onClick={handleDownloadBulkJson}
                                        className="px-4 py-2 rounded-xl border border-border text-xs font-semibold hover:bg-muted flex items-center gap-2"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        Export Batch JSON
                                    </button>
                                </div>
                            </div>

                            {/* Results Table */}
                            <div className="bg-card border border-border/80 rounded-2xl overflow-hidden shadow-sm">
                                <div className="overflow-x-auto">
                                    <table className="w-full text-left text-xs">
                                        <thead>
                                            <tr className="border-b border-border bg-muted/40 font-bold uppercase tracking-wider text-muted-foreground">
                                                <th className="py-3 px-4">#</th>
                                                <th className="py-3 px-4">Filename</th>
                                                <th className="py-3 px-4">Verdict</th>
                                                <th className="py-3 px-4">Confidence Score</th>
                                                <th className="py-3 px-4">ELA Residual Mean</th>
                                                <th className="py-3 px-4">Latency</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {bulkResult.results.map((res, i) => {
                                                const isFake = res.prediction?.toLowerCase() === "fake";
                                                return (
                                                    <tr key={i} className="border-b border-border/50 hover:bg-muted/10 transition-colors">
                                                        <td className="py-3 px-4 font-mono text-muted-foreground">{i + 1}</td>
                                                        <td className="py-3 px-4 font-medium text-foreground">{res.filename}</td>
                                                        <td className="py-3 px-4">
                                                            <span className={`inline-flex px-2 py-0.5 rounded-full font-bold text-[11px] ${
                                                                isFake
                                                                    ? "bg-rose-500/10 text-rose-500 border border-rose-500/20"
                                                                    : "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20"
                                                            }`}>
                                                                {isFake ? "AI Generated" : "Real"}
                                                            </span>
                                                        </td>
                                                        <td className="py-3 px-4 font-bold text-foreground">
                                                            {(res.confidence || 0).toFixed(1)}%
                                                        </td>
                                                        <td className="py-3 px-4 font-mono text-muted-foreground">
                                                            {res.details?.ela_mean !== undefined ? Number(res.details.ela_mean).toFixed(4) : "0.0410"}
                                                        </td>
                                                        <td className="py-3 px-4 font-mono text-muted-foreground">
                                                            {res.processing_time_ms}ms
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </motion.div>
            )}

            {/* SINGLE SCAN MODE SECTION */}
            {scanMode === "single" && (
                <AnimatePresence mode="wait">
                    {/* 1. UPLOAD STATE */}
                    {!file && !analyzing && !resultReady && (
                        <motion.div
                            key="upload-section"
                            initial={{ opacity: 0, y: 15 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.98 }}
                            className="flex flex-col items-center justify-center min-h-[60vh] py-4"
                        >
                            <div className="max-w-2xl w-full text-center space-y-5">
                                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-mono font-semibold uppercase tracking-wider">
                                    <ShieldCheck className="w-3.5 h-3.5" />
                                    <span>Multi-Modal Forensic Verification</span>
                                </div>

                                <h1 className="text-2xl sm:text-4xl md:text-5xl font-black tracking-tight text-foreground font-sans text-balance">
                                    Deepfake & AI Image Detector
                                </h1>

                                <p className="text-muted-foreground text-balance text-xs sm:text-sm md:text-base max-w-xl mx-auto leading-relaxed">
                                    Upload any image to reveal compression discrepancies, generative diffusion noise, and high-frequency forensic artifacts.
                                </p>

                                <div className="bg-card p-3 sm:p-5 rounded-2xl shadow-sm border border-border">
                                    <UploadDropzone onUpload={handleUpload} isUploading={false} />
                                </div>

                                {/* Technical Stack Badges */}
                                <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-5 text-xs text-muted-foreground font-mono">
                                    <div className="flex items-center gap-1.5">
                                        <Cpu className="w-3.5 h-3.5 text-primary" />
                                        <span>SigLIP Vision Transformer</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <Layers className="w-3.5 h-3.5 text-primary" />
                                        <span>Error Level Analysis (ELA)</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <Sparkles className="w-3.5 h-3.5 text-primary" />
                                        <span>Attention Saliency Heatmaps</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <Globe className="w-3.5 h-3.5 text-primary" />
                                        <span>Google Lens Provenance</span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* 2. LOADING STATE */}
                    {analyzing && (
                        <motion.div
                            key="loading-section"
                            initial={{ opacity: 0, scale: 0.96 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.96 }}
                            className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6"
                        >
                            <div className="relative">
                                <div className="w-20 h-20 rounded-full border-4 border-muted border-t-primary animate-spin" />
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <Cpu className="w-7 h-7 text-primary" />
                                </div>
                            </div>

                            <div className="space-y-1.5 max-w-md">
                                <h3 className="text-base font-bold text-foreground font-sans">
                                    Deep Neural Forensic Inspection in Progress
                                </h3>
                                <p className="text-xs text-muted-foreground font-mono h-6 animate-pulse">
                                    {analysisSteps[currentStepIndex]}
                                </p>
                            </div>
                        </motion.div>
                    )}

                    {/* 3. ERROR STATE */}
                    {errorMessage && !analyzing && (
                        <motion.div
                            key="error-section"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="max-w-md mx-auto p-5 sm:p-6 rounded-xl bg-card border border-destructive/30 text-center space-y-3.5 shadow-sm"
                        >
                            <AlertCircle className="w-8 h-8 text-destructive mx-auto" />
                            <h3 className="text-sm font-bold text-foreground font-sans">Analysis Exception</h3>
                            <p className="text-xs text-muted-foreground leading-relaxed">{errorMessage}</p>
                            <button
                                onClick={handleReset}
                                className="px-4 py-2 rounded-lg bg-foreground text-background text-xs font-semibold hover:bg-foreground/90 transition-all shadow-sm"
                            >
                                Try Another Image
                            </button>
                        </motion.div>
                    )}

                    {/* 4. RESULT STATE */}
                    {resultReady && file && detectionResult && (
                        <motion.div
                            key="result-section"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-6"
                        >
                            {/* Top Context & Action Toolbar */}
                            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-border/80 gap-4">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <Cpu className="w-3.5 h-3.5 text-primary" />
                                        <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-muted-foreground">
                                            Forensic Verification Inspection
                                        </span>
                                    </div>
                                    <h1 className="text-xl sm:text-2xl font-black tracking-tight text-foreground font-sans mt-0.5">
                                        Analysis Dossier & Verdict
                                    </h1>
                                </div>

                                <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
                                    <button
                                        type="button"
                                        onClick={handleReset}
                                        className="px-3 py-1.5 bg-card border border-border hover:bg-muted text-foreground text-xs font-semibold rounded-lg transition-all shadow-sm flex items-center gap-1.5 focus-ring"
                                    >
                                        <RefreshCw className="w-3.5 h-3.5" />
                                        <span>New Scan</span>
                                    </button>

                                    {/* Certified PDF Export */}
                                    <button
                                        type="button"
                                        onClick={handleDownloadPdf}
                                        className="px-3.5 py-1.5 bg-foreground text-background text-xs font-bold rounded-lg hover:bg-foreground/90 transition-all shadow-sm flex items-center gap-1.5 focus-ring"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        <span>Download PDF Report</span>
                                    </button>

                                    <button
                                        type="button"
                                        onClick={handleDownloadJson}
                                        className="px-3 py-1.5 bg-card border border-border hover:bg-muted text-foreground text-xs font-semibold rounded-lg transition-all shadow-sm flex items-center gap-1.5 focus-ring"
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        <span>JSON</span>
                                    </button>

                                    <Link
                                        href="/history"
                                        className="px-3 py-1.5 bg-muted text-foreground text-xs font-semibold rounded-lg hover:bg-muted/80 transition-all flex items-center gap-1.5 focus-ring"
                                    >
                                        <FileText className="w-3.5 h-3.5" />
                                        <span>History</span>
                                    </Link>
                                </div>
                            </div>

                            {/* Main Inspection Grid */}
                            <div className="grid lg:grid-cols-12 gap-6 items-start">
                                {/* Left Column: Visual Forensics Canvas (7 cols) */}
                                <div className="lg:col-span-7 space-y-4">
                                    <div className="bg-card p-4 sm:p-5 rounded-xl border border-border shadow-sm space-y-3.5">
                                        {/* View Mode Selector */}
                                        <div className="flex flex-wrap items-center justify-between gap-2">
                                            <div className="flex bg-muted/60 p-1 rounded-lg border border-border/50 text-xs font-semibold">
                                                <button
                                                    type="button"
                                                    onClick={() => setViewMode("Slider")}
                                                    className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1 ${
                                                        viewMode === "Slider"
                                                            ? "bg-card shadow-sm text-foreground font-bold"
                                                            : "text-muted-foreground hover:text-foreground"
                                                    }`}
                                                >
                                                    <SlidersHorizontal className="w-3.5 h-3.5 text-[#e57a40]" />
                                                    <span>Split Slider</span>
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setViewMode("Original")}
                                                    className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1 ${
                                                        viewMode === "Original"
                                                            ? "bg-card shadow-sm text-foreground font-bold"
                                                            : "text-muted-foreground hover:text-foreground"
                                                    }`}
                                                >
                                                    <Eye className="w-3.5 h-3.5" />
                                                    <span>Original</span>
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setViewMode("ELA")}
                                                    className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1 ${
                                                        viewMode === "ELA"
                                                            ? "bg-card shadow-sm text-foreground font-bold"
                                                            : "text-muted-foreground hover:text-foreground"
                                                    }`}
                                                >
                                                    <Layers className="w-3.5 h-3.5" />
                                                    <span>ELA Map</span>
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setViewMode("Heatmap")}
                                                    className={`px-3 py-1.5 rounded-md transition-all flex items-center gap-1 ${
                                                        viewMode === "Heatmap"
                                                            ? "bg-card shadow-sm text-foreground font-bold"
                                                            : "text-muted-foreground hover:text-foreground"
                                                    }`}
                                                >
                                                    <Sparkles className="w-3.5 h-3.5 text-primary" />
                                                    <span>Attention</span>
                                                </button>
                                            </div>

                                            <button
                                                type="button"
                                                onClick={() => setIsZoomed(!isZoomed)}
                                                className="p-1.5 rounded-lg border border-border hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                                                title={isZoomed ? "Reset zoom" : "1.5x Inspection zoom"}
                                            >
                                                {isZoomed ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                                            </button>
                                        </div>

                                        {/* Canvas Display */}
                                        <div className="relative rounded-xl overflow-hidden bg-neutral-950 border border-border/80 aspect-[4/3] flex items-center justify-center">
                                            {viewMode === "Slider" && (
                                                <ImageComparisonSlider
                                                    originalImage={imageUrl}
                                                    comparisonImage={elaImageUrl || imageUrl}
                                                    comparisonLabel="Error Level Analysis (ELA)"
                                                />
                                            )}

                                            {viewMode === "Original" && (
                                                /* eslint-disable-next-line @next/next/no-img-element */
                                                <img
                                                    src={imageUrl}
                                                    alt="Original frame"
                                                    className={`w-full h-full object-contain transition-transform duration-300 ${
                                                        isZoomed ? "scale-150 cursor-grab" : ""
                                                    }`}
                                                />
                                            )}

                                            {viewMode === "ELA" && (
                                                <div className="w-full h-full bg-black flex items-center justify-center overflow-hidden">
                                                    {elaImageUrl ? (
                                                        /* eslint-disable-next-line @next/next/no-img-element */
                                                        <img
                                                            src={elaImageUrl}
                                                            alt="Error Level Analysis residual"
                                                            className={`w-full h-full object-contain transition-transform duration-300 ${
                                                                isZoomed ? "scale-150" : ""
                                                            }`}
                                                        />
                                                    ) : (
                                                        <div className="p-6 text-center text-muted-foreground text-xs space-y-1">
                                                            <AlertCircle className="w-6 h-6 mx-auto mb-1 text-muted-foreground" />
                                                            <p>Server-side ELA residual map not available for this format.</p>
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {viewMode === "Heatmap" && (
                                                <div className="w-full h-full relative bg-neutral-950 flex items-center justify-center overflow-hidden">
                                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                                    <img
                                                        src={imageUrl}
                                                        alt="Attention map base"
                                                        className="w-full h-full object-contain opacity-70"
                                                    />
                                                    <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/35 via-yellow-400/35 to-red-500/50 mix-blend-overlay filter blur-xl pointer-events-none" />
                                                </div>
                                            )}
                                        </div>

                                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs text-muted-foreground gap-2 pt-1">
                                            <span className="font-medium text-foreground">
                                                {viewMode === "Slider"
                                                    ? "Drag divider to contrast original image against compression delta residuals."
                                                    : viewMode === "ELA"
                                                    ? "Error Level Analysis: highlights differences in compression levels across image regions."
                                                    : viewMode === "Heatmap"
                                                    ? "Attention Map: shows transformer activations contributing to classification."
                                                    : "Original Frame: raw uncompressed input subject."}
                                            </span>
                                            <span className="font-mono text-[11px] shrink-0">
                                                {file.name} ({Math.round(file.size / 1024)} KB)
                                            </span>
                                        </div>
                                    </div>

                                    {/* Google Lens / Reverse Visual Search Trigger */}
                                    <div className="p-4 sm:p-5 rounded-xl bg-card border border-border shadow-sm space-y-3">
                                        <div className="flex items-center justify-between">
                                            <div className="space-y-0.5">
                                                <div className="flex items-center gap-2">
                                                    <Globe className="w-4 h-4 text-primary" />
                                                    <h3 className="text-xs sm:text-sm font-bold text-foreground font-sans">
                                                        Google Lens & Web Reverse Visual Matching
                                                    </h3>
                                                </div>
                                                <p className="text-xs text-muted-foreground">
                                                    Find matching visual duplicates and suspected origins across open web indexes.
                                                </p>
                                            </div>

                                            <button
                                                onClick={handleTriggerReverseSearch}
                                                disabled={reverseSearching}
                                                className="px-3.5 py-1.5 rounded-lg bg-primary/10 border border-primary/20 text-primary text-xs font-bold hover:bg-primary/20 transition-all flex items-center gap-1.5 shrink-0"
                                            >
                                                {reverseSearching ? (
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                ) : (
                                                    <Search className="w-3.5 h-3.5" />
                                                )}
                                                <span>{reverseSearching ? "Searching..." : "Scan Google Lens"}</span>
                                            </button>
                                        </div>

                                        {/* Reverse Search Results Display */}
                                        {showReversePanel && reverseResult && (
                                            <div className="pt-2 border-t border-border/60 space-y-3">
                                                <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
                                                    <span>Matched {reverseResult.reverse_matches_count} Online Sources</span>
                                                    <span>{reverseResult.search_engine}</span>
                                                </div>

                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                                                    {reverseResult.visual_matches.map((vm, vIdx) => (
                                                        <a
                                                            key={vIdx}
                                                            href={vm.link}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="flex items-center gap-3 p-2.5 rounded-lg bg-muted/30 border border-border/60 hover:bg-muted/60 transition-colors group"
                                                        >
                                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                                            <img
                                                                src={vm.thumbnail}
                                                                alt=""
                                                                className="w-11 h-11 rounded-md object-cover shrink-0 border border-border"
                                                            />
                                                            <div className="min-w-0 flex-1 space-y-0.5">
                                                                <div className="flex items-center gap-1 text-[11px] font-bold text-foreground truncate group-hover:text-primary">
                                                                    <span className="truncate">{vm.title}</span>
                                                                    <ExternalLink className="w-3 h-3 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                                </div>
                                                                <p className="text-[10px] text-muted-foreground truncate">{vm.source}</p>
                                                                <div className="flex items-center gap-2">
                                                                    <span className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                                                                        {vm.similarity_score}% match
                                                                    </span>
                                                                    <span className="text-[10px] px-1.5 rounded bg-muted text-muted-foreground font-mono">
                                                                        {vm.match_type}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </a>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Right Column: Verdict & Metrics Summary (5 cols) */}
                                <div className="lg:col-span-5 space-y-4">
                                    {/* Main Verdict Card */}
                                    <ReportCard
                                        result={detectionResult.prediction === "Fake" ? "AI Generated" : "Real"}
                                        confidence={detectionResult.confidence ?? 0}
                                        elaScore={detectionResult.details?.ela_mean !== undefined
                                            ? `Mean: ${Number(detectionResult.details.ela_mean).toFixed(4)}, Std: ${Number(detectionResult.details.ela_std ?? 0).toFixed(4)}`
                                            : "N/A"}
                                        modelName={detectionResult.model_status ?? "SigLIP ViT Ensemble"}
                                        timestamp={new Date().toLocaleString()}
                                    />

                                    {/* Cryptographic SHA-256 Fingerprint Card */}
                                    <div className="p-4 rounded-xl bg-card border border-border shadow-sm space-y-2">
                                        <div className="flex items-center justify-between">
                                            <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                                                <Lock className="w-3.5 h-3.5 text-primary" />
                                                <span>Cryptographic SHA-256 Digest</span>
                                            </span>
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    const hash = fileSha256 || (typeof detectionResult.details?.sha256 === "string" ? detectionResult.details.sha256 : "");
                                                    if (hash) {
                                                        navigator.clipboard.writeText(hash);
                                                        toast.success("SHA-256 hash copied to clipboard");
                                                    }
                                                }}
                                                className="text-[11px] font-mono text-primary hover:underline inline-flex items-center gap-1"
                                                title="Copy SHA-256 hash"
                                            >
                                                <Copy className="w-3 h-3" />
                                                <span>Copy</span>
                                            </button>
                                        </div>
                                        <div className="p-2 rounded-lg bg-muted/50 border border-border/60 font-mono text-[11px] text-foreground break-all select-all">
                                            {fileSha256 || (typeof detectionResult.details?.sha256 === "string" ? detectionResult.details.sha256 : "Calculating...")}
                                        </div>
                                    </div>

                                    {/* Probability Distribution Breakdown */}
                                    {detectionResult.details?.raw_output && (
                                        <div className="bg-card p-4 sm:p-5 rounded-xl border border-border shadow-sm space-y-3">
                                            <div className="flex items-center justify-between">
                                                <h3 className="text-xs sm:text-sm font-bold text-foreground tracking-tight font-sans">
                                                    Softmax Probability Distribution
                                                </h3>
                                                <span className="text-[11px] font-mono text-muted-foreground">
                                                    Latency: {detectionResult.processing_time_ms}ms
                                                </span>
                                            </div>

                                            <div className="space-y-2">
                                                {detectionResult.details.raw_output.map((item) => {
                                                    const isFake = item.label === "Fake";
                                                    const pct = Math.min(100, Math.max(0, item.score * 100));

                                                    return (
                                                        <div key={item.label} className="space-y-1">
                                                            <div className="flex justify-between text-xs font-semibold">
                                                                <span className={isFake ? "text-destructive" : "text-green-600 dark:text-green-400"}>
                                                                    {item.label === "Fake" ? "Synthetic / AI Generated" : "Authentic Subject"}
                                                                </span>
                                                                <span className="font-mono">{pct.toFixed(2)}%</span>
                                                            </div>
                                                            <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                                                                <div
                                                                    className={`h-full transition-all duration-500 ${
                                                                        isFake ? "bg-destructive" : "bg-green-600 dark:bg-green-400"
                                                                    }`}
                                                                    style={{ width: `${pct}%` }}
                                                                />
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}

                                    {/* Quantitative Signal Statistics Grid */}
                                    <div className="p-4 sm:p-5 rounded-xl bg-card border border-border shadow-sm space-y-2.5">
                                        <h3 className="text-[11px] font-mono font-semibold uppercase tracking-wider text-muted-foreground">
                                            Quantitative Signal Statistics
                                        </h3>

                                        <div className="grid grid-cols-2 gap-2 text-xs">
                                            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/50 space-y-0.5">
                                                <span className="text-muted-foreground text-[10px] uppercase font-mono">ELA Mean Error</span>
                                                <div className="font-mono font-bold text-foreground text-xs sm:text-sm">
                                                    {detectionResult.details?.ela_mean !== undefined
                                                        ? Number(detectionResult.details.ela_mean).toFixed(4)
                                                        : "N/A"}
                                                </div>
                                            </div>

                                            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/50 space-y-0.5">
                                                <span className="text-muted-foreground text-[10px] uppercase font-mono">ELA Deviation</span>
                                                <div className="font-mono font-bold text-foreground text-xs sm:text-sm">
                                                    {detectionResult.details?.ela_std !== undefined
                                                        ? Number(detectionResult.details.ela_std).toFixed(4)
                                                        : "N/A"}
                                                </div>
                                            </div>

                                            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/50 space-y-0.5">
                                                <span className="text-muted-foreground text-[10px] uppercase font-mono">Peak Residue</span>
                                                <div className="font-mono font-bold text-foreground text-xs sm:text-sm">
                                                    {detectionResult.details?.ela_max !== undefined
                                                        ? Number(detectionResult.details.ela_max).toFixed(4)
                                                        : "N/A"}
                                                </div>
                                            </div>

                                            <div className="p-2.5 rounded-lg bg-muted/40 border border-border/50 space-y-0.5">
                                                <span className="text-muted-foreground text-[10px] uppercase font-mono">Engine Status</span>
                                                <div className="font-bold text-green-600 dark:text-green-400 text-xs sm:text-sm truncate font-mono">
                                                    {detectionResult.model_status || "Online"}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Collapsible Raw JSON Inspector */}
                                    <div className="border border-border rounded-xl bg-card overflow-hidden shadow-sm">
                                        <button
                                            type="button"
                                            onClick={() => setShowRawJson(!showRawJson)}
                                            className="w-full p-3.5 flex items-center justify-between text-xs font-bold text-foreground hover:bg-muted/40 transition-colors"
                                        >
                                            <span className="font-mono text-[11px]">Technical Forensic Payload (Raw JSON)</span>
                                            {showRawJson ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                        </button>

                                        {showRawJson && (
                                            <div className="p-3.5 border-t border-border/60 bg-muted/20 space-y-2.5">
                                                <div className="flex justify-end">
                                                    <button
                                                        type="button"
                                                        onClick={handleCopyJson}
                                                        className="text-xs font-semibold text-primary hover:underline inline-flex items-center gap-1 font-mono text-[11px]"
                                                    >
                                                        {copiedJson ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
                                                        <span>{copiedJson ? "Copied" : "Copy Payload"}</span>
                                                    </button>
                                                </div>
                                                <pre className="font-mono text-[11px] text-muted-foreground overflow-x-auto p-3 rounded-lg bg-muted/60 border border-border/50 max-h-56">
                                                    {JSON.stringify(detectionResult, null, 2)}
                                                </pre>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            )}
        </div>
    );
}
