"use client";

import {
    Download,
    FileSignature,
    ShieldCheck,
    QrCode,
    Copy,
    Check,
    ArrowLeft,
    Printer,
    Cpu,
    Calendar,
    Hash,
    Loader2,
    AlertCircle
} from "lucide-react";
import Link from "next/link";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";
import { generatePdfReport } from "@/lib/pdfReportGenerator";

export default function ReportPage() {
    const params = useParams();
    const id = (params?.id as string) || "";
    const [copied, setCopied] = useState(false);
    const [loading, setLoading] = useState(true);
    const [notFound, setNotFound] = useState(false);
    const [realItem, setRealItem] = useState<DetectionHistoryItem | null>(null);

    useEffect(() => {
        if (!id) {
            setLoading(false);
            setNotFound(true);
            return;
        }

        setLoading(true);
        detectionApi.getReport(id)
            .then((data) => {
                if (data && data.scanId) {
                    setRealItem(data);
                    setNotFound(false);
                } else {
                    setNotFound(true);
                }
            })
            .catch((err) => {
                console.error("Report lookup failed:", err);
                setNotFound(true);
            })
            .finally(() => {
                setLoading(false);
            });
    }, [id]);

    const displayResult = realItem
        ? (realItem.resultLabel?.toLowerCase() === "fake" ? "AI Generated" : realItem.resultLabel?.toLowerCase() === "real" ? "Real" : "Suspicious")
        : "Analysis Completed";

    // Try to parse metadata if available
    let parsedMeta: Record<string, unknown> = {};
    if (realItem?.analysisMetadata) {
        try {
            parsedMeta = JSON.parse(realItem.analysisMetadata);
        } catch {
            parsedMeta = {};
        }
    }

    const hasRealHash = typeof parsedMeta.sha256 === "string" && parsedMeta.sha256.trim() !== "";
    const shaHash = hasRealHash ? (parsedMeta.sha256 as string) : "Unavailable from analysis service";

    const reportData = {
        id: realItem ? `AVS-${realItem.scanId}` : `AVS-${id}`,
        result: displayResult as "Real" | "AI Generated" | "Suspicious",
        confidence: realItem?.confidenceScore != null ? realItem.confidenceScore : null,
        timestamp: realItem?.scanTimestamp ? new Date(realItem.scanTimestamp).toLocaleString() : new Date().toLocaleString(),
        model: "SigLIP ViT + Error Level Analysis (ELA)",
        imageThumb: realItem?.imagePath || null,
        authenticityHash: shaHash,
        examiner: "AIVerifySnap Media Analysis Engine",
        fileName: realItem?.imagePath ? realItem.imagePath.replace(/^.*[\\/]/, '') : `scan_${id}.png`,
    };

    const handleCopyHash = () => {
        if (!hasRealHash) {
            toast.error("SHA-256 hash not available for this record");
            return;
        }
        navigator.clipboard.writeText(reportData.authenticityHash);
        setCopied(true);
        toast.success("Authenticity SHA-256 hash copied to clipboard");
        setTimeout(() => setCopied(false), 2000);
    };

    const handlePrint = () => {
        window.print();
    };

    const handleDownloadPdf = () => {
        generatePdfReport({
            reportId: reportData.id,
            fileName: reportData.fileName,
            result: reportData.result,
            confidence: reportData.confidence != null ? reportData.confidence : 0,
            timestamp: reportData.timestamp,
            sha256Hash: reportData.authenticityHash,
            model: reportData.model,
        });
        toast.success("Forensic PDF report downloaded");
    };

    if (loading) {
        return (
            <div className="min-h-[calc(100vh-4rem)] p-8 flex flex-col items-center justify-center space-y-4">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
                <p className="text-xs sm:text-sm font-semibold text-muted-foreground">
                    Retrieving analysis report #{id}...
                </p>
            </div>
        );
    }

    if (notFound && !realItem) {
        return (
            <div className="min-h-[calc(100vh-4rem)] p-8 flex flex-col items-center justify-center">
                <div className="max-w-md w-full bg-card border border-border p-8 rounded-3xl text-center space-y-4 shadow-sm">
                    <div className="w-14 h-14 rounded-full bg-destructive/10 text-destructive flex items-center justify-center mx-auto">
                        <AlertCircle className="w-7 h-7" />
                    </div>
                    <h2 className="text-xl font-bold text-foreground">Analysis Record Not Found</h2>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                        We could not retrieve an analysis record matching identifier <code className="font-mono bg-muted px-1.5 py-0.5 rounded text-foreground font-bold">#{id}</code> in the database.
                    </p>
                    <div className="pt-2 flex items-center justify-center gap-3">
                        <Link
                            href="/history"
                            className="px-5 py-2.5 rounded-xl bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors"
                        >
                            Back to History
                        </Link>
                        <Link
                            href="/detect"
                            className="px-5 py-2.5 rounded-xl bg-foreground text-background text-xs font-semibold hover:bg-foreground/90 transition-colors shadow-sm"
                        >
                            Perform New Scan
                        </Link>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 sm:p-6 md:p-8 flex flex-col items-center justify-center bg-muted/10 print:bg-white print:p-0">
            <div className="max-w-3xl w-full space-y-6">
                {/* Navigation and Top Actions */}
                <div className="flex items-center justify-between print:hidden">
                    <Link
                        href="/reports"
                        className="inline-flex items-center gap-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <ArrowLeft className="w-3.5 h-3.5" /> Back to Reports
                    </Link>

                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={handlePrint}
                            className="flex items-center gap-1.5 px-4 py-2 bg-card border border-border text-foreground text-xs font-semibold rounded-xl hover:bg-muted transition-all shadow-sm focus-ring"
                        >
                            <Printer className="w-3.5 h-3.5" /> Print Report
                        </button>
                        <button
                            type="button"
                            onClick={handleDownloadPdf}
                            className="flex items-center gap-1.5 px-4 py-2 bg-foreground text-background text-xs font-semibold rounded-xl hover:bg-foreground/90 transition-all shadow-sm focus-ring"
                        >
                            <Download className="w-3.5 h-3.5" /> Download PDF
                        </button>
                    </div>
                </div>

                {/* Report Card */}
                <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-card border-2 border-border shadow-xl p-6 sm:p-10 md:p-12 rounded-[2.5rem] space-y-8 relative overflow-hidden print:border-none print:shadow-none print:p-4 print:rounded-none"
                >
                    {/* Watermark */}
                    <ShieldCheck className="absolute -bottom-16 -right-16 w-80 h-80 text-foreground/[0.03] z-0 pointer-events-none select-none" />

                    {/* Report Header */}
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-border pb-6 relative z-10 gap-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 bg-primary/10 rounded-2xl text-primary">
                                <ShieldCheck className="w-8 h-8" />
                            </div>
                            <div>
                                <h2 className="text-xl font-black uppercase tracking-widest text-foreground font-sans">
                                    AIVerifySnap
                                </h2>
                                <p className="text-[11px] text-muted-foreground uppercase tracking-widest font-semibold">
                                    Automated Image Analysis Report
                                </p>
                            </div>
                        </div>

                        <div className="sm:text-right flex flex-col sm:items-end">
                            <span className="text-[11px] text-muted-foreground uppercase tracking-wider font-semibold">
                                Report ID
                            </span>
                            <span className="font-mono text-base font-bold text-foreground bg-muted/80 px-3 py-1 rounded-lg mt-0.5 border border-border/40">
                                {reportData.id}
                            </span>
                        </div>
                    </div>

                    {/* Content Section */}
                    <div className="grid sm:grid-cols-2 gap-8 relative z-10 items-center">
                        <div className="space-y-4">
                            <div>
                                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2.5">
                                    Analyzed Media Frame
                                </h3>
                                <div className="aspect-square relative rounded-2xl overflow-hidden border border-border bg-muted/30 shadow-inner flex items-center justify-center">
                                    {reportData.imageThumb ? (
                                        // eslint-disable-next-line @next/next/no-img-element
                                        <img
                                            src={reportData.imageThumb}
                                            alt="Subject media"
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="flex flex-col items-center justify-center text-muted-foreground p-4 text-center space-y-1">
                                            <Hash className="w-8 h-8 opacity-40" />
                                            <span className="text-xs">No media preview available</span>
                                        </div>
                                    )}
                                </div>
                                <p className="text-[11px] text-muted-foreground font-mono mt-2 truncate" title={reportData.fileName}>
                                    File: {reportData.fileName}
                                </p>
                            </div>
                        </div>

                        <div className="space-y-6 flex flex-col justify-center">
                            {/* Verdict Box */}
                            <div className="p-6 bg-muted/40 border border-border rounded-2xl text-center shadow-sm space-y-2">
                                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                                    Analysis Verdict
                                </h3>
                                <div className="flex justify-center pt-1">
                                    <ResultBadge status={reportData.result} size="lg" />
                                </div>
                                <div className="text-4xl font-black tracking-tight font-mono pt-2 text-foreground">
                                    {reportData.confidence != null ? `${Number(reportData.confidence).toFixed(1)}%` : "N/A"}
                                </div>
                                <p className="text-xs text-muted-foreground">
                                    {reportData.confidence != null ? "Statistical Neural Confidence" : "Confidence score not supplied by model"}
                                </p>
                            </div>

                            {/* Details List */}
                            <div className="space-y-3 text-xs">
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-muted-foreground flex items-center gap-1.5">
                                        <Cpu className="w-3.5 h-3.5 text-primary" /> Engine
                                    </span>
                                    <span className="font-semibold text-foreground">{reportData.model}</span>
                                </div>
                                <div className="flex justify-between items-center border-b border-border pb-2">
                                    <span className="text-muted-foreground flex items-center gap-1.5">
                                        <Calendar className="w-3.5 h-3.5 text-primary" /> Timestamp
                                    </span>
                                    <span className="font-semibold text-foreground font-mono">{reportData.timestamp}</span>
                                </div>
                                <div className="flex flex-col gap-1 border-b border-border pb-2">
                                    <div className="flex items-center justify-between">
                                        <span className="text-muted-foreground flex items-center gap-1.5">
                                            <Hash className="w-3.5 h-3.5 text-primary" /> Image SHA-256 Digest
                                        </span>
                                        {hasRealHash && (
                                            <button
                                                type="button"
                                                onClick={handleCopyHash}
                                                aria-label="Copy hash"
                                                className="text-[11px] font-semibold text-primary hover:underline flex items-center gap-1"
                                            >
                                                {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
                                                {copied ? "Copied" : "Copy"}
                                            </button>
                                        )}
                                    </div>
                                    <span className="font-mono text-[10px] text-muted-foreground break-all bg-muted/50 p-2 rounded-lg border border-border/40 select-all">
                                        {reportData.authenticityHash}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Footer of the Report */}
                    <div className="flex flex-col sm:flex-row items-start sm:items-end justify-between pt-6 border-t border-border relative z-10 gap-6">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-background border border-border rounded-xl shadow-sm shrink-0">
                                <QrCode className="w-12 h-12 text-foreground" />
                            </div>
                            <div className="text-[11px] text-muted-foreground max-w-[240px] leading-relaxed">
                                Automated media analysis report generated by AIVerifySnap.
                            </div>
                        </div>

                        <div className="text-left sm:text-right shrink-0">
                            <FileSignature className="w-10 h-10 text-muted-foreground/30 inline-block mb-1" />
                            <div className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold border-t-2 border-border pt-1">
                                {reportData.examiner}
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
