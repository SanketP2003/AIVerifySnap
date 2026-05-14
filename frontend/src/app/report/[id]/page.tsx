"use client";

import { Download, FileSignature, ShieldCheck, QrCode, Loader2, ArrowLeft, Clock, Cpu, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import Image from "next/image";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";

interface AnalysisMetadata {
    model_status?: string;
    processing_time_ms?: number;
    elapsed_ms?: number;
    confidence?: number;
    verdict?: string;
    prediction?: string;
    is_deepfake?: boolean;
    riskLevel?: string;
    details?: {
        ela_mean?: number;
        ela_std?: number;
        ela_max?: number;
        ela_image_base64?: string;
        ela_heatmap_base64?: string;
        raw_output?: Array<{ label: string; score: number }>;
    };
}

function mapResult(label: string): "Real" | "AI Generated" | "Suspicious" {
    if (label.toLowerCase() === "fake") return "AI Generated";
    if (label.toLowerCase() === "real") return "Real";
    return "Suspicious";
}

function parseMetadata(raw: string | null | undefined): AnalysisMetadata | null {
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

function formatDate(ts: string | null | undefined): string {
    if (!ts) return "—";
    try {
        const d = new Date(ts);
        return d.toLocaleString("en-IN", {
            dateStyle: "medium",
            timeStyle: "short",
        });
    } catch {
        return ts;
    }
}

function generateHash(scanId: number | string): string {
    const seed = String(scanId);
    const chars = "0123456789abcdef";
    let hash = "";
    for (let i = 0; i < 40; i++) {
        hash += chars[(seed.charCodeAt(i % seed.length) * (i + 7)) % chars.length];
    }
    return hash;
}

export default function ReportPage() {
    const { id } = useParams();
    const router = useRouter();
    const [report, setReport] = useState<DetectionHistoryItem | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchReport() {
            try {
                setLoading(true);
                const data = await detectionApi.getReport(id as string);
                setReport(data);
            } catch (err) {
                console.error("Failed to fetch report:", err);
                setError("Failed to load report. The scan may not exist or the backend is unavailable.");
            } finally {
                setLoading(false);
            }
        }
        if (id) fetchReport();
    }, [id]);

    if (loading) {
        return (
            <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center gap-3">
                <Loader2 className="w-7 h-7 text-accent animate-spin" />
                <p className="text-sm text-muted-foreground">Loading forensic report...</p>
            </div>
        );
    }

    if (error || !report) {
        return (
            <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center gap-4 px-6">
                <div className="p-6 rounded-2xl bg-destructive/5 border border-destructive/15 text-center max-w-md">
                    <XCircle className="w-10 h-10 text-destructive mx-auto mb-3" />
                    <p className="text-[15px] font-medium text-destructive">{error || "Report not found."}</p>
                </div>
                <button onClick={() => router.push("/history")}
                    className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                    <ArrowLeft className="w-4 h-4" /> Back to History
                </button>
            </div>
        );
    }

    const metadata = parseMetadata(report.analysisMetadata);
    const displayResult = mapResult(report.resultLabel);
    const confidence = report.confidenceScore ?? metadata?.confidence ?? 0;
    const modelName = metadata?.model_status || "Deepfake Detector";
    const processingTime = metadata?.processing_time_ms;
    const elapsedTime = metadata?.elapsed_ms;
    const riskLevel = metadata?.riskLevel || (confidence > 75 ? "HIGH" : confidence > 40 ? "MEDIUM" : "LOW");
    const elaImageBase64 = metadata?.details?.ela_image_base64;
    const elaHeatmapBase64 = metadata?.details?.ela_heatmap_base64;
    const elaMean = metadata?.details?.ela_mean;
    const elaStd = metadata?.details?.ela_std;
    const elaMax = metadata?.details?.ela_max;
    const rawOutput = metadata?.details?.raw_output;
    const authenticityHash = generateHash(report.scanId);

    const riskColor = riskLevel === "HIGH" ? "text-destructive" : riskLevel === "MEDIUM" ? "text-orange-500" : "text-green-500";
    const RiskIcon = riskLevel === "HIGH" ? AlertTriangle : riskLevel === "MEDIUM" ? AlertTriangle : CheckCircle2;

    return (
        <div className="min-h-[calc(100vh-4rem)] p-4 md:p-8 flex flex-col items-center bg-muted/20">
            <div className="max-w-3xl w-full space-y-5">
                {/* Top Bar */}
                <div className="flex items-center justify-between">
                    <button onClick={() => router.push("/history")}
                        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                        <ArrowLeft className="w-4 h-4" /> Back to History
                    </button>
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-foreground text-background font-semibold rounded-xl hover:opacity-90 transition-all shadow-sm text-[13px]">
                        <Download className="w-4 h-4" /> Download PDF
                    </button>
                </div>

                {/* Certificate Card */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="bg-card border shadow-xl p-8 md:p-12 rounded-[1.5rem] space-y-8 relative overflow-hidden"
                >
                    {/* Watermark */}
                    <ShieldCheck className="absolute -bottom-16 -right-16 w-80 h-80 text-primary/[0.03] z-0 pointer-events-none" />

                    {/* Header */}
                    <div className="flex flex-col md:flex-row justify-between items-start border-b pb-7 relative z-10 gap-5" style={{ borderColor: "hsl(var(--border) / 0.5)" }}>
                        <div className="flex items-center gap-3">
                            <ShieldCheck className="w-9 h-9 text-accent" />
                            <div>
                                <h2 className="text-xl font-black uppercase tracking-widest text-foreground">AIVerifySnap</h2>
                                <p className="text-[11px] text-muted-foreground uppercase tracking-[0.2em] font-medium">Forensic Verification Certificate</p>
                            </div>
                        </div>
                        <div className="text-right flex flex-col items-end">
                            <span className="text-[11px] text-muted-foreground uppercase tracking-[0.15em] font-semibold">Report ID</span>
                            <span className="font-mono text-lg font-bold text-foreground bg-muted px-3 py-1 rounded-lg mt-1">#{report.scanId}</span>
                        </div>
                    </div>

                    {/* Verdict Section */}
                    <div className="grid md:grid-cols-2 gap-8 relative z-10">
                        {/* Left: File Info & ELA */}
                        <div className="space-y-5">
                            <div className="p-5 rounded-xl bg-muted/30 border space-y-3" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">Analyzed Media</h3>
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">Filename</span>
                                        <span className="font-medium truncate max-w-[180px]">{report.imagePath || "—"}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-muted-foreground">Scan Time</span>
                                        <span className="font-medium">{formatDate(report.scanTimestamp)}</span>
                                    </div>
                                    {report.user && (
                                        <div className="flex justify-between text-sm">
                                            <span className="text-muted-foreground">Scanned by</span>
                                            <span className="font-medium">{report.user.name}</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* ELA Visualization */}
                            {elaImageBase64 && (
                                <div className="space-y-2">
                                    <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">Error Level Analysis (ELA)</h3>
                                    <div className="aspect-video relative rounded-xl overflow-hidden border bg-muted/20 shadow-inner" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                        <Image
                                            src={`data:image/png;base64,${elaImageBase64}`}
                                            alt="ELA visualization"
                                            fill
                                            className="object-contain"
                                            unoptimized
                                        />
                                    </div>
                                </div>
                            )}

                            {elaHeatmapBase64 && (
                                <div className="space-y-2">
                                    <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">ELA Heatmap</h3>
                                    <div className="aspect-video relative rounded-xl overflow-hidden border bg-muted/20 shadow-inner" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                        <Image
                                            src={`data:image/png;base64,${elaHeatmapBase64}`}
                                            alt="ELA heatmap"
                                            fill
                                            className="object-contain"
                                            unoptimized
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Right: Verdict & Details */}
                        <div className="space-y-5 flex flex-col">
                            {/* Verdict Card */}
                            <div className="p-6 bg-muted/30 border rounded-xl text-center space-y-3" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">Final Verdict</h3>
                                <div className="flex justify-center">
                                    <ResultBadge status={displayResult} />
                                </div>
                                <div className="text-4xl font-black font-mono mt-2">{confidence.toFixed(1)}%</div>
                                <p className="text-muted-foreground text-[13px]">Confidence Score</p>
                            </div>

                            {/* Risk Level */}
                            <div className={`flex items-center gap-3 p-4 rounded-xl border ${
                                riskLevel === "HIGH" ? "bg-destructive/5 border-destructive/20" :
                                riskLevel === "MEDIUM" ? "bg-orange-500/5 border-orange-500/20" :
                                "bg-green-500/5 border-green-500/20"
                            }`}>
                                <RiskIcon className={`w-5 h-5 ${riskColor}`} />
                                <div>
                                    <div className={`text-[13px] font-semibold ${riskColor}`}>Risk Level: {riskLevel}</div>
                                    <p className="text-[12px] text-muted-foreground">
                                        {riskLevel === "HIGH" ? "High probability of manipulation detected" :
                                         riskLevel === "MEDIUM" ? "Some indicators of possible manipulation" :
                                         "Low probability of manipulation"}
                                    </p>
                                </div>
                            </div>

                            {/* Technical Details */}
                            <div className="p-5 rounded-xl bg-muted/30 border space-y-3 flex-1" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">Technical Details</h3>
                                <div className="space-y-2.5">
                                    <div className="flex justify-between text-sm border-b pb-2" style={{ borderColor: "hsl(var(--border) / 0.3)" }}>
                                        <span className="text-muted-foreground flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5" /> Model</span>
                                        <span className="font-medium text-[13px] text-right max-w-[180px]">{modelName}</span>
                                    </div>
                                    {processingTime !== undefined && (
                                        <div className="flex justify-between text-sm border-b pb-2" style={{ borderColor: "hsl(var(--border) / 0.3)" }}>
                                            <span className="text-muted-foreground flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> Processing</span>
                                            <span className="font-medium font-mono text-[13px]">{processingTime}ms</span>
                                        </div>
                                    )}
                                    {elapsedTime !== undefined && (
                                        <div className="flex justify-between text-sm border-b pb-2" style={{ borderColor: "hsl(var(--border) / 0.3)" }}>
                                            <span className="text-muted-foreground flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> Total Elapsed</span>
                                            <span className="font-medium font-mono text-[13px]">{elapsedTime}ms</span>
                                        </div>
                                    )}
                                    {elaMean !== undefined && (
                                        <div className="flex justify-between text-sm border-b pb-2" style={{ borderColor: "hsl(var(--border) / 0.3)" }}>
                                            <span className="text-muted-foreground">ELA Mean</span>
                                            <span className="font-medium font-mono text-[13px]">{elaMean.toFixed(4)}</span>
                                        </div>
                                    )}
                                    {elaStd !== undefined && (
                                        <div className="flex justify-between text-sm border-b pb-2" style={{ borderColor: "hsl(var(--border) / 0.3)" }}>
                                            <span className="text-muted-foreground">ELA Std Dev</span>
                                            <span className="font-medium font-mono text-[13px]">{elaStd.toFixed(4)}</span>
                                        </div>
                                    )}
                                    {elaMax !== undefined && (
                                        <div className="flex justify-between text-sm">
                                            <span className="text-muted-foreground">ELA Max</span>
                                            <span className="font-medium font-mono text-[13px]">{elaMax.toFixed(4)}</span>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Raw Scores */}
                            {rawOutput && rawOutput.length > 0 && (
                                <div className="p-5 rounded-xl bg-muted/30 border space-y-3" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                    <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.15em]">Model Output Scores</h3>
                                    <div className="space-y-2">
                                        {rawOutput.map((item, i) => (
                                            <div key={i} className="space-y-1">
                                                <div className="flex justify-between text-[13px]">
                                                    <span className="font-medium capitalize">{item.label}</span>
                                                    <span className="font-mono text-muted-foreground">{(item.score * 100).toFixed(2)}%</span>
                                                </div>
                                                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all ${
                                                            item.label.toLowerCase() === "fake" ? "bg-destructive" : "bg-green-500"
                                                        }`}
                                                        style={{ width: `${item.score * 100}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="flex items-end justify-between pt-7 border-t relative z-10 gap-4" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                        <div className="flex items-center gap-3">
                            <QrCode className="w-14 h-14 p-2 bg-background border rounded-lg shadow-sm" />
                            <div className="text-[11px] text-muted-foreground max-w-[200px] space-y-1">
                                <p>Scan to verify this certificate online.</p>
                                <p className="font-mono text-[10px] text-foreground/50 break-all leading-tight">{authenticityHash}</p>
                            </div>
                        </div>

                        <div className="text-right">
                            <FileSignature className="w-10 h-10 text-muted-foreground/30 inline-block mb-1.5" />
                            <div className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-semibold border-t-2 pt-1">
                                Authorized Signature
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
