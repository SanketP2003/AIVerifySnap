"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
    FileText,
    Search,
    Filter,
    Download,
    ArrowRight,
    Shield,
    RefreshCw,
    Copy,
    Check
} from "lucide-react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";
import { ResultBadge } from "@/components/shared/ResultBadge";
import { generatePdfReport } from "@/lib/pdfReportGenerator";

function getSha256(item: DetectionHistoryItem): string {
    if (item.analysisMetadata) {
        try {
            const meta = JSON.parse(item.analysisMetadata);
            if (typeof meta.sha256 === "string" && meta.sha256.trim() !== "") {
                return meta.sha256;
            }
        } catch {
            // ignore malformed JSON
        }
    }
    return "";
}

export default function ReportsCatalogPage() {
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState<"all" | "fake" | "real">("all");
    const [copiedHash, setCopiedHash] = useState<string | null>(null);

    const loadData = () => {
        setLoading(true);
        setError(null);
        detectionApi.getHistory()
            .then((data) => {
                setHistory(data || []);
            })
            .catch((err) => {
                console.error("Reports loading error:", err);
                setError(err instanceof Error ? err.message : "Failed to load reports from analysis service.");
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        loadData();
    }, []);

    const copyToClipboard = (text: string) => {
        if (!text) return;
        navigator.clipboard.writeText(text);
        setCopiedHash(text);
        setTimeout(() => setCopiedHash(null), 2000);
    };

    const downloadDossier = (item: DetectionHistoryItem) => {
        const hash = getSha256(item) || "Unavailable from analysis service";
        const dossier = {
            title: "AIVerifySnap Forensic Analysis Report",
            scanId: item.scanId,
            timestamp: item.scanTimestamp,
            imagePath: item.imagePath,
            verdict: item.resultLabel,
            confidenceScore: item.confidenceScore != null ? item.confidenceScore : "Not provided",
            sha256Hash: hash,
            metadata: item.analysisMetadata,
            issuer: "AIVerifySnap Media Analysis Engine",
            disclaimer: "Statistical neural detection and Error Level Analysis decision support."
        };
        const blob = new Blob([JSON.stringify(dossier, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `analysis-dossier-scan-${item.scanId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const downloadPdfDossier = (item: DetectionHistoryItem) => {
        const hash = getSha256(item);
        const fileName = item.imagePath ? item.imagePath.replace(/^.*[\\/]/, '') : `scan_${item.scanId}.png`;
        generatePdfReport({
            reportId: `AVS-${item.scanId}`,
            fileName: fileName,
            result: item.resultLabel || "Analysis Completed",
            confidence: item.confidenceScore != null ? item.confidenceScore : 0,
            timestamp: item.scanTimestamp ? new Date(item.scanTimestamp).toLocaleString() : new Date().toLocaleString(),
            sha256Hash: hash || "Unavailable from analysis service",
            model: "SigLIP ViT + Error Level Analysis (ELA)",
        });
    };

    const filteredReports = useMemo(() => {
        return history.filter((item) => {
            const hash = getSha256(item);
            const matchesSearch =
                (item.imagePath && item.imagePath.toLowerCase().includes(searchQuery.toLowerCase())) ||
                String(item.scanId).includes(searchQuery) ||
                (hash && hash.toLowerCase().includes(searchQuery.toLowerCase()));

            const matchesStatus =
                statusFilter === "all" ||
                item.resultLabel?.toLowerCase() === statusFilter;

            return matchesSearch && matchesStatus;
        });
    }, [history, searchQuery, statusFilter]);

    const totalScans = history.length;
    const fakeCount = history.filter((h) => h.resultLabel?.toLowerCase() === "fake").length;
    const realCount = history.filter((h) => h.resultLabel?.toLowerCase() === "real").length;
    const hashedScansCount = history.filter((h) => !!getSha256(h)).length;

    return (
        <div className="p-4 sm:p-6 md:p-8 max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-border/80 gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="w-2 h-2 rounded-full bg-primary" />
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
                            Dossiers &middot; Forensic Records
                        </span>
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-foreground font-sans">
                        Forensic Reports & Dossiers
                    </h1>
                    <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
                        Inspect, verify, and export automated forensic analysis reports and technical dossiers.
                    </p>
                </div>

                <div className="flex items-center gap-2.5 w-full sm:w-auto">
                    <button
                        type="button"
                        onClick={loadData}
                        disabled={loading}
                        className="px-3.5 py-2 rounded-xl bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors flex items-center gap-1.5 shadow-sm"
                        title="Sync latest reports"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                        <span>Sync</span>
                    </button>
                    <Link
                        href="/detect"
                        className="flex-1 sm:flex-none px-5 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-md flex items-center justify-center gap-2"
                    >
                        <span>Run New Scan</span>
                        <ArrowRight className="w-4 h-4" />
                    </Link>
                </div>
            </div>

            {/* Metrics Row */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                        Total Reports
                    </span>
                    <div className="text-3xl font-black font-mono text-foreground">{totalScans}</div>
                    <p className="text-[11px] text-muted-foreground">Historical records</p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                        Synthetic Findings
                    </span>
                    <div className="text-3xl font-black font-mono text-destructive">{fakeCount}</div>
                    <p className="text-[11px] text-destructive font-medium">Manipulated frames</p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                        Authentic Findings
                    </span>
                    <div className="text-3xl font-black font-mono text-green-600">{realCount}</div>
                    <p className="text-[11px] text-green-600 font-medium">Camera source frames</p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-1">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                        Cryptographic Hashes
                    </span>
                    <div className="text-3xl font-black font-mono text-foreground flex items-center gap-2">
                        <span>{hashedScansCount}/{totalScans}</span>
                        <Shield className="w-5 h-5 text-primary" />
                    </div>
                    <p className="text-[11px] text-muted-foreground">SHA-256 verified</p>
                </div>
            </div>

            {/* Filter and Search Bar */}
            <div className="p-4 rounded-2xl bg-card border border-border shadow-sm flex flex-col sm:flex-row items-center justify-between gap-3">
                <div className="relative w-full sm:w-80">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search by ID, filename, or hash..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-4 py-2 bg-muted/40 border border-border/80 rounded-xl text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-foreground"
                    />
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                    <span className="text-xs text-muted-foreground font-semibold flex items-center gap-1 hidden md:flex">
                        <Filter className="w-3.5 h-3.5" />
                        <span>Filter:</span>
                    </span>
                    <div className="flex rounded-xl bg-muted/60 p-1 border border-border/60 text-xs font-semibold">
                        <button
                            type="button"
                            onClick={() => setStatusFilter("all")}
                            className={`px-3 py-1 rounded-lg transition-colors ${statusFilter === "all" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                        >
                            All ({totalScans})
                        </button>
                        <button
                            type="button"
                            onClick={() => setStatusFilter("fake")}
                            className={`px-3 py-1 rounded-lg transition-colors ${statusFilter === "fake" ? "bg-background text-destructive shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                        >
                            Synthetic ({fakeCount})
                        </button>
                        <button
                            type="button"
                            onClick={() => setStatusFilter("real")}
                            className={`px-3 py-1 rounded-lg transition-colors ${statusFilter === "real" ? "bg-background text-green-600 shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
                        >
                            Authentic ({realCount})
                        </button>
                    </div>
                </div>
            </div>

            {/* Error State */}
            {error && !loading && (
                <div className="p-12 rounded-3xl bg-destructive/10 border border-destructive/20 text-center space-y-4 shadow-sm">
                    <div className="w-16 h-16 rounded-full bg-destructive/20 text-destructive flex items-center justify-center mx-auto">
                        <FileText className="w-8 h-8" />
                    </div>
                    <div className="space-y-1">
                        <h3 className="text-base font-bold text-foreground">Failed to Load Reports</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground max-w-md mx-auto">
                            {error}
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={loadData}
                        className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-sm"
                    >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>Retry</span>
                    </button>
                </div>
            )}

            {/* Reports List */}
            {!error && filteredReports.length > 0 ? (
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
                    {filteredReports.slice(0, 30).map((item) => {
                        const isFake = item.resultLabel?.toLowerCase() === "fake";
                        const status = isFake ? "AI Generated" : "Real";
                        const fileName = item.imagePath ? item.imagePath.replace(/^.*[\\/]/, '') : "unnamed_frame.png";
                        const hash = getSha256(item);
                        const shortHash = hash ? `${hash.slice(0, 10)}...${hash.slice(-8)}` : "Unavailable";

                        return (
                            <div
                                key={item.scanId}
                                className="p-6 rounded-3xl bg-card border border-border shadow-sm flex flex-col justify-between space-y-4 hover:border-foreground/30 hover:shadow-md transition-all group"
                            >
                                <div className="space-y-3">
                                    {/* Top Bar of Card */}
                                    <div className="flex items-center justify-between pb-3 border-b border-border/60">
                                        <span className="font-mono text-xs font-bold text-muted-foreground">
                                            REPORT #{item.scanId}
                                        </span>
                                        <ResultBadge status={status} size="sm" />
                                    </div>

                                    {/* Subject Title */}
                                    <div>
                                        <h3 className="font-bold text-sm text-foreground truncate group-hover:text-primary transition-colors" title={fileName}>
                                            {fileName}
                                        </h3>
                                        <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
                                            {item.scanTimestamp ? new Date(item.scanTimestamp).toLocaleString() : "Recent Scan"}
                                        </p>
                                    </div>

                                    {/* Metrics pill */}
                                    <div className="p-3 rounded-2xl bg-muted/30 border border-border/40 space-y-2 text-xs">
                                        <div className="flex justify-between items-center">
                                            <span className="text-muted-foreground">Model Confidence</span>
                                            <span className="font-mono font-bold text-foreground">
                                                {item.confidenceScore != null ? `${item.confidenceScore.toFixed(1)}%` : "Not provided"}
                                            </span>
                                        </div>
                                        <div className="flex justify-between items-center text-[11px]">
                                            <span className="text-muted-foreground">SHA-256 Digest</span>
                                            {hash ? (
                                                <button
                                                    type="button"
                                                    onClick={() => copyToClipboard(hash)}
                                                    className="font-mono text-muted-foreground hover:text-foreground flex items-center gap-1"
                                                    title="Click to copy full SHA-256 hash"
                                                >
                                                    <span>{shortHash}</span>
                                                    {copiedHash === hash ? (
                                                        <Check className="w-3 h-3 text-green-500" />
                                                    ) : (
                                                        <Copy className="w-3 h-3 opacity-60" />
                                                    )}
                                                </button>
                                            ) : (
                                                <span className="font-mono text-muted-foreground italic">Not provided</span>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Actions */}
                                <div className="pt-2 border-t border-border/60 flex items-center gap-2">
                                    <Link
                                        href={`/report/${item.scanId}`}
                                        className="flex-1 py-2 px-3 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all text-center flex items-center justify-center gap-1.5 shadow-sm"
                                    >
                                        <span>View Report</span>
                                        <ArrowRight className="w-3.5 h-3.5" />
                                    </Link>
                                    <button
                                        type="button"
                                        onClick={() => downloadPdfDossier(item)}
                                        className="p-2 rounded-xl bg-primary/10 hover:bg-primary/20 text-primary text-xs font-semibold transition-colors flex items-center justify-center"
                                        title="Download Forensic PDF Report"
                                    >
                                        <FileText className="w-4 h-4" />
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => downloadDossier(item)}
                                        className="p-2 rounded-xl bg-muted hover:bg-muted/80 text-foreground text-xs font-semibold transition-colors flex items-center justify-center"
                                        title="Download JSON Technical Dossier"
                                    >
                                        <Download className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : !error && (
                <div className="p-12 rounded-3xl bg-card border border-border shadow-sm text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto text-muted-foreground">
                        <FileText className="w-8 h-8" />
                    </div>
                    <div className="space-y-1">
                        <h3 className="text-base font-bold text-foreground">
                            {searchQuery ? "No Matching Reports Found" : "No Reports Recorded"}
                        </h3>
                        <p className="text-xs sm:text-sm text-muted-foreground max-w-md mx-auto">
                            {searchQuery
                                ? "Try broadening your search query or reset status filters."
                                : "Upload an image in the analysis workspace to generate your first technical report."}
                        </p>
                    </div>
                    <Link
                        href="/detect"
                        className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-sm"
                    >
                        <span>Run First Verification</span>
                    </Link>
                </div>
            )}
        </div>
    );
}
