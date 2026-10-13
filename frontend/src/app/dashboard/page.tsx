"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
    PlusCircle,
    ScanSearch,
    History,
    Shield,
    Activity,
    ArrowRight,
    Sparkles,
    CheckCircle2,
    AlertTriangle,
    FileText,
    HardDrive,
    RefreshCw
} from "lucide-react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";
import { ResultBadge } from "@/components/shared/ResultBadge";

export default function DashboardOverviewPage() {
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadData = () => {
        setLoading(true);
        setError(null);
        detectionApi.getHistory()
            .then((data) => setHistory(data || []))
            .catch((err) => {
                console.error("Dashboard history error:", err);
                setError(err instanceof Error ? err.message : "Failed to load scan records from service.");
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        loadData();
    }, []);

    const totalScans = history.length;
    const fakeCount = history.filter((h) => h.resultLabel?.toLowerCase() === "fake").length;
    const realCount = history.filter((h) => h.resultLabel?.toLowerCase() === "real").length;
    const avgConfidence = totalScans > 0
        ? Math.round(history.reduce((acc, h) => acc + (h.confidenceScore || 0), 0) / totalScans)
        : 0;

    return (
        <div className="p-4 sm:p-6 md:p-8 max-w-7xl mx-auto space-y-8">
            {/* Workspace Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-border/80 gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="w-2 h-2 rounded-full bg-primary" />
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground font-mono">
                            Operational Workspace &middot; Detection Engine
                        </span>
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-foreground font-sans">
                        Workspace Overview
                    </h1>
                    <p className="text-xs sm:text-sm text-muted-foreground mt-0.5">
                        Central console for media authenticity verification, analysis history, and forensic tools.
                    </p>
                </div>

                <div className="flex items-center gap-2.5 w-full sm:w-auto">
                    <button
                        type="button"
                        onClick={loadData}
                        disabled={loading}
                        className="px-3.5 py-2 rounded-xl bg-card border border-border text-xs font-semibold text-foreground hover:bg-muted transition-colors flex items-center gap-1.5 shadow-sm"
                        title="Refresh analysis state"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
                        <span>Sync</span>
                    </button>
                    <Link
                        href="/detect"
                        className="flex-1 sm:flex-none px-5 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-md flex items-center justify-center gap-2"
                    >
                        <PlusCircle className="w-4 h-4" />
                        <span>Start New Analysis</span>
                    </Link>
                </div>
            </div>

            {/* KPI Summary Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-2 hover:border-foreground/20 transition-colors">
                    <div className="flex items-center justify-between text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        <span>Total Scans</span>
                        <History className="w-4 h-4 text-primary" />
                    </div>
                    <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">
                        {error ? "Unavailable" : totalScans}
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                        Authenticated verification events
                    </p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-2 hover:border-foreground/20 transition-colors">
                    <div className="flex items-center justify-between text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        <span>Synthetic Detected</span>
                        <AlertTriangle className="w-4 h-4 text-destructive" />
                    </div>
                    <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-destructive">
                        {error ? "Unavailable" : fakeCount}
                    </div>
                    <p className="text-[11px] text-destructive/90 font-medium">
                        {totalScans > 0 ? `${Math.round((fakeCount / totalScans) * 100)}% manipulation rate` : "0% manipulation rate"}
                    </p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-2 hover:border-foreground/20 transition-colors">
                    <div className="flex items-center justify-between text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        <span>Verified Authentic</span>
                        <CheckCircle2 className="w-4 h-4 text-green-600" />
                    </div>
                    <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-green-600">
                        {error ? "Unavailable" : realCount}
                    </div>
                    <p className="text-[11px] text-green-600/90 font-medium">
                        {totalScans > 0 ? `${Math.round((realCount / totalScans) * 100)}% camera frames` : "0% genuine frames"}
                    </p>
                </div>

                <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-2 hover:border-foreground/20 transition-colors">
                    <div className="flex items-center justify-between text-xs font-bold text-muted-foreground uppercase tracking-wider">
                        <span>Neural Certainty</span>
                        <Activity className="w-4 h-4 text-primary" />
                    </div>
                    <div className="text-3xl sm:text-4xl font-black font-mono tracking-tight text-foreground">
                        {error ? "Unavailable" : `${avgConfidence}%`}
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                        Average model confidence score
                    </p>
                </div>
            </div>

            {/* Main Grid: Recent Scans + Quick Tools */}
            <div className="grid lg:grid-cols-12 gap-8 items-start">
                {/* Left (8 cols): Recent Analysis Activity */}
                <div className="lg:col-span-8 space-y-4">
                    <div className="flex items-center justify-between">
                        <h2 className="text-base font-bold text-foreground tracking-tight flex items-center gap-2">
                            <History className="w-4 h-4 text-primary" />
                            <span>Recent Forensic Verifications</span>
                        </h2>
                        {totalScans > 0 && !error && (
                            <Link href="/history" className="text-xs font-semibold text-primary hover:underline flex items-center gap-1">
                                <span>View History</span>
                                <ArrowRight className="w-3.5 h-3.5" />
                            </Link>
                        )}
                    </div>

                    {error && !loading ? (
                        <div className="p-8 sm:p-12 rounded-3xl bg-destructive/10 border border-destructive/20 text-center space-y-4 shadow-sm">
                            <div className="w-16 h-16 rounded-full bg-destructive/20 text-destructive flex items-center justify-center mx-auto">
                                <HardDrive className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-base font-bold text-foreground">
                                    Failed to Load Scan History
                                </h3>
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
                                <span>Retry Connection</span>
                            </button>
                        </div>
                    ) : totalScans > 0 ? (
                        <div className="bg-card border border-border rounded-3xl divide-y divide-border/60 overflow-hidden shadow-sm">
                            {history.slice(0, 5).map((row) => {
                                const isFake = row.resultLabel?.toLowerCase() === "fake";
                                const status = isFake ? "AI Generated" : "Real";
                                const fileName = row.imagePath ? row.imagePath.replace(/^.*[\\/]/, '') : "unnamed_frame.png";

                                return (
                                    <div
                                        key={row.scanId}
                                        className="p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 hover:bg-muted/30 transition-colors group"
                                    >
                                        <div className="flex items-center gap-3.5 min-w-0">
                                            <span className="font-mono text-xs font-bold text-muted-foreground shrink-0">
                                                #{row.scanId}
                                            </span>
                                            <div className="min-w-0">
                                                <p className="text-sm font-semibold text-foreground truncate group-hover:text-primary transition-colors" title={fileName}>
                                                    {fileName}
                                                </p>
                                                <p className="text-xs text-muted-foreground font-mono mt-0.5">
                                                    {row.scanTimestamp ? new Date(row.scanTimestamp).toLocaleString() : "Recent"}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4 w-full sm:w-auto justify-between sm:justify-end">
                                            <ResultBadge status={status} size="sm" />
                                            <span className="font-mono text-xs font-bold text-foreground">
                                                {row.confidenceScore != null ? `${row.confidenceScore.toFixed(1)}%` : "N/A"}
                                            </span>
                                            <Link
                                                href={`/report/${row.scanId}`}
                                                className="px-3 py-1.5 rounded-xl bg-muted hover:bg-muted/80 text-foreground text-xs font-semibold transition-colors flex items-center gap-1"
                                            >
                                                <span>Report</span>
                                                <ArrowRight className="w-3 h-3" />
                                            </Link>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="p-8 sm:p-12 rounded-3xl bg-card border border-border shadow-sm text-center space-y-4">
                            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto text-muted-foreground">
                                <HardDrive className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-base font-bold text-foreground">
                                    No Scans Recorded in Current Session
                                </h3>
                                <p className="text-xs sm:text-sm text-muted-foreground max-w-md mx-auto">
                                    Upload an image in New Analysis to generate your first Error Level Analysis and SigLIP ViT forensic report.
                                </p>
                            </div>
                            <Link
                                href="/detect"
                                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-foreground text-background text-xs font-bold uppercase tracking-wider hover:bg-foreground/90 transition-all shadow-sm"
                            >
                                <Sparkles className="w-4 h-4" />
                                <span>Analyze Your First Image</span>
                            </Link>
                        </div>
                    )}
                </div>

                {/* Right (4 cols): Quick Tool Shortcuts & Engine Status */}
                <div className="lg:col-span-4 space-y-6">
                    {/* Tool Shortcuts */}
                    <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-3">
                        <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                            Forensic Toolkit
                        </h3>

                        <div className="space-y-2">
                            <Link
                                href="/detect"
                                className="p-3.5 rounded-2xl bg-muted/40 hover:bg-muted border border-border/50 flex items-center justify-between transition-colors group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-xl bg-primary/10 text-primary group-hover:scale-105 transition-transform">
                                        <ScanSearch className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-foreground">Image Authenticity</p>
                                        <p className="text-[11px] text-muted-foreground">SigLIP ViT + ELA Engine</p>
                                    </div>
                                </div>
                                <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
                            </Link>

                            <Link
                                href="/protection"
                                className="p-3.5 rounded-2xl bg-muted/40 hover:bg-muted border border-border/50 flex items-center justify-between transition-colors group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-xl bg-primary/10 text-primary group-hover:scale-105 transition-transform">
                                        <Shield className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-foreground">Likeness Protection</p>
                                        <p className="text-[11px] text-muted-foreground">Biometric face-clone index</p>
                                    </div>
                                </div>
                                <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
                            </Link>

                            <Link
                                href="/admin"
                                className="p-3.5 rounded-2xl bg-muted/40 hover:bg-muted border border-border/50 flex items-center justify-between transition-colors group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-xl bg-primary/10 text-primary group-hover:scale-105 transition-transform">
                                        <Activity className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-foreground">Operations Center</p>
                                        <p className="text-[11px] text-muted-foreground">Cluster load & threat feed</p>
                                    </div>
                                </div>
                                <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
                            </Link>

                            <Link
                                href="/history"
                                className="p-3.5 rounded-2xl bg-muted/40 hover:bg-muted border border-border/50 flex items-center justify-between transition-colors group"
                            >
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-xl bg-primary/10 text-primary group-hover:scale-105 transition-transform">
                                        <FileText className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-foreground">Ledger Export</p>
                                        <p className="text-[11px] text-muted-foreground">CSV & JSON download</p>
                                    </div>
                                </div>
                                <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
                            </Link>
                        </div>
                    </div>

                    {/* Engine Health Card */}
                    <div className="p-5 rounded-3xl bg-card border border-border shadow-sm space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                                Inference Health
                            </span>
                            <span className="flex items-center gap-1 text-[11px] font-mono font-bold text-green-600">
                                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                                100% OPERATIONAL
                            </span>
                        </div>

                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between text-muted-foreground">
                                <span>Vision Transformer (ViT)</span>
                                <span className="font-mono text-foreground font-semibold">Active</span>
                            </div>
                            <div className="flex justify-between text-muted-foreground">
                                <span>Error Level Analysis (ELA)</span>
                                <span className="font-mono text-foreground font-semibold">Active</span>
                            </div>
                            <div className="flex justify-between text-muted-foreground">
                                <span>Target Image Constraint</span>
                                <span className="font-mono text-foreground font-semibold">512px client scaled</span>
                            </div>
                            <div className="flex justify-between text-muted-foreground">
                                <span>Mean Ingest Latency</span>
                                <span className="font-mono text-foreground font-semibold">~140ms</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
