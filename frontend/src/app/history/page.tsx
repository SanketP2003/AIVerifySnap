"use client";

import { Search, Filter, ArrowRight, Loader2 } from "lucide-react";
import { ResultBadge } from "@/components/shared/ResultBadge";
import Link from "next/link";
import { useState, useEffect } from "react";
import { detectionApi, DetectionHistoryItem } from "@/lib/api";

export default function HistoryPage() {
    const [filter, setFilter] = useState("All");
    const [history, setHistory] = useState<DetectionHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        async function fetchHistory() {
            try {
                setLoading(true);
                const data = await detectionApi.getHistory();
                setHistory(data);
            } catch (err) {
                console.error("Failed to fetch history:", err);
                setError("Failed to load scan history. Make sure the backend is running.");
            } finally {
                setLoading(false);
            }
        }
        fetchHistory();
    }, []);

    const mapResult = (label: string): "Real" | "AI Generated" | "Suspicious" => {
        if (label.toLowerCase() === "fake") return "AI Generated";
        if (label.toLowerCase() === "real") return "Real";
        return "Suspicious";
    };

    const filteredHistory = history.filter((item) => {
        const displayResult = mapResult(item.resultLabel);
        const matchesFilter = filter === "All" || displayResult === filter;
        const matchesSearch = searchQuery === "" ||
            item.scanId.toString().includes(searchQuery) ||
            item.imagePath?.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesFilter && matchesSearch;
    });

    if (!mounted) {
        return (
            <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 flex flex-col items-center justify-center space-y-3">
                <Loader2 className="w-6 h-6 text-accent animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-[calc(100vh-4rem)] max-w-[1120px] mx-auto px-6 py-10 space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-heading-lg font-serif text-foreground">Scan History</h1>
                    <p className="text-caption text-muted-foreground mt-1">Review your previously analyzed images and reports.</p>
                </div>
                <div className="flex items-center gap-3 w-full md:w-auto">
                    <div className="relative flex-1 md:w-56">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" strokeWidth={1.8} />
                        <input
                            type="text"
                            placeholder="Search by ID or filename..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-4 py-2.5 rounded-lg border border-border bg-card text-[14px] placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent/40 transition-all"
                        />
                    </div>
                    <button className="flex items-center gap-2 px-4 py-2.5 border border-border rounded-lg bg-card hover:bg-muted/50 transition-colors text-[13px] font-medium text-muted-foreground">
                        <Filter className="w-3.5 h-3.5" strokeWidth={1.8} /> Filter
                    </button>
                </div>
            </div>

            {/* Filter pills */}
            <div className="flex flex-wrap gap-2">
                {["All", "AI Generated", "Real", "Suspicious"].map(f => (
                    <button key={f} onClick={() => setFilter(f)}
                        className={`px-4 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 ${
                            filter === f
                                ? "bg-foreground text-background"
                                : "bg-muted/60 text-muted-foreground hover:bg-muted"
                        }`}>
                        {f}
                    </button>
                ))}
            </div>

            {/* States */}
            {loading && (
                <div className="flex flex-col items-center justify-center py-24 space-y-3">
                    <Loader2 className="w-6 h-6 text-accent animate-spin" />
                    <p className="text-caption text-muted-foreground">Loading scan history...</p>
                </div>
            )}
            {error && (
                <div className="p-5 rounded-xl bg-destructive/5 border border-destructive/15 text-center">
                    <p className="text-[14px] font-medium text-destructive">{error}</p>
                </div>
            )}

            {/* Table */}
            {!loading && !error && (
                <div className="card-elevated rounded-xl overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-[14px] text-left">
                            <thead>
                                <tr style={{ background: "hsl(var(--muted) / 0.4)" }}>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">Scan ID</th>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">File</th>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground">Result</th>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground hidden sm:table-cell">Confidence</th>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground hidden md:table-cell">Date</th>
                                    <th className="px-5 py-3.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y" style={{ borderColor: "hsl(var(--border) / 0.4)" }}>
                                {filteredHistory.length === 0 ? (
                                    <tr>
                                        <td colSpan={6} className="px-5 py-16 text-center text-muted-foreground text-caption">
                                            {history.length === 0 ? "No scans yet. Go to the Detect page to analyze an image." : "No results match your filter."}
                                        </td>
                                    </tr>
                                ) : (
                                    filteredHistory.map((row) => {
                                        const displayResult = mapResult(row.resultLabel);
                                        const conf = row.confidenceScore ?? 0;
                                        return (
                                            <tr key={row.scanId} className="hover:bg-muted/20 transition-colors">
                                                <td className="px-5 py-3.5 whitespace-nowrap">
                                                    <span className="font-mono text-[12px] text-muted-foreground">#{row.scanId}</span>
                                                </td>
                                                <td className="px-5 py-3.5 whitespace-nowrap">
                                                    <span className="text-[13px] truncate max-w-[180px] inline-block">{row.imagePath || "—"}</span>
                                                </td>
                                                <td className="px-5 py-3.5 whitespace-nowrap">
                                                    <ResultBadge status={displayResult} />
                                                </td>
                                                <td className="px-5 py-3.5 whitespace-nowrap hidden sm:table-cell">
                                                    <div className="flex items-center gap-2.5">
                                                        <div className="w-14 h-1.5 bg-muted rounded-full overflow-hidden">
                                                            <div className={`h-full rounded-full ${conf > 75 ? 'bg-destructive' : conf > 40 ? 'bg-warning' : 'bg-success'}`}
                                                                style={{ width: `${conf}%` }} />
                                                        </div>
                                                        <span className="text-[13px] font-medium tabular-nums">{conf.toFixed(1)}%</span>
                                                    </div>
                                                </td>
                                                <td className="px-5 py-3.5 whitespace-nowrap text-muted-foreground text-[13px] hidden md:table-cell">
                                                    {row.scanTimestamp ? new Date(row.scanTimestamp).toLocaleDateString() : "—"}
                                                </td>
                                                <td className="px-5 py-3.5 whitespace-nowrap text-right">
                                                    <Link href={`/report/${row.scanId}`}
                                                        className="inline-flex items-center gap-1 text-accent hover:text-accent/80 text-[13px] font-medium transition-colors">
                                                        View <ArrowRight className="w-3.5 h-3.5" />
                                                    </Link>
                                                </td>
                                            </tr>
                                        );
                                    })
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
